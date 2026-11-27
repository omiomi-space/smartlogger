"""GPS / accelerometer / barometer fusion with a Kalman filter (dead reckoning between GPS fixes).

World frame is ENU (X=East, Y=North, Z=Up), same as Android's rotation vector.

State (9):  p_E p_N p_U | v_E v_N v_U | b_E b_N b_U
  p : position relative to the start [m]
  v : velocity [m/s]
  b : bias of the world-frame linear acceleration [m/s^2]

Prediction (every IMU packet, ~50 Hz):
  a_world = R(q) * lin_acc_device                (gravity already removed by Android)
  p += v dt + 1/2 (a_world - b) dt^2
  v += (a_world - b) dt

Updates:
  GPS position (E, N)      sigma = reported accuracy (min 2 m), robust gating
  GPS velocity (E, N)      from speed + bearing
  barometric height (U)    relative to the first pressure sample
  ZUPT (v = 0)             when the phone is detected to be still (gyro + accel quiet)
  PDR step velocity        on every detected step: step length / step period along the
                           direction the phone is pointing (pedestrian dead reckoning)
  standing (v_EN = 0)      no step for a while (hand-held phone is never perfectly still)
  weak velocity damping    when GPS is stale and PDR is off, so integration cannot run away
  camera VIO position      ARCore pose (if the app sends "ar"): displacement since an anchor
                           (re-anchored every 2 s, on tracking jumps and tracking loss), rotated
                           from ARCore's world frame into ENU; replaces PDR while tracking.
                           Being anchored, it keeps the full displacement across Wi-Fi gaps.

ARCore world -> ENU yaw: from the app's magnetometer-free game rotation vector ("qg", stable
indoors, follows ARCore re-localisation jumps quickly) plus a very slow magnetometer term for
North; without "qg" the magnetometer-based estimate is smoothed heavily instead.

Double-integrating a hand-held phone's acceleration alone drifts within seconds, so for
walking the step-based PDR velocity is what carries the position between GPS fixes.

Only numpy is needed; this module is independent of the 3D viewer so it can be tested offline.
"""

import math
from collections import deque

import numpy as np

EARTH_R = 6378137.0


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    # v' = v + 2w (u x v) + 2 u x (u x v), u = (x, y, z)
    ux, uy, uz = x, y, z
    vx, vy, vz = v
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    ccx, ccy, ccz = uy * cz - uz * cy, uz * cx - ux * cz, ux * cy - uy * cx
    return np.array([vx + 2 * (w * cx + ccx), vy + 2 * (w * cy + ccy), vz + 2 * (w * cz + ccz)])


class FusionParams:
    accel_noise = 0.6          # white noise on world linear accel [m/s^2]
    bias_walk = 0.02           # accel bias random walk [m/s^2 / sqrt(s)]
    gps_min_sigma = 2.0        # floor for GPS horizontal sigma [m]
    gps_vel_sigma = 0.3        # GPS velocity sigma [m/s]
    baro_sigma = 0.4           # barometric height sigma [m]
    baro_every = 5             # use every n-th pressure sample
    zupt_sigma = 0.03          # zero-velocity update sigma [m/s]
    still_gyro = 0.12          # rad/s threshold for "still"
    still_acc = 0.35           # m/s^2 threshold for "still"
    still_window = 0.4         # seconds the phone must be quiet
    still_gps_factor = 3.0     # GPS position sigma multiplier while still
    stale_gps = 3.0            # min. seconds without GPS before damping velocity (x2.5 GPS interval)
    damping_sigma = 1.5        # pseudo-measurement v=0 sigma while GPS is stale [m/s]
    gate_chi2 = 13.8           # 99.9% for 2 dof; larger innovations get their R inflated
    gps_vel_max_acc = 10.0     # ignore GPS speed when the fix is worse than this [m] (indoors)
    # pedestrian dead reckoning
    pdr = True
    step_lp_hz = 4.0           # low-pass of vertical world accel for step detection
    step_thresh = 0.8          # min. peak of vertical accel for a step [m/s^2]
    step_min_interval = 0.33   # s
    step_k = 0.45              # Weinberg step length: K * (amax - amin)^(1/4)
    step_len_min = 0.3
    step_len_max = 1.1
    pdr_vel_sigma = 0.25       # m/s
    step_timeout = 1.2         # s without a step -> standing
    standing_sigma = 0.15      # m/s
    # camera visual-inertial odometry (ARCore)
    ar = True
    ar_anchor_period = 2.0     # s between re-anchoring
    ar_pos_sigma = 0.03        # m, + ar_pos_sigma_rel * displacement since the anchor
    ar_pos_sigma_rel = 0.02
    ar_max_speed = 4.0         # m/s between two camera frames; faster = re-localisation jump
    ar_timeout = 1.0           # s without a tracking AR pose -> fall back to PDR
    ar_yaw_alpha_game = 0.05   # EMA per AR frame, ARCore->game yaw (stable, fast)
    ar_yaw_alpha_mag = 0.0005  # EMA per AR frame (~70 s), ARCore->ENU yaw from magnetometer (no "qg")
    ar_yaw_fast_frames = 30    # after a jump, adapt quickly for this many frames
    north_tau = 60.0           # s, smoothing of game->ENU yaw (magnetometer, disturbed indoors)


class Fusion:
    def __init__(self, params=None):
        self.prm = params or FusionParams()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        self.x = np.zeros(9)
        self.P = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05]) ** 2
        self.last_t = None
        self.origin = None          # (lat0, lon0)
        self.origin_offset = np.zeros(2)
        self.p0 = None              # reference pressure [hPa]
        self.gps_alt0 = None
        self.last_gps_t = None
        self.last_gps_wall = None   # packet time (s) of the last GPS update
        self.gps_points = deque(maxlen=2000)   # raw GPS fixes in ENU, for display
        self.still = False
        self._quiet = deque()
        self._baro_n = 0
        self.n_gps = 0
        self.gps_interval = 1.0     # running estimate of the GPS fix interval [s]
        self.jump = np.zeros(3)     # position correction applied by GPS updates (for smooth display)
        self.a_world = np.zeros(3)
        # PDR
        self._lp = 0.0
        self._lp1 = 0.0
        self._lp2 = 0.0
        self._amax = -1e9
        self._amin = 1e9
        self._fwd = np.zeros(2)
        self.last_step_t = None
        self.steps = 0
        self.step_len = 0.0
        self.heading = None          # deg, 0 = North, 90 = East
        self._gps_moving_until = -1.0
        # AR
        self._ar_last_n = None
        self._ar_prev = None         # (t_ar, p_ar) of the previous camera frame
        self._anc = None             # (filter position, p_ar, t_ar) at the anchor
        self._yaw_a = None           # smoothed (cos, sin): ARCore->game (with qg) or ARCore->ENU
        self._yaw_n = None           # smoothed (cos, sin): game->ENU (magnetometer north)
        self._yaw_fast = 0
        self.ar_jumps = 0
        self.n_ar_frames = 0
        self.ar_last_t = None        # packet time of the last used AR pose
        self.ar_state = None

    # ------------------------------------------------------------------ helpers
    def enu(self, lat, lon):
        lat0, lon0 = self.origin
        e = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
        n = math.radians(lat - lat0) * EARTH_R
        return np.array([e, n]) + self.origin_offset

    def _update(self, H, z, R, gate=True):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        if gate:
            d2 = float(y @ np.linalg.solve(S, y))
            lim = self.prm.gate_chi2 * (len(z) / 2.0)
            if d2 > lim:  # outlier-ish: trust the measurement less instead of dropping it
                R = R * (d2 / lim)
                S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form

    def _predict(self, dt, a):
        F = np.eye(9)
        I3 = np.eye(3)
        F[0:3, 3:6] = I3 * dt
        F[0:3, 6:9] = -0.5 * I3 * dt * dt
        F[3:6, 6:9] = -I3 * dt
        u = np.concatenate([0.5 * a * dt * dt, a * dt, np.zeros(3)])
        self.x = F @ self.x + u
        G = np.vstack([0.5 * I3 * dt * dt, I3 * dt, np.zeros((3, 3))])
        Q = G @ G.T * self.prm.accel_noise ** 2
        Q[6:9, 6:9] += I3 * self.prm.bias_walk ** 2 * dt
        self.P = F @ self.P @ F.T + Q

    def _detect_still(self, t, pkt, lin_norm):
        gyr = pkt.get("gyr")
        g = math.sqrt(sum(v * v for v in gyr)) if gyr else 0.0
        quiet = g < self.prm.still_gyro and lin_norm < self.prm.still_acc
        if not quiet:
            self._quiet.clear()
            self.still = False
            return
        self._quiet.append(t)
        while self._quiet and t - self._quiet[0] > self.prm.still_window * 2:
            self._quiet.popleft()
        self.still = (t - self._quiet[0]) >= self.prm.still_window

    def _pdr(self, t, dt, q, a):
        """Step detection on vertical world accel; returns True if a step was detected."""
        alpha = 1.0 - math.exp(-2 * math.pi * self.prm.step_lp_hz * dt) if dt > 0 else 0.0
        self._lp2, self._lp1 = self._lp1, self._lp
        self._lp += alpha * (a[2] - self._lp)
        self._amax = max(self._amax, self._lp)
        self._amin = min(self._amin, self._lp)
        # forward direction: horizontal part of (device +Y) + (device -Z); works flat or upright
        y = quat_rotate(q, (0.0, 1.0, 0.0))
        z = quat_rotate(q, (0.0, 0.0, -1.0))
        self._fwd += np.array([y[0] + z[0], y[1] + z[1]])
        peak = self._lp1 > self._lp2 and self._lp1 >= self._lp and self._lp1 > self.prm.step_thresh
        if not peak or (self.last_step_t is not None and t - self.last_step_t < self.prm.step_min_interval):
            return False
        period = 0.6 if self.last_step_t is None else min(max(t - self.last_step_t, 0.35), 1.0)
        rng = max(self._amax - self._amin, 0.0)
        L = min(max(self.prm.step_k * rng ** 0.25, self.prm.step_len_min), self.prm.step_len_max)
        n = np.linalg.norm(self._fwd)
        if n > 1e-6:
            d = self._fwd / n
            self.heading = math.degrees(math.atan2(d[0], d[1])) % 360
            H = np.zeros((2, 9)); H[0, 3] = 1; H[1, 4] = 1
            self._update(H, d * (L / period), np.eye(2) * self.prm.pdr_vel_sigma ** 2)
        self.steps += 1
        self.step_len = L
        self.last_step_t = t
        self._amax, self._amin = -1e9, 1e9
        self._fwd[:] = 0
        return True

    @staticmethod
    def _qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

    @staticmethod
    def _yaw_cs(C):
        """(cos, sin) of the rotation about Up contained in C, or None if C is not ~vertical."""
        up = quat_rotate(C, (0.0, 1.0, 0.0))
        if up[2] < 0.9:
            return None
        v = quat_rotate(C, (1.0, 0.0, 0.0))
        n = math.hypot(v[0], v[1]) or 1.0
        return np.array([v[0] / n, v[1] / n])

    @staticmethod
    def _ema_cs(cur, new, alpha):
        if cur is None:
            return new
        m = cur + alpha * (new - cur)
        return m / (np.linalg.norm(m) or 1.0)

    def _north(self, q, qg, dt):
        """Slowly track the game->ENU yaw (game frame has arbitrary heading, ENU is North-up)."""
        if q is None or qg is None or dt <= 0:
            return
        w, x, y, z = qg
        D = self._qmul(q, (w, -x, -y, -z))        # game world -> ENU, rotation about Up
        v = quat_rotate(D, (1.0, 0.0, 0.0))
        n = math.hypot(v[0], v[1])
        if n < 0.5:
            return
        cs = np.array([v[0] / n, v[1] / n])
        self._yaw_n = self._ema_cs(self._yaw_n, cs, min(1.0, dt / self.prm.north_tau)
                                   if self._yaw_n is not None else 1.0)

    def _ar_rot(self):
        """(cos, sin) of the full ARCore->ENU yaw (axis swap applied separately)."""
        c1, s1 = self._yaw_a
        if self._yaw_n is None:
            return c1, s1
        c2, s2 = self._yaw_n
        return c1 * c2 - s1 * s2, s1 * c2 + c1 * s2

    def _ar_to_enu(self, v):
        """ARCore world vector (X, Y=up, Z) -> ENU."""
        c, s_ = self._ar_rot()
        x, y, z = v[0], -v[2], v[1]          # axis swap: Y-up -> Z-up
        return np.array([c * x - s_ * y, s_ * x + c * y, z])

    def _ar(self, t, q_enu, ar, qg=None):
        """Use an ARCore pose as an anchored position measurement. Returns True if applied."""
        self.ar_state = ar.get("st")
        if ar.get("st") != "TRACKING" or "p" not in ar or "q" not in ar:
            self._anc = None
            self._ar_prev = None
            return False
        if ar.get("n") == self._ar_last_n:
            return False
        self._ar_last_n = ar.get("n")
        t_ar = ar.get("ts", t * 1000.0) / 1000.0
        p_ar = np.array(ar["p"], dtype=float)
        w, x, y, z = ar["q"]
        q_ar_conj = (w, -x, -y, -z)

        # tracking jump (re-localisation) between consecutive frames -> new anchor, re-learn yaw
        jumped = False
        if self._ar_prev is not None:
            dt_ar = max(t_ar - self._ar_prev[0], 1 / 30.0)
            if np.linalg.norm(p_ar - self._ar_prev[1]) / dt_ar > self.prm.ar_max_speed and dt_ar < 0.5:
                jumped = True
                self.ar_jumps += 1
                self._yaw_fast = self.prm.ar_yaw_fast_frames
        self._ar_prev = (t_ar, p_ar)

        # ARCore -> game (if qg) or ARCore -> ENU yaw
        ref = qg if qg is not None else q_enu
        if ref is not None:
            cs = self._yaw_cs(self._qmul(ref, q_ar_conj))
            if cs is not None:
                if self._yaw_fast > 0:
                    alpha = 0.3
                    self._yaw_fast -= 1
                else:
                    alpha = self.prm.ar_yaw_alpha_game if qg is not None else self.prm.ar_yaw_alpha_mag
                if qg is None and self._yaw_a is not None and self.n_ar_frames < 90:
                    alpha = max(alpha, 0.05)    # learn the initial alignment quickly
                self._yaw_a = self._ema_cs(self._yaw_a, cs, alpha)
        if self._yaw_a is None:
            return False
        self.n_ar_frames += 1

        if self._anc is None or jumped:
            self._anc = (self.x[0:3].copy(), p_ar, t_ar)
            self.ar_last_t = t
            return False
        anc_x, anc_p, anc_t = self._anc
        d = self._ar_to_enu(p_ar - anc_p)
        z = anc_x + d
        sig = self.prm.ar_pos_sigma + self.prm.ar_pos_sigma_rel * float(np.linalg.norm(d))
        H = np.zeros((3, 9)); H[:, 0:3] = np.eye(3)
        self._update(H, z, np.eye(3) * sig * sig, gate=False)
        if t_ar - anc_t >= self.prm.ar_anchor_period:
            self._anc = (self.x[0:3].copy(), p_ar, t_ar)
        self.ar_last_t = t
        return True

    @property
    def ar_tracking(self):
        return self.ar_last_t is not None and self.last_t is not None and \
            self.last_t - self.ar_last_t < self.prm.ar_timeout

    @property
    def walking(self):
        return self.last_step_t is not None and self.last_t is not None and             self.last_t - self.last_step_t < self.prm.step_timeout

    # ------------------------------------------------------------------ main entry
    def process(self, pkt):
        """Feed one decoded packet. Returns True if the state changed."""
        t = pkt.get("t")
        if t is None:
            return False
        t = t / 1000.0
        if self.last_t is None:
            self.last_t = t
        dt = t - self.last_t
        if dt < 0:          # out-of-order packet
            return False
        dt = min(dt, 0.2)
        self.last_t = t

        q, lin = pkt.get("q"), pkt.get("lin")
        a = np.zeros(3)
        lin_norm = 0.0
        if q is not None and lin is not None:
            n = math.sqrt(sum(c * c for c in q)) or 1.0
            q = [c / n for c in q]
            a = quat_rotate(q, lin)
            lin_norm = float(np.linalg.norm(lin))
        self.a_world = a
        if dt > 0:
            self._predict(dt, a)

        # zero-velocity update
        self._detect_still(t, pkt, lin_norm)
        if self.still:
            H = np.zeros((3, 9)); H[:, 3:6] = np.eye(3)
            self._update(H, np.zeros(3), np.eye(3) * self.prm.zupt_sigma ** 2, gate=False)

        # camera VIO (ARCore)
        qg = pkt.get("qg")
        if qg is not None:
            n = math.sqrt(sum(c * c for c in qg)) or 1.0
            qg = [c / n for c in qg]
            self._north(q, qg, dt)
        ar = pkt.get("ar")
        if self.prm.ar and ar:
            self._ar(t, q, ar, qg)

        # pedestrian dead reckoning (skipped while camera tracking is available)
        if self.prm.pdr and q is not None and lin is not None and not self.ar_tracking:
            self._pdr(t, dt, q, a)
            if not self.walking and not self.still and t > self._gps_moving_until:
                H = np.zeros((2, 9)); H[0, 3] = 1; H[1, 4] = 1
                self._update(H, np.zeros(2), np.eye(2) * self.prm.standing_sigma ** 2, gate=False)

        # barometric height
        p = pkt.get("p")
        gps = pkt.get("gps")
        if p:
            if self.p0 is None:
                self.p0 = p
            self._baro_n += 1
            if self._baro_n % self.prm.baro_every == 0:
                h = 44330.0 * (1.0 - (p / self.p0) ** (1 / 5.255))
                H = np.zeros((1, 9)); H[0, 2] = 1
                self._update(H, np.array([h]), np.array([[self.prm.baro_sigma ** 2]]))
        elif gps and gps.get("alt") is not None and gps.get("t") != self.last_gps_t:
            if self.gps_alt0 is None:
                self.gps_alt0 = gps["alt"]
            H = np.zeros((1, 9)); H[0, 2] = 1
            self._update(H, np.array([gps["alt"] - self.gps_alt0]), np.array([[25.0]]))

        # GPS (only on a new fix)
        if gps and gps.get("t") != self.last_gps_t:
            self.last_gps_t = gps.get("t")
            self._gps_update(gps, t)

        # GPS stale -> softly pull velocity to zero so dead reckoning stays bounded
        stale = max(self.prm.stale_gps, 2.5 * self.gps_interval)
        if self.last_gps_wall is None or t - self.last_gps_wall > stale:
            if not self.still and not self.prm.pdr:
                H = np.zeros((2, 9)); H[0, 3] = 1; H[1, 4] = 1
                self._update(H, np.zeros(2), np.eye(2) * self.prm.damping_sigma ** 2, gate=False)
        return True

    def _gps_update(self, gps, t):
        if self.origin is None:
            self.origin = (gps["lat"], gps["lon"])
            self.origin_offset = self.x[0:2].copy()   # no jump when the first fix arrives
        z = self.enu(gps["lat"], gps["lon"])
        before = self.x[0:3].copy()
        self.gps_points.append((z[0], z[1], self.x[2]))
        sig = max(gps.get("acc") or 10.0, self.prm.gps_min_sigma)
        if self.still and self.n_gps > 3:
            sig *= self.prm.still_gps_factor   # standing still: don't let GPS wander drag us around
        H = np.zeros((2, 9)); H[0, 0] = 1; H[1, 1] = 1
        self._update(H, z, np.eye(2) * sig * sig, gate=self.n_gps > 0)
        spd, brg = gps.get("spd"), gps.get("brg")
        if spd is not None and (gps.get("acc") or 99.0) <= self.prm.gps_vel_max_acc:
            if spd > 1.0:
                self._gps_moving_until = t + 3.0   # e.g. in a vehicle: don't force "standing"
            if brg is not None and spd > 0.3:
                b = math.radians(brg)
                zv = np.array([spd * math.sin(b), spd * math.cos(b)])
            else:
                zv = np.zeros(2)
            H = np.zeros((2, 9)); H[0, 3] = 1; H[1, 4] = 1
            self._update(H, zv, np.eye(2) * self.prm.gps_vel_sigma ** 2)
        self.jump += self.x[0:3] - before
        if self._anc is not None:   # keep camera displacement measurements consistent with GPS
            self._anc = (self._anc[0] + (self.x[0:3] - before), self._anc[1], self._anc[2])
        if self.last_gps_wall is not None:
            gap = min(t - self.last_gps_wall, 30.0)
            self.gps_interval += 0.2 * (gap - self.gps_interval)
        self.n_gps += 1
        self.last_gps_wall = t

    # ------------------------------------------------------------------ outputs
    @property
    def pos(self):
        return self.x[0:3]

    @property
    def vel(self):
        return self.x[3:6]

    @property
    def bias(self):
        return self.x[6:9]

    def gps_age(self):
        if self.last_gps_wall is None or self.last_t is None:
            return None
        return self.last_t - self.last_gps_wall
