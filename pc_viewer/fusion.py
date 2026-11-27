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
  weak velocity damping    when GPS is stale, so pure dead reckoning cannot run away

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
            a = quat_rotate([c / n for c in q], lin)
            lin_norm = float(np.linalg.norm(lin))
        self.a_world = a
        if dt > 0:
            self._predict(dt, a)

        # zero-velocity update
        self._detect_still(t, pkt, lin_norm)
        if self.still:
            H = np.zeros((3, 9)); H[:, 3:6] = np.eye(3)
            self._update(H, np.zeros(3), np.eye(3) * self.prm.zupt_sigma ** 2, gate=False)

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
            if not self.still:
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
        if spd is not None:
            if brg is not None and spd > 0.3:
                b = math.radians(brg)
                zv = np.array([spd * math.sin(b), spd * math.cos(b)])
            else:
                zv = np.zeros(2)
            H = np.zeros((2, 9)); H[0, 3] = 1; H[1, 4] = 1
            self._update(H, zv, np.eye(2) * self.prm.gps_vel_sigma ** 2)
        self.jump += self.x[0:3] - before
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
