"""Fake SmartLogger phone, for testing the viewer and the fusion filter without a phone.

The simulated person walks an ellipse (~15 x 8 m) at ~1.4 m/s with stops, holding the
phone tilted in front of them. Linear acceleration is the true second derivative of the
trajectory (plus walking bounce, noise and a bias) expressed in the device frame, so it is
consistent with the motion. GPS comes at 1 Hz with a slowly wandering error, the
barometer is noisy.

  python simulate_sender.py                 # stream to 127.0.0.1:5005 in real time
  python simulate_sender.py --gps-rate 0.5  # GPS every 2 s
"""

import argparse
import json
import math
import random
import socket
import time

LAT0, LON0, ALT0 = 35.681236, 139.767125, 40.0
G = 9.80665


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qaxis(ax, ay, az, ang):
    s = math.sin(ang / 2)
    return [math.cos(ang / 2), ax * s, ay * s, az * s]


def qrot(q, v):
    w, x, y, z = q
    p = qmul(qmul(q, [0.0, *v]), [w, -x, -y, -z])
    return p[1:]


def qinv_rot(q, v):
    w, x, y, z = q
    return qrot([w, -x, -y, -z], v)


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def speed_profile(t):
    """Walking speed [m/s]: stand 5 s, walk 30 s, stop 6 s, walk 25 s, ... (period 70 s)."""
    tt = t % 70.0
    if tt < 5 or 35 <= tt < 41:
        return 0.0
    if tt < 35:
        return 1.4 * smoothstep(5, 7, tt) * (1 - smoothstep(33, 35, tt))
    return 1.4 * smoothstep(41, 43, tt) * (1 - smoothstep(68, 70, tt))


class Trajectory:
    """Precomputed arc length so that position is a smooth function of time."""

    def __init__(self, duration, h=0.005):
        self.h = h
        n = int(duration / h) + 400
        self.s = [0.0] * n
        for i in range(1, n):
            self.s[i] = self.s[i - 1] + speed_profile((i - 0.5) * h) * h

    def arc(self, t):
        i = t / self.h
        i0 = int(i)
        f = i - i0
        return self.s[i0] * (1 - f) + self.s[i0 + 1] * f

    def pos(self, t):
        th = self.arc(t) / 11.5              # ellipse with mean radius ~11.5 m
        walking = speed_profile(t) > 0.05
        bounce = 0.03 * math.sin(2 * math.pi * 1.9 * t) * min(speed_profile(t) / 1.4, 1.0)
        climb = 1.5 * (1 - math.cos(2 * math.pi * t / 140.0))  # slow 3 m up/down (ramp)
        return [15.0 * math.sin(th), 8.0 * (1 - math.cos(th)), climb + bounce], walking

    def accel(self, t, d=0.01):
        p0, _ = self.pos(t - d)
        p1, _ = self.pos(t)
        p2, _ = self.pos(t + d)
        return [(p2[i] - 2 * p1[i] + p0[i]) / (d * d) for i in range(3)]

    def vel(self, t, d=0.01):
        p0, _ = self.pos(t - d)
        p2, _ = self.pos(t + d)
        return [(p2[i] - p0[i]) / (2 * d) for i in range(3)]


def generate(duration, rate=50.0, gps_rate=1.0, seed=1, t0_ms=None, ar=False):
    """Yield (true_pos_enu, packet_dict) at `rate` Hz with simulated timestamps."""
    rnd = random.Random(seed)
    traj = Trajectory(duration + 5)
    t0_ms = int(time.time() * 1000) if t0_ms is None else t0_ms
    bias_dev = [0.04, -0.06, 0.05]       # accelerometer bias in device frame
    gps_err = [0.0, 0.0]
    next_gps = 0.0
    gps = None
    n = int(duration * rate)
    heading = 0.0
    # fake ARCore: its world frame is Y-up with an arbitrary yaw; 2 % scale error + slow drift
    ar_yaw = math.radians(rnd.uniform(0, 360))
    C = qmul(qaxis(0, 0, 1, ar_yaw), qaxis(1, 0, 0, math.pi / 2))   # ARCore world -> ENU
    Cinv = [C[0], -C[1], -C[2], -C[3]]
    ar_drift = [0.0, 0.0, 0.0]
    ar_n = 0
    for k in range(n):
        t = k / rate
        p, walking = traj.pos(t)
        v = traj.vel(t)
        a = traj.accel(t)
        if math.hypot(v[0], v[1]) > 0.2:
            heading = math.atan2(-v[0], v[1])   # rotation about Up; 0 = facing North
        q = qmul(qaxis(0, 0, 1, heading), qaxis(1, 0, 0, math.radians(40)))
        q = qmul(q, qaxis(0, 1, 0, 0.05 * math.sin(t * 1.7)))
        lin_dev = qinv_rot(q, a)
        lin_dev = [lin_dev[i] + bias_dev[i] + rnd.gauss(0, 0.08) for i in range(3)]
        acc_dev = [lin_dev[i] + qinv_rot(q, [0, 0, G])[i] for i in range(3)]
        gyr = [rnd.gauss(0, 0.02), rnd.gauss(0, 0.02),
               rnd.gauss(0, 0.02) + (0.15 if walking else 0.0) * math.sin(t * 3)]
        if walking:
            gyr = [g + 0.3 * math.sin(2 * math.pi * 1.9 * t + i) for i, g in enumerate(gyr)]
        pressure = 1013.25 * (1 - (p[2] + ALT0) / 44330.0) ** 5.255 + rnd.gauss(0, 0.02)
        t_ms = t0_ms + int(t * 1000)
        if t >= next_gps:
            next_gps += 1.0 / gps_rate
            # slowly wandering GPS error (random walk with pull to 0), sigma ~2.5 m
            gps_err = [0.9 * e + rnd.gauss(0, 1.1) for e in gps_err]
            e, nn = p[0] + gps_err[0], p[1] + gps_err[1]
            spd = math.hypot(v[0], v[1]) + abs(rnd.gauss(0, 0.1))
            gps = {
                "lat": LAT0 + math.degrees(nn / 6378137.0),
                "lon": LON0 + math.degrees(e / (6378137.0 * math.cos(math.radians(LAT0)))),
                "alt": ALT0 + p[2] + rnd.gauss(0, 3.0), "acc": 4.0,
                "spd": spd,
                "brg": (math.degrees(math.atan2(v[0], v[1])) % 360) if spd > 0.5 else None,
                "prov": "sim", "t": t_ms,
            }
        pkt = {"v": 1, "seq": k, "t": t_ms, "q": q, "acc": acc_dev, "gyr": gyr,
               "mag": [12.0, 25.0, -40.0], "lin": lin_dev, "p": pressure}
        if gps:
            pkt["gps"] = gps
        if ar and k % max(1, int(rate / 30)) == 0:
            ar_drift = [d + rnd.gauss(0, 0.002) for d in ar_drift]
            pe = [p[i] * 1.02 + ar_drift[i] for i in range(3)]
            ar_n += 1
            pkt["ar"] = {"st": "TRACKING", "n": ar_n, "ts": t * 1000.0,
                         "p": qrot(Cinv, pe), "q": qmul(Cinv, q)}
        elif ar and k > 0:
            pkt["ar"] = last_ar
        if ar:
            last_ar = pkt.get("ar")
        yield p, pkt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--rate", type=float, default=50)
    ap.add_argument("--gps-rate", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=600, help="seconds")
    ap.add_argument("--ar", action="store_true", help="also send fake ARCore poses")
    ap.add_argument("--replay", default=None, help="resend a JSONL file recorded with viewer.py --record")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.time()
    if args.replay:
        t_first = None
        with open(args.replay, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line).get("t", 0) / 1000.0
                t_first = t if t_first is None else t_first
                delay = start + (t - t_first) - time.time()
                if delay > 0:
                    time.sleep(delay)
                sock.sendto(line.encode(), (args.host, args.port))
        return
    for k, (_, pkt) in enumerate(generate(args.duration, args.rate, args.gps_rate, ar=args.ar)):
        delay = start + k / args.rate - time.time()
        if delay > 0:
            time.sleep(delay)
        sock.sendto(json.dumps(pkt).encode(), (args.host, args.port))


if __name__ == "__main__":
    main()
