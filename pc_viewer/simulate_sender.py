"""Sends fake SmartLogger packets so the viewer can be tested without a phone.

The simulated phone walks a 20 m circle, rocks/tilts, and climbs 3 m in the middle.
"""

import argparse
import json
import math
import socket
import time


def quat_axis_angle(ax, ay, az, ang):
    s = math.sin(ang / 2)
    return [math.cos(ang / 2), ax * s, ay * s, az * s]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--rate", type=float, default=50)
    ap.add_argument("--duration", type=float, default=0, help="seconds (0 = forever)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    lat0, lon0 = 35.681236, 139.767125
    t0 = time.time()
    seq = 0
    gps_t = 0
    while True:
        t = time.time() - t0
        if args.duration and t > args.duration:
            break
        ang = t * 0.2
        e, n = 10 * math.cos(ang), 10 * math.sin(ang)
        up = 3 * max(0.0, math.sin(t * 0.1))
        heading = ang + math.pi  # facing direction of travel-ish
        q = qmul(quat_axis_angle(0, 0, 1, heading), quat_axis_angle(1, 0, 0, 0.6 + 0.3 * math.sin(t * 1.3)))
        q = qmul(q, quat_axis_angle(0, 1, 0, 0.25 * math.sin(t * 0.7)))
        pkt = {
            "v": 1, "seq": seq, "t": int(time.time() * 1000), "q": q,
            "acc": [0.3 * math.sin(t * 3), 9.81 * math.sin(0.6), 9.81 * math.cos(0.6)],
            "gyr": [0.4 * math.cos(t * 1.3), 0.2 * math.cos(t * 0.7), 0.2],
            "mag": [12.0, 25.0, -40.0],
            "lin": [0.5 * math.sin(t * 2), 0.0, 0.3 * math.cos(t * 2)],
            "p": 1013.25 * (1 - up / 44330.0) ** 5.255,
        }
        if int(t) != gps_t or seq == 0:
            gps_t = int(t)
        pkt["gps"] = {
            "lat": lat0 + math.degrees(n / 6378137.0),
            "lon": lon0 + math.degrees(e / (6378137.0 * math.cos(math.radians(lat0)))),
            "alt": 40 + up, "acc": 3.0, "spd": 2.0, "brg": 0.0, "prov": "sim", "t": gps_t * 1000,
        }
        sock.sendto(json.dumps(pkt).encode(), (args.host, args.port))
        seq += 1
        time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    main()
