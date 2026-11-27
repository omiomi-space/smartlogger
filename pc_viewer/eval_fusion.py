"""Offline accuracy check of the fusion filter against the simulator's ground truth.

Compares three estimates of horizontal position:
  gps-hold : last GPS fix (what you get without dead reckoning)
  gps-lerp : the viewer's old behaviour (exponential smoothing toward the last fix)
  ekf      : fusion.Fusion (accel dead reckoning + GPS + baro + ZUPT)

  python eval_fusion.py [--duration 300] [--gps-rate 1] [--seeds 5]
"""

import argparse
import math

import numpy as np

from fusion import Fusion
from simulate_sender import generate


def run(duration, gps_rate, seed, rate=50.0):
    f = Fusion()
    hold = None
    lerp = np.zeros(2)
    errs = {"gps-hold": [], "gps-lerp": [], "ekf": []}
    zerr = []
    origin_truth = None
    k_lerp = 1.0 - math.exp(-4.0 / rate)
    last_gps_t = None
    for p_true, pkt in generate(duration, rate, gps_rate, seed=seed, t0_ms=0):
        f.process(pkt)
        if f.origin is None:
            continue
        if origin_truth is None:
            # filter ENU = truth - (truth at first fix) + offset; align frames once
            origin_truth = np.array(p_true[:2]) - f.pos[:2]
        truth = np.array(p_true[:2]) - origin_truth
        g = pkt.get("gps")
        if g and g["t"] != last_gps_t:
            last_gps_t = g["t"]
            hold = f.enu(g["lat"], g["lon"])
        lerp += (hold - lerp) * k_lerp
        errs["gps-hold"].append(np.linalg.norm(hold - truth))
        errs["gps-lerp"].append(np.linalg.norm(lerp - truth))
        errs["ekf"].append(np.linalg.norm(f.pos[:2] - truth))
        zerr.append(abs(f.pos[2] - p_true[2]))
    return {k: (math.sqrt(np.mean(np.square(v))), np.percentile(v, 95)) for k, v in errs.items()}, \
        math.sqrt(np.mean(np.square(zerr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--gps-rate", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    tot = {}
    for s in range(args.seeds):
        res, z = run(args.duration, args.gps_rate, s + 1)
        line = "  ".join(f"{k}: rms {v[0]:5.2f} p95 {v[1]:5.2f}" for k, v in res.items())
        print(f"seed {s + 1}: {line}  | height rms {z:.2f} m")
        for k, v in res.items():
            tot.setdefault(k, []).append(v[0])
    print("mean rms [m]: " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in tot.items()))


if __name__ == "__main__":
    main()
