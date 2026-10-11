#!/usr/bin/env python3
"""Recompute every shipped estimate from the per-document sums and compare it with result.json.

usage: verify_data.py DATA_DIR [EXPECTED_PAIRS]   (exit 1, with a list of problems, on any mismatch)

If DATA_DIR/EXPECTED_PAIRS.json exists, the set of (base, derived, base_rev, derived_rev) found in
the result files must equal it exactly: no missing, duplicated or unrelated pair.

The check is a pure function, check_pair(), so a negative control can drive it to fail.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
import prior_shift as ps  # noqa: E402

TOL = 1e-6


def load_sums(npz):
    z = np.load(npz)
    old = {k[4:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("old_")}
    new = {k[4:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("new_")}
    return old, new, torch.from_numpy(z["ntok"])


def check_pair(result: dict, old: dict, new: dict, ntok) -> list:
    """Problems found when recomputing point estimates (bootstrap CIs are not recomputed: B=2000)."""
    problems = []
    if "id" not in old or len(old) < 2:
        problems.append(f"too few corrections in docsums: {sorted(old)}")
    shapes = {v.shape for v in list(old.values()) + list(new.values())} | {ntok.shape}
    if len(shapes) != 1:
        problems.append(f"per-document arrays differ in shape: {shapes}")
    if not all(bool(torch.isfinite(v).all()) for v in list(old.values()) + list(new.values())):
        problems.append("non-finite per-document sums")
    if bool((ntok < 0).any()):
        problems.append("negative per-document token count")
    if problems:
        return problems
    if set(old) != set(new):
        problems.append(f"old/new correction keys differ: {sorted(set(old) ^ set(new))}")
        return problems
    if not result.get("estimates"):
        return ["result.json has no estimates"]
    est, _ = ps.stats_from_docsums(old, new, ntok, B=1)
    missing = sorted(k for k in est if k.startswith(("Delta", "S_sym_", "S_card_")) and k not in result["estimates"])
    if missing:
        problems.append(f"result.json lacks recomputable estimates: {missing[:5]}")
    for k, v in result["estimates"].items():
        if k not in est:
            problems.append(f"{k}: in result.json, not recomputable")
        elif not (abs(est[k] - v) <= TOL * max(1.0, abs(v)) or (v != v and est[k] != est[k])):
            problems.append(f"{k}: result.json {v!r} vs recomputed {est[k]!r}")
    if int(ntok.sum()) <= 0:
        problems.append("no EVAL tokens")
    return problems


def main(root, expected=None):
    paths = sorted(glob.glob(os.path.join(root, "*", "*", "result.json")))
    if not paths:
        raise SystemExit(f"no result.json under {root}")
    if expected is not None and len(paths) != expected:
        print(f"FAIL expected {expected} pairs, found {len(paths)}")
        return 1
    exp_path = os.path.join(root, "EXPECTED_PAIRS.json")
    if os.path.exists(exp_path):
        want = sorted(json.load(open(exp_path)))
        got = sorted(json.load(open(p))["pair"] + json.load(open(p))["revisions"] for p in paths)
        if got != want:
            print(f"FAIL pair set differs from EXPECTED_PAIRS.json: missing {[w for w in want if w not in got]}, "
                  f"unexpected {[g for g in got if g not in want]}")
            return 1
    bad = 0
    for p in paths:
        old, new, ntok = load_sums(os.path.join(os.path.dirname(p), "docsums.npz"))
        probs = check_pair(json.load(open(p)), old, new, ntok)
        print(("FAIL " if probs else "ok   ") + os.path.relpath(p, root))
        for q in probs:
            print("     " + q)
        bad += bool(probs)
    print(f"{len(paths) - bad}/{len(paths)} pairs reproduce from per-document sums")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data",
                          int(sys.argv[2]) if len(sys.argv) > 2 else None))
