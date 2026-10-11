#!/usr/bin/env python3
"""Negative controls for verify_data.py: it must accept a shipped pair and reject each corruption.

usage: test_verify_data.py DATA_DIR
"""
import contextlib
import copy
import io
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_data as vd  # noqa: E402


def main(root):
    p = sorted(glob.glob(os.path.join(root, "*", "*", "result.json")))[0]
    r = json.load(open(p))
    old, new, ntok = vd.load_sums(os.path.join(os.path.dirname(p), "docsums.npz"))
    assert vd.check_pair(r, old, new, ntok) == [], "baseline pair rejected"
    cases = {}
    r2 = copy.deepcopy(r); r2["estimates"]["Delta"] += 1e-3; cases["wrong Delta"] = (r2, old, new, ntok)
    r3 = copy.deepcopy(r); r3["estimates"] = {"Delta": r["estimates"]["Delta"]}; cases["estimates dropped"] = (r3, old, new, ntok)
    o4 = {k: v.clone() for k, v in old.items()}; o4["aff"] = o4["aff"] * 1.001; cases["perturbed sums"] = (r, o4, new, ntok)
    n5 = ntok.clone(); n5[0] = -n5[0]; cases["negative token count"] = (r, old, new, n5)
    o6 = {k: v.clone() for k, v in old.items()}; o6["id"][0] = float("nan"); cases["non-finite sum"] = (r, o6, new, ntok)
    o7 = {"id": old["id"]}; cases["corrections missing"] = (r, o7, {"id": new["id"]}, ntok)
    for name, args in cases.items():
        assert vd.check_pair(*args), f"verify_data accepted: {name}"
    with contextlib.redirect_stdout(io.StringIO()):  # its FAIL line is the expected outcome here
        rc = vd.main(root, expected=len(glob.glob(os.path.join(root, "*", "*", "result.json"))) + 1)
    assert rc == 1, "verify_data accepted a wrong pair count"
    # a misassembled directory with the right count but a duplicated pair must fail
    import shutil, tempfile
    with tempfile.TemporaryDirectory() as t:
        shutil.copytree(root, os.path.join(t, "d"))
        ps = sorted(glob.glob(os.path.join(t, "d", "*", "*", "result.json")))
        shutil.copy(ps[0], ps[1])
        with contextlib.redirect_stdout(io.StringIO()):
            rc = vd.main(os.path.join(t, "d"), expected=len(ps))
        assert rc == 1, "verify_data accepted a duplicated pair"
    print(f"verify_data negative controls: baseline accepted, {len(cases) + 2} corruptions rejected")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
