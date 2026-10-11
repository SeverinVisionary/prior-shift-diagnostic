#!/usr/bin/env python3
"""Tabulate every out/<group>/<model>/result.json into summary.json and summary.md (descriptive only).

usage: summarize.py DATA_DIR
"""
import glob
import json
import os
import sys

FLOOR = 0.29  # S_aff under pure random context damage (test_prior_shift.py, case ctx)
KEYS = ["Delta", "S_sym_aff", "S_card_aff", "S_sym_cf_beta100000", "S_sym_bias", "S_sym_bias_top4k", "S_sym_aff_top4k"]


def main(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "result.json"))):
        if os.sep + "smoke" + os.sep in p:
            continue
        r = json.load(open(p))
        e, ci = r["estimates"], r["ci95"]
        row = {"group": p.split(os.sep)[-3], "base": r["pair"][0], "new": r["pair"][1],
               "revisions": r["revisions"], "T_new": r["new_meta"]["T_aff"], "T_old": r["old_meta"]["T_aff"],
               "T_new_top4k": r["new_meta"].get("T_aff_topk"), "T_old_top4k": r["old_meta"].get("T_aff_topk"),
               "jensen_ok": r["old_meta"]["jensen_in_sample_E"]["holds"] and r["new_meta"]["jensen_in_sample_E"]["holds"]}
        for k in KEYS:
            row[k], row[k + "_ci"] = e.get(k), ci.get(k)
        nan = [k for k in KEYS if row[k] is not None and row[k] != row[k]]
        row["valid"] = row["Delta"] > 0 and row["Delta_ci"][0] >= 0.005 and not nan and row["jensen_ok"]
        row["flag_bias"] = row["valid"] and max(row["S_sym_cf_beta100000"], row["S_sym_bias_top4k"]) >= 0.20
        row["flag_aff_above_floor"] = row["valid"] and row["S_sym_aff_top4k"] >= FLOOR + 0.10
        rows.append(row)
    json.dump(rows, open(os.path.join(root, "summary.json"), "w"), indent=1)
    f = lambda v: "—" if v is None else f"{v:.3f}"
    lines = ["| group | new | Δ | S_sym^aff | S_sym^cf | S_sym^bias(top4k) | S_sym^aff(top4k) | T new/old | valid | flags |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flags = ",".join(k[5:] for k in ("flag_bias", "flag_aff_above_floor") if r[k]) or "—"
        lines.append(f"| {r['group']} | {r['new'].split('/')[-1]} | {f(r['Delta'])} | {f(r['S_sym_aff'])} | "
                     f"{f(r['S_sym_cf_beta100000'])} | {f(r['S_sym_bias_top4k'])} | {f(r['S_sym_aff_top4k'])} | "
                     f"{f(r['T_new'])}/{f(r['T_old'])} | {r['valid']} | {flags} |")
    open(os.path.join(root, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
