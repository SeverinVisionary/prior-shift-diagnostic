#!/usr/bin/env bash
# Self-asserting check of this deposit on a CPU (a few minutes). Exit status 0 is the result.
# It does not re-run the models (that needs a GPU; see README.md, "Re-running the models").
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c "import numpy, torch" || { echo "needs python3 with numpy and torch" >&2; exit 2; }
TMP=$(mktemp -d "${TMPDIR:-/tmp}/prior_shift_check.XXXXXX"); TMP=$(cd "$TMP" && pwd); trap 'rm -rf "$TMP"' EXIT

echo "1/4 checksums"
if command -v sha256sum >/dev/null; then sha256sum -c --quiet SHA256SUMS; else shasum -a 256 -c --quiet SHA256SUMS; fi

echo "2/4 synthetic known-answer tests"
if ! ( cd code && "$PY" test_prior_shift.py > "$TMP/tests.txt" 2> "$TMP/tests.err" ); then
  grep -E "FAIL" "$TMP/tests.txt" || true; tail -20 "$TMP/tests.err"; exit 1
fi
grep -q "^ALL PASS$" "$TMP/tests.txt"
echo "  passed: $(grep -c '^PASS' "$TMP/tests.txt") of $(grep -cE '^(PASS|FAIL)' "$TMP/tests.txt")"
grep -E "^\[(ctx|proxy)\]" "$TMP/tests.txt" | sed 's/^/  /'   # the values README.md cites

echo "3/4 every estimate recomputed from the per-document sums"
"$PY" verify_data.py data 14
"$PY" test_verify_data.py data > "$TMP/negctl.txt" 2>&1 || { cat "$TMP/negctl.txt"; exit 1; }
sed 's/^/  /' "$TMP/negctl.txt"

echo "4/4 summary table regenerated"
cp -r data "$TMP/data"; rm -f "$TMP/data/summary.json" "$TMP/data/summary.md"
"$PY" code/summarize.py "$TMP/data" > /dev/null
cmp "$TMP/data/summary.json" data/summary.json
cmp "$TMP/data/summary.md" data/summary.md
echo "ALL CHECKS PASSED"
