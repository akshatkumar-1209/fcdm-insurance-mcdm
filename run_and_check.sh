#!/bin/bash
# One command: run SRSP ARTASI on all clusters and check that every table and alternative is shown.
# Usage:  bash run_and_check.sh
IN="SRSP CRITIC Weights - All Clusters.xlsx"
OUT="SRSP Results Final.xlsx"
[ -f "$IN" ] || { echo "Missing input workbook: $IN (upload it next to this script)"; exit 1; }
pip install -q numpy openpyxl 2>/dev/null
python srsp_artasi.py "$IN" -o "$OUT" > output.txt || { echo "Run failed - see output.txt"; exit 1; }
python - "$OUT" <<'PY'
import sys, openpyxl
exp = {"Cluster1":21,"Cluster2":41,"Cluster3":58,"Cluster4":33,"Cluster5":60,"Cluster6":28,"Cluster7":42}
wb = openpyxl.load_workbook(sys.argv[1]); ok = True
for s in wb.sheetnames:
    if s.endswith("_ARTASI"):
        m = wb[s].max_row - 8; e = exp.get(s.replace("_ARTASI",""))
        print(f"{s:18s} {m:3d} alternatives", "" if e is None else ("OK" if m==e else f"MISMATCH (expected {e})")); ok &= (e is None or m==e)
print("Excel check:", "PASS" if ok else "FAIL")
PY
T=$(grep -c "^TABLE" output.txt); D=$(grep -c "^DECISION" output.txt)
echo "Tables printed: $T (expected 7 per cluster)   Decisions: $D (one per cluster)"
echo; tail -12 output.txt
echo; echo "Full tables are in output.txt and $OUT"
