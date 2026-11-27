#!/bin/bash
# One command: run SRSP ARTASI for every cluster, print each cluster's step tables on screen, then check them.
# Usage:  bash run_and_check.sh [input_workbook.xlsx]
# Input  : the Excel workbook that holds the cluster data and the CRITIC weights.
#          Default = "SRSP CRITIC ... ARTASI ... All Clusters.xlsx" (formula workbook) if present,
#          otherwise "SRSP CRITIC Weights - All Clusters.xlsx".
# Output : "SRSP Python ARTASI Tables.xlsx"  (never overwrites your own results workbook)
IN="$1"
if [ -z "$IN" ]; then
  IN=$(ls SRSP\ CRITIC*ARTASI*All\ Clusters.xlsx 2>/dev/null | head -1)
  [ -z "$IN" ] && IN="SRSP CRITIC Weights - All Clusters.xlsx"
fi
OUT="SRSP Python ARTASI Tables.xlsx"
[ -f "$IN" ] || { echo "Missing input workbook: $IN (upload it next to this script)"; exit 1; }
pip install -q numpy openpyxl 2>/dev/null
echo "Input workbook : $IN"
echo "Running ARTASI for every cluster..."
python srsp_artasi.py "$IN" -o "$OUT" | tee output.txt
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "Run failed - see output.txt"; exit 1; }
echo; echo "================ CHECKS ================"
python - "$OUT" "$IN" <<'PY'
import sys, openpyxl, numpy as np
import srsp_artasi as S
exp = {"Cluster1":21,"Cluster2":41,"Cluster3":58,"Cluster4":33,"Cluster5":60,"Cluster6":28,"Cluster7":42}
wb = openpyxl.load_workbook(sys.argv[1]); ok = True
for s in wb.sheetnames:
    if s.endswith("_ARTASI"):
        m = wb[s].max_row - 8; e = exp.get(s.replace("_ARTASI",""))
        print(f"{s:18s} {m:3d} alternatives", "" if e is None else ("OK" if m==e else f"MISMATCH (expected {e})")); ok &= (e is None or m==e)
print("Excel output check:", "PASS" if ok else "FAIL")
# cross-check: Omega already calculated by Excel formulas in the input workbook vs Python Omega
res = S.run(sys.argv[2]); wv = openpyxl.load_workbook(sys.argv[2], data_only=True); done = 0; worst = 0.0
for name, o in res.items():
    ws = wv[name]
    col = next((c for c in range(1, ws.max_column+1) if isinstance(ws.cell(1,c).value, str) and "omega" in ws.cell(1,c).value.lower() and "step" in ws.cell(1,c).value.lower()), None)
    if col is None: continue
    xl = np.array([ws.cell(2+i, col).value for i in range(len(o["data"]["ids"]))], float)
    worst = max(worst, float(np.abs(xl - o["res"]["Omega"]).max())); done += 1
if done: print(f"Excel-formula Omega vs Python Omega: {done} clusters compared, max difference {worst:.1e} ->", "MATCH" if worst < 1e-8 else "DIFFERENT")
else: print("(input workbook has no Excel ARTASI formula columns, so no formula cross-check)")
PY
T=$(grep -c "^TABLE" output.txt); R=$(grep -c "^RANKING" output.txt)
echo "Cluster tables (I+, I-, Omega): $T (expected 7)   Final rankings: $R (expected 7)"
echo; echo "Full tables are in output.txt and $OUT"
