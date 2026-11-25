#!/usr/bin/env python3
"""
Universal Modified ARTASI ranking for the SRSP cluster workbook.

Usage
-----
    python srsp_artasi.py  "SRSP CRITIC Weights - All Clusters.xlsx"  [-o output.xlsx]
                           [--types max,max,max] [--alpha 0.5] [--phi 1.0]

Input workbook (per cluster sheet, detected automatically)
----------------------------------------------------------
  * Row 1 headers; column A = alternative id; columns B.. = criteria (contiguous
    headers, stops at the first blank header cell, e.g. F1 | -F2 | F3).
  * Data rows start at row 2 and run while column B is numeric.
  * A cell in column E whose text is 'wj' marks the CRITIC weight row; the weights
    sit in columns F.. of that row (same order as the criteria). They are READ, never
    recalculated (the workbook must contain cached values, i.e. saved by Excel).
  * Sheets without a 'wj' row are skipped.

Criteria directions
-------------------
  --types gives one 'max'/'min' per criterion (default: all 'max'). In this project
  F2 has already been replaced by -F2 in the workbook, so all three are maximised.

The ARTASI maths is the repository's `ModifiedARTASI` (fcdm/modified_artasi.py),
copied unchanged into modified_artasi.py.
"""
import argparse, sys, os
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modified_artasi import ModifiedARTASI


def read_cluster(ws, wsv):
    """Return dict(ids, names, X, weights) for one sheet or None if it has no 'wj' row."""
    wrow = None
    for r in range(1, ws.max_row + 1):
        v = ws.cell(r, 5).value
        if isinstance(v, str) and v.strip().lower() == "wj":
            wrow = r
            break
    if wrow is None:
        return None
    names, c = [], 2
    while ws.cell(1, c).value not in (None, ""):
        names.append(str(ws.cell(1, c).value).strip()); c += 1
    n = len(names)
    ids, rows = [], []
    r = 2
    while isinstance(wsv.cell(r, 2).value, (int, float)):
        rows.append([wsv.cell(r, 2 + j).value for j in range(n)])
        ids.append(wsv.cell(r, 1).value); r += 1
    X = np.array(rows, dtype=float)
    w = [wsv.cell(wrow, 6 + j).value for j in range(n)]
    if any(not isinstance(x, (int, float)) for x in w):
        raise ValueError(f"{ws.title}: weight cells have no cached values - open and save the "
                         "workbook in Excel first.")
    w = np.array(w, dtype=float)
    if np.any(w < 0) or abs(w.sum() - 1) > 1e-6:
        raise ValueError(f"{ws.title}: weights must be non-negative and sum to 1 (got {w}).")
    return dict(ids=ids, names=names, X=X, weights=w)


def competition_rank(omega, tol=1e-9):
    """1 + number of alternatives with strictly larger Omega (ties share a rank)."""
    return np.array([1 + int(np.sum(omega > o + tol)) for o in omega])


def run(path, types=None, alpha=0.5, phi=1.0):
    wb = openpyxl.load_workbook(path)                  # formulas
    wbv = openpyxl.load_workbook(path, data_only=True)  # cached values
    out = {}
    for ws in wb.worksheets:
        d = read_cluster(ws, wbv[ws.title])
        if d is None:
            print(f"  skipped sheet '{ws.title}' (no 'wj' weight row)"); continue
        t = types or ["max"] * len(d["names"])
        if len(t) != len(d["names"]):
            raise ValueError("--types must list one entry per criterion")
        model = ModifiedARTASI(d["weights"], t, alpha=alpha, phi=phi)
        res = model.evaluate(d["X"], [str(i) for i in d["ids"]])
        res["rank"] = competition_rank(res["Omega"])
        out[ws.title] = dict(data=d, res=res, types=t)
        best = [d["ids"][i] for i in np.where(res["rank"] == 1)[0]]
        print(f"  {ws.title}: {len(d['ids'])} alternatives, best = {best}")
    return out


def export(out, path, alpha, phi):
    wb = openpyxl.Workbook()
    F = Font(name="Times New Roman", size=12); FB = Font(name="Times New Roman", size=12, bold=True)
    hdr = PatternFill("solid", fgColor="FDE9D9"); best = PatternFill("solid", fgColor="C6EFCE")
    th = Side(style="thin"); bd = Border(left=th, right=th, top=th, bottom=th)
    ctr = Alignment(horizontal="center", vertical="center")

    def put(ws, r, c, v, bold=False, fill=None, fmt=None):
        x = ws.cell(r, c, v); x.font = FB if bold else F; x.border = bd; x.alignment = ctr
        if fill: x.fill = fill
        if fmt: x.number_format = fmt

    # Summary
    s = wb.active; s.title = "Best_Alternatives"
    for j, h in enumerate(["Cluster", "No. of alternatives", "Best alternative (Sl No.)", "Omega (score)",
                           "I+", "I-", "2nd best", "3rd best"], 1):
        put(s, 1, j, h, True, hdr)
    for i, (name, o) in enumerate(out.items(), 2):
        d, r = o["data"], o["res"]
        order = np.argsort(r["rank"], kind="stable")
        b = order[0]
        put(s, i, 1, name); put(s, i, 2, len(d["ids"]))
        tied = [str(d["ids"][k]) for k in np.where(r["rank"] == 1)[0]]
        put(s, i, 3, ", ".join(tied)); put(s, i, 4, float(r["Omega"][b]), fmt="0.0000")
        put(s, i, 5, float(r["I_pos"][b]), fmt="0.0000"); put(s, i, 6, float(r["I_neg"][b]), fmt="0.0000")
        put(s, i, 7, d["ids"][order[1]] if len(order) > 1 else ""); put(s, i, 8, d["ids"][order[2]] if len(order) > 2 else "")
    for j in range(1, 9): s.column_dimensions[get_column_letter(j)].width = 24
    s.cell(len(out) + 3, 1, f"Modified ARTASI, alpha={alpha}, phi={phi}; psi in [1,100]; weights = CRITIC weights read from the input workbook. "
           "Higher Omega = better.").font = F

    # Weights used
    wsw = wb.create_sheet("Weights_Used")
    maxc = max(len(o["data"]["names"]) for o in out.values())
    put(wsw, 1, 1, "Cluster", True, hdr)
    for j in range(maxc): put(wsw, 1, 2 + j, f"w({out[next(iter(out))]['data']['names'][j]})", True, hdr)
    put(wsw, 1, 2 + maxc, "Sum", True, hdr); put(wsw, 1, 3 + maxc, "Criteria direction", True, hdr)
    for i, (name, o) in enumerate(out.items(), 2):
        put(wsw, i, 1, name)
        for j, w in enumerate(o["data"]["weights"]): put(wsw, i, 2 + j, float(w), fmt="0.000000")
        put(wsw, i, 2 + maxc, float(o["data"]["weights"].sum()), fmt="0.000000")
        put(wsw, i, 3 + maxc, ", ".join(f"{n}:{t}" for n, t in zip(o["data"]["names"], o["types"])))
    for j in range(1, 4 + maxc): wsw.column_dimensions[get_column_letter(j)].width = 18
    wsw.column_dimensions[get_column_letter(3 + maxc)].width = 36

    # One sheet per cluster (original order + rank, then a sorted ranking block)
    for name, o in out.items():
        d, r = o["data"], o["res"]
        ws = wb.create_sheet(f"{name}_ARTASI")
        heads = ["Sl No."] + d["names"] + ["I+", "I-", "Omega", "Rank"]
        for j, h in enumerate(heads, 1): put(ws, 1, j, h, True, hdr)
        for i in range(len(d["ids"])):
            row = i + 2
            put(ws, row, 1, d["ids"][i])
            for j in range(len(d["names"])): put(ws, row, 2 + j, float(d["X"][i, j]))
            k = 2 + len(d["names"])
            put(ws, row, k, float(r["I_pos"][i]), fmt="0.0000"); put(ws, row, k + 1, float(r["I_neg"][i]), fmt="0.0000")
            put(ws, row, k + 2, float(r["Omega"][i]), fmt="0.0000"); put(ws, row, k + 3, int(r["rank"][i]), True)
            if r["rank"][i] == 1:
                for j in range(1, k + 4): ws.cell(row, j).fill = best
        # sorted ranking
        c0 = len(heads) + 2
        for j, h in enumerate(["Rank", "Sl No.", "Omega"], c0): put(ws, 1, j, h, True, hdr)
        for q, i in enumerate(np.argsort(r["rank"], kind="stable"), 2):
            put(ws, q, c0, int(r["rank"][i]), True); put(ws, q, c0 + 1, d["ids"][i]); put(ws, q, c0 + 2, float(r["Omega"][i]), fmt="0.0000")
        for j in range(1, c0 + 3): ws.column_dimensions[get_column_letter(j)].width = 15
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workbook"); ap.add_argument("-o", "--output", default="SRSP_ARTASI_Results.xlsx")
    ap.add_argument("--types", default=None, help="comma list of max/min per criterion (default all max)")
    ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--phi", type=float, default=1.0)
    a = ap.parse_args()
    types = [t.strip().lower() for t in a.types.split(",")] if a.types else None
    out = run(a.workbook, types, a.alpha, a.phi)
    if not out: sys.exit("No cluster sheets found.")
    export(out, a.output, a.alpha, a.phi)
    print(f"Saved {a.output}")


if __name__ == "__main__":
    main()
