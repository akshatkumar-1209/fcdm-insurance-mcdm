#!/usr/bin/env python3
"""
SRSP Reservoir - Modified ARTASI ranking, step by step
=======================================================
This program does NOT calculate CRITIC weights. It only READS the CRITIC weights you already
computed in the workbook, then calculates everything else needed for the ARTASI ranking.

Input : workbook with one sheet per cluster:
          * row 1 headers  F1 | F2 (or -F2) | F3  (any columns; headers must be exactly F<number> or -F<number>)
          * column A = Sl No., one alternative per row below the headers
          * a CRITIC weight row: a cell labelled  wj / w_j / weight(s)  with the weights to its right
            (if there is no label, a row below the data whose values are in [0,1] and sum to 1 is used)
Output: (1) every ARTASI table printed on screen, step by step, for each cluster
        (2) one Excel workbook with the same tables + the best alternative of each cluster.

Run   : python srsp_artasi.py "SRSP CRITIC Weights - All Clusters.xlsx" -o SRSP_ARTASI_Results.xlsx
        --cluster Cluster1   only one sheet            --top 10   rows shown in the final ranking table (default all)
        --types max,max,max  criteria directions (default all max, because F2 was converted to -F2)

FORMULAS (x_ij = value of alternative i on criterion j, m alternatives, n criteria, w_j = CRITIC weight)
-------------------------------------------------------------------------------------------------------
Step 0  Remove negatives     x_ij = x'_ij + |min_i x'_ij|        (only for a column that has negative values, e.g. -F2)
Step 1  Reference bounds     p_max_j = max_i x_ij + (max_i x_ij)^(1/m)
                             p_min_j = min_i x_ij - (min_i x_ij)^(1/m)
Step 2  Standardise [1,100]  phi_ij = PsiL + (x_ij - p_min_j)/(p_max_j - p_min_j) * (PsiU - PsiL)
                             benefit criterion: zeta_ij = phi_ij
                             cost criterion   : zeta_ij = max_k phi_kj + min_k phi_kj - phi_ij
Step 3  Ideal usefulness     theta+_ij = zeta_ij / max_k zeta_kj * w_j * PsiU
Step 4  Anti-ideal usefulness  t_ij = min_k zeta_kj / zeta_ij * w_j * PsiU
                             theta-_ij = max_k t_kj + min_k t_kj - t_ij
Step 5  Aggregation          I+_i = sum_j theta+_ij          I-_i = sum_j theta-_ij
Step 6  Final utility        f(I+) = I+/(I+ + I-)            f(I-) = I-/(I+ + I-)
                             Omega_i = (I+ + I-) * [ alpha*f(I+)^phi + (1-alpha)*f(I-)^phi ]^(1/phi)
                             (alpha = 0.5, phi = 1  ->  Omega = (I+ + I-)/2)
Step 7  Ranking              rank 1 = largest Omega; best alternative of the cluster = rank 1
"""
import argparse
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

PSI_L, PSI_U, ALPHA, PHI = 1.0, 100.0, 0.5, 1.0

# ------------------------------------------------------------------ reading the workbook
import re
CRIT = re.compile(r"^-?\s*F\s*\d+$", re.I)
WLAB = {"wj", "w_j", "weight", "weights", "criticweight", "criticweights", "criticweightswj"}

def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)

def read_cluster(ws, wsv):
    """Read alternatives and CRITIC weights from one sheet. Returns None if no weights are found."""
    cols = [c for c in range(1, ws.max_column + 1)
            if isinstance(wsv.cell(1, c).value, str) and CRIT.match(wsv.cell(1, c).value.strip())]
    if not cols:
        return None
    names = [wsv.cell(1, c).value.strip() for c in cols]
    n, ids, rows, r = len(cols), [], [], 2
    while all(_num(wsv.cell(r, c).value) for c in cols):
        rows.append([wsv.cell(r, c).value for c in cols]); ids.append(wsv.cell(r, 1).value); r += 1
    if not rows:
        return None
    last = r - 1
    w = None
    for rr in range(1, ws.max_row + 1):                       # 1) labelled weight row
        for cc in range(1, ws.max_column + 1):
            v = wsv.cell(rr, cc).value
            if isinstance(v, str) and re.sub(r"[^a-z_]", "", v.lower()) in WLAB:
                nums = [wsv.cell(rr, k).value for k in range(cc + 1, ws.max_column + 1) if _num(wsv.cell(rr, k).value)]
                if len(nums) >= n:
                    w = nums[:n]; break
        if w: break
    if w is None:                                               # 2) unlabelled row summing to 1 below the data
        for rr in range(last + 1, ws.max_row + 1):
            nums = [wsv.cell(rr, k).value for k in range(1, ws.max_column + 1) if _num(wsv.cell(rr, k).value)]
            for i in range(len(nums) - n + 1):
                seg = nums[i:i + n]
                if all(0 <= x <= 1 for x in seg) and abs(sum(seg) - 1) < 1e-6:
                    w = seg; break
            if w: break
    if w is None:
        return None
    w = np.array(w, float)
    if np.any(w < 0) or abs(w.sum() - 1) > 1e-6:
        raise ValueError(f"{ws.title}: CRITIC weights must be >= 0 and sum to 1 (got {w}).")
    return dict(ids=ids, names=names, X=np.array(rows, float), weights=w)

# ------------------------------------------------------------------ the ARTASI steps
def step0_remove_negatives(X):
    Xs = X.copy()
    for j in range(Xs.shape[1]):
        if Xs[:, j].min() < 0:
            Xs[:, j] += abs(Xs[:, j].min())
    return Xs

def step1_bounds(Xs):
    m = Xs.shape[0]
    mx, mn = Xs.max(0), Xs.min(0)
    return mx + mx ** (1.0 / m), mn - mn ** (1.0 / m)            # p_max, p_min

def step2_standardise(Xs, p_max, p_min, types):
    phi = PSI_L + (Xs - p_min) / (p_max - p_min) * (PSI_U - PSI_L)
    zeta = phi.copy()
    for j, t in enumerate(types):
        if t == "min":                                              # cost criterion: reverse
            zeta[:, j] = phi[:, j].max() + phi[:, j].min() - phi[:, j]
    return phi, zeta

def step3_ideal(zeta, w):
    return zeta / zeta.max(0) * w * PSI_U                           # theta+

def step4_anti_ideal(zeta, w):
    t = zeta.min(0) / zeta * w * PSI_U
    return t, t.max(0) + t.min(0) - t                               # t (temp), theta-

def step6_omega(I_pos, I_neg):
    s = I_pos + I_neg
    fp, fn = I_pos / s, I_neg / s
    return fp, fn, s * (ALPHA * fp ** PHI + (1 - ALPHA) * fn ** PHI) ** (1.0 / PHI)

def rank_desc(omega, tol=1e-9):
    return np.array([1 + int(np.sum(omega > o + tol)) for o in omega])   # ties share a rank

def evaluate_cluster(d, types):
    Xs = step0_remove_negatives(d["X"])
    p_max, p_min = step1_bounds(Xs)
    phi, zeta = step2_standardise(Xs, p_max, p_min, types)
    th_pos = step3_ideal(zeta, d["weights"])
    t_tmp, th_neg = step4_anti_ideal(zeta, d["weights"])
    I_pos, I_neg = th_pos.sum(1), th_neg.sum(1)
    f_pos, f_neg, omega = step6_omega(I_pos, I_neg)
    return dict(X_trans=Xs, p_max=p_max, p_min=p_min, phi=phi, zeta=zeta, theta_pos=th_pos, theta_tmp=t_tmp,
                theta_neg=th_neg, I_pos=I_pos, I_neg=I_neg, f_pos=f_pos, f_neg=f_neg, Omega=omega,
                rank=rank_desc(omega))

def run(path, types=None, only=None):
    wb, wbv = openpyxl.load_workbook(path), openpyxl.load_workbook(path, data_only=True)
    out = {}
    for ws in wb.worksheets:
        if only and ws.title != only:
            continue
        d = read_cluster(ws, wbv[ws.title])
        if d is None:
            print(f"(skipped sheet '{ws.title}': no criteria columns / CRITIC weights found)")
            continue
        t = types or ["max"] * len(d["names"])
        if len(t) != len(d["names"]):
            raise ValueError("--types needs one max/min entry per criterion")
        out[ws.title] = dict(data=d, res=evaluate_cluster(d, t), types=t)
    return out

# ------------------------------------------------------------------ on-screen report
def table(headers, rows, fmt):
    cells = [[h for h in headers]] + [[f(v) for f, v in zip(fmt, r)] for r in rows]
    wd = [max(len(str(c[i])) for c in cells) for i in range(len(headers))]
    line = "  ".join("-" * w for w in wd)
    s = ["  ".join(str(v).rjust(w) for v, w in zip(cells[0], wd)), line]
    s += ["  ".join(str(v).rjust(w) for v, w in zip(c, wd)) for c in cells[1:]]
    return "\n".join(s)

def report(name, o, top=None):
    d, r, nm, w = o["data"], o["res"], o["data"]["names"], o["data"]["weights"]
    m, n = len(d["ids"]), len(nm)
    f4, f6 = (lambda v: f"{v:.4f}"), (lambda v: f"{v:.6f}")
    ids = d["ids"]
    print("\n" + "=" * 100 + f"\n{name}: {m} alternatives | criteria {nm} | directions {o['types']}\n" + "=" * 100)
    print("\nTABLE 1  CRITIC weights w_j (read from the workbook, not calculated here)")
    print(table(["criterion", "w_j"], [[nm[j], w[j]] for j in range(n)], [str, f6]) + f"\n(sum = {w.sum():.6f})")
    print("\nTABLE 2  Step 0 - decision matrix after removing negatives:  x = x' + |min x'|  (columns without negatives unchanged)")
    print(table(["Sl No."] + nm, [[ids[i], *r["X_trans"][i]] for i in range(m)], [str] + [f4] * n))
    print(f"\nTABLE 3  Step 1 - reference bounds:  p_max = max(x) + max(x)^(1/m),  p_min = min(x) - min(x)^(1/m),  m = {m}")
    print(table(["criterion", "min x", "max x", "p_min", "p_max"],
                [[nm[j], r["X_trans"][:, j].min(), r["X_trans"][:, j].max(), r["p_min"][j], r["p_max"][j]] for j in range(n)],
                [str, f4, f4, f4, f4]))
    print(f"\nTABLE 4  Step 2 - standardised values  zeta = {PSI_L:g} + (x - p_min)/(p_max - p_min) * ({PSI_U:g} - {PSI_L:g})")
    print(table(["Sl No."] + nm, [[ids[i], *r["zeta"][i]] for i in range(m)], [str] + [f4] * n))
    print(table(["", *nm], [["max zeta (ideal)", *r["zeta"].max(0)], ["min zeta (anti-ideal)", *r["zeta"].min(0)]], [str] + [f4] * n))
    print("\nTABLE 5  Step 3 - ideal usefulness  theta+ = zeta / max(zeta) * w * 100,   I+ = sum of theta+")
    print(table(["Sl No."] + [f"th+ {c}" for c in nm] + ["I+"], [[ids[i], *r["theta_pos"][i], r["I_pos"][i]] for i in range(m)],
                [str] + [f4] * (n + 1)))
    print("\nTABLE 6  Step 4 - anti-ideal usefulness  t = min(zeta)/zeta * w * 100,  theta- = max(t) + min(t) - t,   I- = sum of theta-")
    print(table(["Sl No."] + [f"th- {c}" for c in nm] + ["I-"], [[ids[i], *r["theta_neg"][i], r["I_neg"][i]] for i in range(m)],
                [str] + [f4] * (n + 1)))
    order = np.argsort(r["rank"], kind="stable")
    shown = order if not top else order[:top]
    print(f"\nTABLE 7  Steps 5-7 - final utility  Omega = (I+ + I-) * [alpha*f(I+) + (1-alpha)*f(I-)],  alpha = {ALPHA:g}, "
          f"f(I+) = I+/(I+ + I-),  f(I-) = I-/(I+ + I-)   -> sorted by rank")
    print(table(["Rank", "Sl No.", "I+", "I-", "f(I+)", "f(I-)", "Omega"],
                [[r["rank"][i], ids[i], r["I_pos"][i], r["I_neg"][i], r["f_pos"][i], r["f_neg"][i], r["Omega"][i]] for i in shown],
                [lambda v: str(int(v)), str, f4, f4, f4, f4, f4]))
    b = order[0]
    ties = [ids[i] for i in np.where(r["rank"] == 1)[0]]
    print(f"\nDECISION  best alternative of {name} = rank 1 = highest Omega: Sl No. {', '.join(map(str, ties))} "
          f"(I+ = {r['I_pos'][b]:.4f}, I- = {r['I_neg'][b]:.4f}, Omega = {r['Omega'][b]:.4f}); "
          f"runner-up Sl No. {ids[order[1]]} (Omega {r['Omega'][order[1]]:.4f})")

# ------------------------------------------------------------------ Excel export
def export(out, path):
    wb = openpyxl.Workbook()
    F, FB = Font(name="Times New Roman", size=12), Font(name="Times New Roman", size=12, bold=True)
    hf, gf = PatternFill("solid", fgColor="FDE9D9"), PatternFill("solid", fgColor="C6EFCE")
    th = Side(style="thin"); bd = Border(left=th, right=th, top=th, bottom=th)
    ctr = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def put(ws, r, c, v, bold=False, fill=None, fmt=None):
        x = ws.cell(r, c, v); x.font = FB if bold else F; x.border = bd; x.alignment = ctr
        if fill: x.fill = fill
        if fmt: x.number_format = fmt

    s = wb.active; s.title = "Best_Alternatives"
    for j, h in enumerate(["Cluster", "No. of alternatives", "Best alternative (Sl No.)", "I+", "I-", "Omega",
                           "2nd (Sl No.)", "3rd (Sl No.)"], 1): put(s, 1, j, h, True, hf)
    for i, (name, o) in enumerate(out.items(), 2):
        d, r = o["data"], o["res"]; od = np.argsort(r["rank"], kind="stable"); b = od[0]
        vals = [name, len(d["ids"]), ", ".join(str(d["ids"][k]) for k in np.where(r["rank"] == 1)[0]),
                float(r["I_pos"][b]), float(r["I_neg"][b]), float(r["Omega"][b]), d["ids"][od[1]], d["ids"][od[2]]]
        for j, v in enumerate(vals, 1): put(s, i, j, v, j == 3, gf if j == 3 else None, "0.0000" if j in (4, 5, 6) else None)
    for j in range(1, 9): s.column_dimensions[get_column_letter(j)].width = 24

    g = wb.create_sheet("Formulas")
    for i, line in enumerate(__doc__.split("FORMULAS", 1)[1].split("\n"), 1):
        g.cell(i, 1, ("FORMULAS" + line) if i == 1 else line).font = F
    g.column_dimensions["A"].width = 130

    for name, o in out.items():
        d, r, nm = o["data"], o["res"], o["data"]["names"]; n, m = len(nm), len(d["ids"])
        ws = wb.create_sheet(f"{name}_ARTASI")
        blocks = [("Raw x'", d["X"], 4), ("Step 0: x (shifted)", r["X_trans"], 4), ("Step 2: phi (1-100)", r["phi"], 4),
                  ("Step 2: zeta", r["zeta"], 4), ("Step 3: theta+ (ideal)", r["theta_pos"], 4),
                  ("Step 4: theta- (anti-ideal)", r["theta_neg"], 4)]
        put(ws, 1, 1, "Sl No.", True, hf); put(ws, 2, 1, "", True, hf); c = 2
        for title, M, _ in blocks:
            for j in range(n):
                put(ws, 1, c + j, title, True, hf); put(ws, 2, c + j, nm[j], True, hf)
                for i in range(m): put(ws, 3 + i, c + j, float(M[i, j]), fmt="0.0000")
            c += n
        for k, (t, v) in enumerate([("Step 5: I+", r["I_pos"]), ("Step 5: I-", r["I_neg"]), ("Step 6: f(I+)", r["f_pos"]),
                                    ("Step 6: f(I-)", r["f_neg"]), ("Step 6: Omega", r["Omega"]), ("Step 7: Rank", r["rank"])]):
            put(ws, 1, c + k, t, True, hf); put(ws, 2, c + k, "", True, hf)
            for i in range(m): put(ws, 3 + i, c + k, float(v[i]) if k < 5 else int(v[i]), k == 5, None, "0.0000" if k < 5 else None)
        for i in range(m):
            put(ws, 3 + i, 1, d["ids"][i])
            if r["rank"][i] == 1:
                for cc in range(1, c + 6): ws.cell(3 + i, cc).fill = gf
        base = 3 + m + 1
        for k, (lab, vec, fm) in enumerate([("CRITIC weight w_j", d["weights"], "0.000000"), ("p_max (Step 1)", r["p_max"], "0.0000"),
                                            ("p_min (Step 1)", r["p_min"], "0.0000"), ("max zeta", r["zeta"].max(0), "0.0000"),
                                            ("min zeta", r["zeta"].min(0), "0.0000")]):
            put(ws, base + k, 1, lab, True, hf)
            for j in range(n): put(ws, base + k, 2 + n + j, float(vec[j]), fmt=fm)
        sc = c + 8
        for j, h in enumerate(["Rank", "Sl No.", "Omega"]): put(ws, 1, sc + j, h, True, hf)
        for q, i in enumerate(np.argsort(r["rank"], kind="stable")):
            put(ws, 3 + q, sc, int(r["rank"][i]), True); put(ws, 3 + q, sc + 1, d["ids"][i]); put(ws, 3 + q, sc + 2, float(r["Omega"][i]), fmt="0.0000")
        ws.row_dimensions[1].height = 34
        for cc in range(1, sc + 3): ws.column_dimensions[get_column_letter(cc)].width = 14
        ws.column_dimensions["A"].width = 20
    wb.save(path)

def main():
    ap = argparse.ArgumentParser(description="Modified ARTASI ranking per cluster, with all step tables.")
    ap.add_argument("workbook"); ap.add_argument("-o", "--output", default="SRSP_ARTASI_Results.xlsx")
    ap.add_argument("--top", type=int, default=None); ap.add_argument("--types", default=None); ap.add_argument("--cluster", default=None)
    a = ap.parse_args()
    out = run(a.workbook, [t.strip().lower() for t in a.types.split(",")] if a.types else None, a.cluster)
    if not out:
        raise SystemExit("No cluster sheets with criteria columns and CRITIC weights found.")
    for name, o in out.items(): report(name, o, a.top)
    print("\n" + "=" * 100 + "\nBEST ALTERNATIVE IN EACH CLUSTER (rank 1 by Omega)\n" + "=" * 100)
    rows = []
    for name, o in out.items():
        r, d = o["res"], o["data"]; b = int(np.argmin(r["rank"]))
        rows.append([name, len(d["ids"]), d["ids"][b], r["I_pos"][b], r["I_neg"][b], r["Omega"][b]])
    print(table(["Cluster", "m", "Best Sl No.", "I+", "I-", "Omega"], rows, [str, str, str] + [lambda v: f"{v:.4f}"] * 3))
    export(out, a.output); print(f"\nSaved {a.output}")

if __name__ == "__main__":
    main()
