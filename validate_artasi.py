"""Usage: python validate_artasi.py <input workbook> <results workbook>
Independent re-implementation (plain python, from the formulas) vs ModifiedARTASI, all clusters."""
import sys, math
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, openpyxl
import srsp_artasi as S

def ref_artasi(X, w, alpha=0.5, phi_exp=1.0, lo=1.0, hi=100.0):
    m, n = len(X), len(X[0])
    cols = [[X[i][j] for i in range(m)] for j in range(n)]
    cols = [[v + abs(min(c)) for v in c] if min(c) < 0 else c[:] for c in cols]      # step 0 (negatives)
    zeta = []
    for c in cols:
        pmax = max(c) + max(c) ** (1 / m); pmin = min(c) - min(c) ** (1 / m)
        zeta.append([lo + (v - pmin) / (pmax - pmin) * (hi - lo) for v in c])          # all 'max'
    thp, thn = [], []
    for j in range(n):
        z = zeta[j]; tp = [v / max(z) * w[j] * hi for v in z]
        tmp = [min(z) / v * w[j] * hi for v in z]
        thp.append(tp); thn.append([max(tmp) + min(tmp) - t for t in tmp])
    Ip = [sum(thp[j][i] for j in range(n)) for i in range(m)]
    In = [sum(thn[j][i] for j in range(n)) for i in range(m)]
    return [(a + b) * (alpha * a / (a + b) + (1 - alpha) * b / (a + b)) for a, b in zip(Ip, In)], Ip, In

path = sys.argv[1]; results = sys.argv[2]
out = S.run(path)
from modified_artasi import ModifiedARTASI
for name, o in out.items():
    ref = ModifiedARTASI(o["data"]["weights"], o["types"]).evaluate(o["data"]["X"])
    print(name, "matches repo ModifiedARTASI class:", np.allclose(ref["Omega"], o["res"]["Omega"], atol=1e-9))
for name, o in out.items():
    d, r = o["data"], o["res"]
    om, ip, ineg = ref_artasi(d["X"].tolist(), d["weights"].tolist())
    dO = max(abs(np.array(om) - r["Omega"])); dI = max(abs(np.array(ip) - r["I_pos"]))
    rk_ref = [1 + sum(1 for b in om if b > a + 1e-9) for a in om]
    print(f"{name}: max|dOmega|={dO:.2e} max|dI+|={dI:.2e} ranks_equal={rk_ref==list(r['rank'])} "
          f"Omega range [{min(om):.2f},{max(om):.2f}] weights_sum={d['weights'].sum():.6f}")
# exported workbook matches
wo = openpyxl.load_workbook(results)
print(wo.sheetnames)
for name in out:
    ws = wo[f"{name}_ARTASI"]; n = len(out[name]["data"]["names"]); m = len(out[name]["data"]["ids"]); col = 2 + 6 * n + 4
    om = [ws.cell(r, col).value for r in range(3, 3 + m)]
    print(name, "export ok:", np.allclose(om, out[name]["res"]["Omega"]))
