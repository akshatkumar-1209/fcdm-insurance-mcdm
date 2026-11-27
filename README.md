# SRSP Reservoir – CRITIC Weights + Modified ARTASI Ranking

## The problem
The Sriram Sagar Reservoir Problem (SRSP) is a multi-objective reservoir operation problem. Each candidate
release policy (a point on the Pareto front, grouped into 7 clusters) is judged on three objectives:

| Objective | Meaning | Direction |
|---|---|---|
| **F1** | Agricultural benefit from using reservoir water (crop production) | Maximise (benefit) |
| **F2** | Cost / expenditure of water-resource utilisation | Minimise (cost) |
| **F3** | Environmental benefit: water released for aquatic / marine life | Maximise (benefit) |

The aim is to **rank the alternatives inside each cluster and pick the best one**.

## Handling the cost objective: F2 → −F2
F1 and F3 should be as large as possible; F2 should be as small as possible. To treat all three the same way,
F2 is replaced by **−F2**, because *maximising −F2 is the same as minimising F2*. After this one change every
criterion is a maximisation (benefit) criterion, so no separate cost-handling is needed downstream.

## How it is solved
1. **CRITIC weights** (done in Excel, `SRSP CRITIC Weights - All Clusters.xlsx`, sheets Cluster1–Cluster7).
   No expert scores are needed; the weights come from the data itself. Notation: `x_aj` = value of alternative `a`
   on criterion `j` (F1, −F2, F3), `n` = number of alternatives, `f_j*` = best (max) and `f_j,*` = worst (min) value of criterion `j`.
   - Normalisation (all criteria maximised): `x_aj = (f_j(a) − f_j,*) / (f_j* − f_j,*)`  → every value lies in [0, 1]
   - Standard deviation of each normalised column: `σ_j = √[ Σ_a (x_aj − x̄_j)² / (n − 1) ]`
   - Correlation between criteria `j` and `k`: `r_jk = Pearson correlation of columns x_j and x_k`
   - Conflict of criterion `j`: `Σ_k (1 − r_jk)`
   - Information content: `C_j = σ_j · Σ_k (1 − r_jk)`
   - **Weight: `w_j = C_j / Σ_k C_k`** (so `w_j ≥ 0` and `Σ w_j = 1`)
2. **Modified ARTASI** (`srsp_artasi.py`) **reads** these weights (it does not recalculate them) and computes, per cluster:
   - Step 0: remove negatives (−F2 is negative): `x = x' + |min x'|`
   - Step 1: bounds `p_max = max(x) + max(x)^(1/m)`, `p_min = min(x) − min(x)^(1/m)`
   - Step 2: standardise to 1–100: `ζ = 1 + (x − p_min)/(p_max − p_min)·99`
   - Step 3: ideal usefulness `θ⁺ = ζ / max ζ · w · 100`, and `I⁺ = Σθ⁺`
   - Step 4: anti-ideal usefulness `θ⁻ = max t + min t − t`, with `t = min ζ / ζ · w · 100`, and `I⁻ = Σθ⁻`
   - Step 5–6: `Ω = (I⁺ + I⁻)·[α·f(I⁺) + (1−α)·f(I⁻)]`, `f(I⁺) = I⁺/(I⁺+I⁻)`, `f(I⁻) = I⁻/(I⁺+I⁻)`, `α = 0.5`
   - Step 7: **rank by Ω (highest = rank 1); the rank-1 alternative is the best in the cluster**

## Run
```bash
pip install numpy openpyxl
python srsp_artasi.py "SRSP CRITIC Weights - All Clusters.xlsx" -o "SRSP Results Final.xlsx"
# options: --cluster Cluster1   --top 10   --types max,max,max
```
The program prints every step as a table (weights, shifted data, bounds, ζ, θ⁺ with I⁺, θ⁻ with I⁻, Ω and rank)
for each cluster, then the best alternative of each cluster, and saves the same tables to Excel.

## Results (best alternative per cluster)
| Cluster | Alternatives | Best (Sl No.) | Ω |
|---|---|---|---|
| Cluster1 | 21 | 1 | 76.8349 |
| Cluster2 | 41 | 14 | 86.3582 |
| Cluster3 | 58 | 29 | 85.0066 |
| Cluster4 | 33 | 16 | 84.7302 |
| Cluster5 | 60 | 5 | 87.3342 |
| Cluster6 | 28 | 5 | 83.9795 |
| Cluster7 | 42 | 28 | 81.0946 |

## Files
- `srsp_artasi.py` – reads CRITIC weights, runs ARTASI, prints and exports all tables
- `SRSP CRITIC Weights - All Clusters.xlsx` – cluster data (with F2 as −F2) and CRITIC weights (input)
- `SRSP Results Final.xlsx` – step tables, Ω, ranks and best alternatives (output)
