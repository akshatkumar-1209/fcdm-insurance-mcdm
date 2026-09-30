# Fuzzy MCDM Framework for Insurance Broker Performance Evaluation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A robust hybrid Multi-Criteria Decision-Making (MCDM) framework combining **Fuzzy Level-Based Weight Assessment (F-LBWA)** and **Modified ARTASI**, augmented with **Decision Stability Intervals (DSI)** and **Monte Carlo Robustness Analysis**.

Based on the research paper:
> **"A robust hybrid MCDM framework with emphasis on decision stability intervals: Performance evaluation of global insurance brokers using fuzzy LBWA and modified ARTASI"**  
> *Applied Soft Computing*, Vol. 190 (2026), 114557.

---

## 📌 Framework Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │          Expert Group Judgments               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Module 1: Fuzzy LBWA Weighting     │
                      └───────────────────┬───────────────────┘
                                          │  Criteria Weights (w_j)
                                          ▼
┌───────────────────────┐     ┌───────────────────────────────────────┐
│ Decision Matrix (X)   │────▶│    Module 2: Modified ARTASI Ranking  │
│ 5 Brokers x 30 Ratios │     └───────────────────┬───────────────────┘
└───────────────────────┘                         │  Utility Scores (Omega_i)
                                                  ▼
                      ┌───────────────────────────────────────┐
                      │   Module 3: DSI Sensitivity Analysis  │
                      └───────────────────┬───────────────────┘
                                          │  Stability Bounds [w_min, w_max]
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Module 4: Monte Carlo Robustness   │
                      └───────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### Module 1: Fuzzy LBWA Criteria Weight Assessment
Derives objective importance weights from non-linear expert rankings while accounting for cognitive imprecision via Triangular Fuzzy Numbers (TFNs) $\tilde{T} = (l, m, u)$.

1. **Significance Level Partitioning**:
   Criteria are grouped into ordered subsets $Q_1, Q_2, \dots, Q_k$ based on relative importance.
   $$\delta = \max_{r} |Q_r|$$

2. **Fuzzy Influence Function**:
   With elasticity coefficient $\theta > \delta$ ($\theta = 14.01$):
   $$\tilde{f}(C_{j}) = \left(\frac{\theta}{j\theta + x_j^{(u)}}, \, \frac{\theta}{j\theta + x_j^{(m)}}, \, \frac{\theta}{j\theta + x_j^{(l)}}\right)$$

3. **Weight Derivation**:
   For the highest-ranked criterion $C_*$ (Credit Rating, $C_{29}$):
   $$\tilde{w}_* = \frac{1}{1 + \sum_{j \neq *} \tilde{f}(C_j)}$$
   For all other criteria $j \neq *$:
   $$\tilde{w}_j = \tilde{f}(C_j) \otimes \tilde{w}_*$$

4. **Defuzzification & Normalization**:
   $$w_j = \frac{w_j^{(l)} + 4w_j^{(m)} + w_j^{(u)}}{6}, \quad W_j = \frac{w_j}{\sum_{k=1}^n w_k}$$

---

### Module 2: Modified ARTASI Alternative Ranking
Evaluates alternatives against absolute dynamic boundary reference points rather than relative extrema, preventing rank reversal anomalies.

1. **Non-Negativity Shift**:
   $$x_{ij} = x'_{ij} + |\min_k x'_{kj}| \quad \text{for criteria with negative values}$$

2. **Dynamic Range Formulation ($m$ alternatives)**:
   $$\wp_j^{\max} = \max_i(x_{ij}) + \left[\max_i(x_{ij})\right]^{1/m}, \quad \wp_j^{\min} = \min_i(x_{ij}) - \left[\min_i(x_{ij})\right]^{1/m}$$

3. **Standardization onto $[\Psi^{(l)}, \Psi^{(u)}] = [1, 100]$**:
   $$\phi_{ij} = \Psi^{(l)} + \frac{x_{ij} - \wp_j^{\min}}{\wp_j^{\max} - \wp_j^{\min}} \left(\Psi^{(u)} - \Psi^{(l)}\right)$$

4. **Cost Criteria Inversion**:
   $$\zeta_{ij} = \begin{cases} \phi_{ij}, & C_j \in \text{Benefit} \\ \max_k \phi_{kj} + \min_k \phi_{kj} - \phi_{ij}, & C_j \in \text{Cost} \end{cases}$$

5. **Degrees of Usefulness**:
   $$\vartheta_{ij}^+ = \frac{\zeta_{ij}}{\max_k \zeta_{kj}} \cdot w_j \cdot \Psi^{(u)}, \quad \vartheta_{ij}^- = \max_k \vartheta_{kj}^+ + \min_k \vartheta_{kj}^+ - \vartheta_{ij}^+$$

6. **Aggregated Comprehensive Utilities**:
   $$I_i^+ = \sum_{j=1}^n \vartheta_{ij}^+, \quad I_i^- = \sum_{j=1}^n \vartheta_{ij}^-$$
   $$\Omega_i = (I_i^+ + I_i^-) \left[ \alpha \cdot \frac{I_i^+}{I_i^+ + I_i^-} + (1 - \alpha) \cdot \frac{I_i^-}{I_i^+ + I_i^-} \right]$$
   *(With aggregation weight $\alpha = 0.5$. Ranking is determined by descending order of $\Omega_i$.)*

---

### Module 3: Decision Stability Intervals (DSI)
Quantifies the exact perturbation threshold a criterion weight or performance value can endure before inducing rank reversal of the optimal alternative $A_*$:

$$w_k \to w_k + \Delta w_k, \quad w_j' = w_j - \frac{\Delta w_k}{n - 1} \quad (\forall j \neq k)$$
$$\operatorname{DSI}(w_k) = [w_k^{\min}, \, w_k^{\max}] \quad \text{such that} \quad \operatorname{Rank}(A_*) = 1$$

- **High Interval Width** ($\Delta w_k \gg 0$): Stable criterion, robust against measurement errors.
- **Narrow Interval Width**: Sensitive critical indicator requiring precise audit.

---

### Module 4: Monte Carlo Robustness & Benchmarking
Simulates $N = 1000$ weight vectors uniformly across the unit simplex via Dirichlet sampling:
$$\mathbf{w}^{(s)} \sim \operatorname{Dir}(\mathbf{1}_n), \quad \sum_{j=1}^n w_j^{(s)} = 1$$

- **Pairwise Stochastic Dominance**:
  $$\mathbb{P}(A_i \succ A_k) = \frac{1}{N} \sum_{s=1}^N \mathbb{I}\left(\Omega_i^{(s)} > \Omega_k^{(s)}\right)$$

- **Comparative Benchmarking (TOPSIS Rank Correlation)**:
  $$\rho = 1 - \frac{6 \sum_{i=1}^m d_i^2}{m(m^2 - 1)}$$

---

## ⚡ Execution in Minimal Lines

Run each validation script directly from terminal:

```bash
python test_01_flbwa.py
python test_02_artasi_ranking.py
python test_03_dsi_sensitivity.py
python test_04_montecarlo_robustness.py
```

### Complete Workflow in 7 Lines of Python:

```python
from fcdm.dataset import get_dataset
from fcdm.flbwa import FuzzyLBWA
from fcdm.modified_artasi import ModifiedARTASI

data = get_dataset()
w = FuzzyLBWA(data["expert_levels"], "C29", 14.01).compute()["crisp_weights"]
model = ModifiedARTASI(weights=w, criteria_types=data["criteria_types"])
ranks = model.evaluate(data["X"], data["alternative_names"])["ranks"]
print(dict(zip(data["alternative_names"], ranks)))
```

---

## 📊 Summary of Results

### 1. Alternative Rankings (2024 S&P 500 Insurance Brokers)

| Rank | Broker Code | Firm Name | $\Omega_i$ Score | Empirical $P(\text{Rank}=1)$ |
|:---:|:---:|---|:---:|:---:|
| **1** | **MMC** | Marsh & McLennan Companies, Inc. | **85.69** | **93.1%** |
| **2** | **AJG** | Arthur J. Gallagher & Co. | **82.22** | 6.7% |
| **3** | **BRO** | Brown & Brown, Inc. | **73.04** | 0.2% |
| **4** | **AON** | Aon plc | **69.45** | 0.0% |
| **5** | **WTW** | Willis Towers Watson Public Limited Company | **55.10** | 0.0% |

### 2. Sensitivity & Stability Highlights
- **Top Winner Invariance**: MMC preserves Rank 1 under weight perturbations of up to $\pm 30\%$ on key financial indicators.
- **Most Critical Indicator**: $C_{15}$ (Current Assets / Total Assets) exhibited the narrowest DSI band.
- **Most Robust Indicator**: $C_{29}$ (Credit Rating) tolerated weight fluctuations up to width $0.5245$.
- **Validation Against TOPSIS**: Spearman rank correlation $\rho = 0.90$ ($p < 0.05$) verifies algorithmic consistency.

---

## 📁 Repository Structure

```text
fcdm-insurance-mcdm/
├── fcdm/
│   ├── __init__.py               # Package initializer
│   ├── dataset.py                # Decision matrix & criteria metadata (Tables 4, 5, 7, 8)
│   ├── flbwa.py                  # Module 1: Fuzzy LBWA implementation
│   ├── modified_artasi.py        # Module 2: Modified ARTASI implementation
│   ├── dsi_sensitivity.py        # Module 3: Decision Stability Intervals (DSI)
│   └── robustness_montecarlo.py  # Module 4: Monte Carlo simulation & TOPSIS benchmark
├── test_01_flbwa.py              # Test 1: Weight derivation validation
├── test_02_artasi_ranking.py     # Test 2: Alternative ranking validation
├── test_03_dsi_sensitivity.py    # Test 3: Weight stability interval validation
├── test_04_montecarlo_robustness.py # Test 4: 1000 MC runs & plot generation
├── monte_carlo_results.png       # Auto-generated visualization
├── requirements.txt              # numpy, scipy, matplotlib
└── README.md                     # Documentation with LaTeX formulations
```

---

## 📄 License
This project is licensed under the MIT License.
