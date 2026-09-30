# FCDM: Fuzzy MCDM Framework for Insurance Broker Performance Evaluation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Complete-brightgreen)]()

> **Implementation of a Robust Hybrid Multi-Criteria Decision Making (MCDM) Framework featuring Fuzzy LBWA and Modified ARTASI with Decision Stability Intervals and Monte Carlo Robustness Assessment.**
>
> Based on the research paper:  
> **"A robust hybrid MCDM framework with emphasis on decision stability intervals: Performance evaluation of global insurance brokers using fuzzy LBWA and modified ARTASI"**  
> *Applied Soft Computing Journal 190 (2026) 114557*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Case Study](#case-study)
- [Modules](#modules)
  - [Module 1 — Fuzzy LBWA](#module-1--fuzzy-lbwa)
  - [Module 2 — Modified ARTASI](#module-2--modified-artasi)
  - [Module 3 — Decision Stability Intervals (DSI)](#module-3--decision-stability-intervals-dsi)
  - [Module 4 — Monte Carlo Robustness](#module-4--monte-carlo-robustness)
- [Test Programs](#test-programs)
- [Installation](#installation)
- [Usage](#usage)
- [Key Results](#key-results)

---

## Overview

This repository implements a hybrid Multi-Criteria Decision Making (MCDM) framework that integrates:

1. **Fuzzy Logic** — handling vagueness and imprecision in expert judgements via Triangular Fuzzy Numbers (TFNs)
2. **LBWA** (Level-Based Weight Assessment) — a minimal-input weighting method requiring only adjacent pairwise comparisons
3. **ARTASI** — a novel MCDM ranking technique resistant to rank reversal using reverse-sorting normalization and flexible aggregation
4. **Decision Stability Intervals (DSI)** — a what-if sensitivity analysis that quantifies the exact perturbation ranges that preserve the original ranking
5. **Monte Carlo Simulation** — probabilistic robustness assessment via 1000 Dirichlet-sampled weight scenarios

The framework is applied to evaluate **5 global insurance brokerage firms** listed on the S&P 500 index across **30 financial and operational performance criteria** for the year 2024.

---

## Case Study

### Alternatives (Insurance Brokers)

| Code | Full Name |
|------|-----------|
| **AJG** | Arthur J. Gallagher & Co. |
| **AON** | Aon plc |
| **BRO** | Brown & Brown, Inc. |
| **MMC** | Marsh & McLennan Companies, Inc. |
| **WTW** | Willis Towers Watson Public Limited Company |

### Evaluation Criteria (30 Indicators)

| Category | Criteria Codes | Type |
|----------|---------------|------|
| Liquidity Ratios | C1 – C3 | Benefit (max) |
| Leverage Ratios | C4 – C6 | Cost (min) |
| Profitability Ratios | C7 – C11 | Benefit (max) |
| Coverage / Efficiency | C12 – C15 | Benefit (max) |
| Cash Flow Statements | C16 – C18 | Mixed |
| Financial Statement Items | C19 – C25 | Mixed |
| Stock Market Performance | C26 – C28 | Mixed |
| Credit & Operational | C29 – C30 | Mixed |

---

## Modules

### Module 1 — Fuzzy LBWA

**File:** [`fcdm/flbwa.py`](fcdm/flbwa.py)

Implements **Fuzzy Level-Based Weight Assessment** for deriving criterion importance weights from expert judgements under uncertainty.

#### Key Equations

**Step 1:** Scale upper bound:
$$\delta = \max\{|Q_1|, |Q_2|, \ldots, |Q_k|\}$$

**Step 4:** Elasticity coefficient: $\theta > \delta$ (set to **14.01** in the paper)

**Step 5:** Fuzzy influence function for criterion $C_{jp}$ at significance level $j$:
$$\tilde{f}(C_{jp}) = \frac{\theta}{j \cdot \theta + \tilde{x}_{jp}}$$

where the TFN components are computed as:
$$f^{(l)} = \frac{\theta}{j\theta + x^{(u)}}, \quad f^{(m)} = \frac{\theta}{j\theta + x^{(m)}}, \quad f^{(u)} = \frac{\theta}{j\theta + x^{(l)}}$$

**Step 6a:** Fuzzy weight of the best criterion ($C_{29}$ — Credit Rating):
$$\tilde{w}_1 = \frac{1}{1 + \sum_{j \neq 1} \tilde{f}(C_j)}$$

**Step 6b:** Fuzzy weights of remaining criteria:
$$\tilde{w}_j = \tilde{f}(C_j) \times \tilde{w}_1, \quad j = 2, 3, \ldots, n$$

**Step 6c:** Centroid defuzzification:
$$W_j = \frac{w_j^{(l)} + 4w_j^{(m)} + w_j^{(u)}}{6}$$

---

### Module 2 — Modified ARTASI

**File:** [`fcdm/modified_artasi.py`](fcdm/modified_artasi.py)

Implements **Modified ARTASI** for ranking alternatives on the standardized decision matrix. Key improvement over classical ARTASI: explicit elimination of negative values before exponentiation.

#### Key Equations

**Step 0** — Negative value elimination:
$$x_{ij} = x'_{ij} + \left|\min_j x'_{ij}\right| \quad \text{(for columns with negatives only)}$$

**Step 2** — Absolute reference bounds (m = number of alternatives):
$$\wp^{\max}_j = \max_i(x_{ij}) + \left[\max_i(x_{ij})\right]^{1/m}, \quad \wp^{\min}_j = \min_i(x_{ij}) - \left[\min_i(x_{ij})\right]^{1/m}$$

**Step 3** — Standardization onto $[\Psi^{(l)}, \Psi^{(u)}] = [1, 100]$:
$$\phi_{ij} = \frac{\Psi^{(u)} - \Psi^{(l)}}{\wp^{\max}_j - \wp^{\min}_j} x_{ij} + \frac{\wp^{\max}_j \Psi^{(l)} - \wp^{\min}_j \Psi^{(u)}}{\wp^{\max}_j - \wp^{\min}_j}$$

For **cost criteria** (reverse-sorting):
$$\zeta_{ij} = -\phi_{ij} + \max_i(\phi_{ij}) + \min_i(\phi_{ij})$$

**Step 4** — Degrees of usefulness:
$$\vartheta^+_{ij} = \frac{\zeta_{ij}}{\max_i(\zeta_{ij})} \cdot w_j \cdot \Psi^{(u)}, \qquad \vartheta^-_{ij} = -\vartheta_{ij} + \max_i(\vartheta_{ij}) + \min_i(\vartheta_{ij})$$

**Step 5** — Aggregated utilities:
$$I^+_i = \sum_{j=1}^n \vartheta^+_{ij}, \qquad I^-_i = \sum_{j=1}^n \vartheta^-_{ij}$$

**Step 6** — Ultimate utility function (with $\alpha = 0.5$, $\phi = 1$):
$$\Omega_i = (I^+_i + I^-_i) \cdot \left[\alpha \cdot f(I^+_i) + (1 - \alpha) \cdot f(I^-_i)\right]$$

where $f(I^+_i) = \dfrac{I^+_i}{I^+_i + I^-_i}$ and $f(I^-_i) = \dfrac{I^-_i}{I^+_i + I^-_i}$

---

### Module 3 — Decision Stability Intervals (DSI)

**File:** [`fcdm/dsi_sensitivity.py`](fcdm/dsi_sensitivity.py)

Implements systematic **what-if analysis** on both criterion weights and decision matrix cells. For each criterion weight $w_k$:

**Perturbation Procedure:**
$$w_k \leftarrow w_k + \omega, \quad w_{j \neq k} \leftarrow w_j - \frac{\omega}{n-1} \quad \text{(so that } \textstyle\sum w_j = 1\text{)}$$

Repeat until rank order changes → record $w^{\max}_k$.

Mirror for decreases → record $w^{\min}_k$.

The **DSI** for criterion $k$ is then: $[w^{\min}_k, \; w^{\max}_k]$

A **wide interval** = the ranking is **robust** to changes in that criterion's weight.  
A **narrow interval** = the ranking is **fragile** and sensitive to small perturbations.

---

### Module 4 — Monte Carlo Robustness

**File:** [`fcdm/robustness_montecarlo.py`](fcdm/robustness_montecarlo.py)

Implements probabilistic robustness evaluation using:

1. **Dirichlet Simplex Sampling** — 1000 random weight vectors uniformly sampled from the 30-dimensional unit simplex:
   $$w_k = \frac{\tilde{w}_k}{\sum_{j=1}^{30} \tilde{w}_j}, \quad \tilde{w}_k \sim \text{Exp}(1)$$

2. **Score & Rank Distributions** — Empirical distributions of $\Omega_i$ and $\text{Rank}(i)$ over all simulations

3. **Pairwise Stochastic Dominance** — Probability that alternative $A_i$ outperforms $A_k$:
   $$P(A_i \succ A_k) = \frac{1}{N}\sum_{s=1}^{N} \mathbf{1}[\Omega_i^{(s)} > \Omega_k^{(s)}]$$

4. **Comparative Benchmarking** — ARTASI vs. TOPSIS with Spearman $\rho$ and Kendall $\tau$ rank correlations

---

## Test Programs

| Program | Description | Key Output |
|---------|-------------|------------|
| [`test_01_flbwa.py`](test_01_flbwa.py) | Executes F-LBWA and validates weight derivation against **Table 7** | 30 fuzzy weights + crisp values |
| [`test_02_artasi_ranking.py`](test_02_artasi_ranking.py) | Runs Modified ARTASI and validates ranking against **Table 10** | Omega scores + final broker ranking |
| [`test_03_dsi_sensitivity.py`](test_03_dsi_sensitivity.py) | Computes weight and matrix DSI intervals comparing against **Table 11 & 12** | Stability intervals for all 30 criteria |
| [`test_04_montecarlo_robustness.py`](test_04_montecarlo_robustness.py) | 1000-simulation MC analysis and ARTASI vs TOPSIS benchmarking | Rank distributions, dominance matrix, visualizations |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/fcdm-insurance-mcdm.git
cd fcdm-insurance-mcdm
pip install numpy scipy matplotlib
```

No additional dependencies required beyond Python standard library.

---

## Usage

```python
from fcdm.dataset import get_dataset
from fcdm.flbwa import FuzzyLBWA
from fcdm.modified_artasi import ModifiedARTASI
from fcdm.dsi_sensitivity import DecisionStabilityIntervals
from fcdm.robustness_montecarlo import MonteCarloRobustness

# Load 2024 insurance broker dataset
data = get_dataset()
X       = data["X"]            # (5 alternatives, 30 criteria) decision matrix
weights = data["weights"]       # Crisp weights from F-LBWA Table 7
c_types = data["criteria_types"] # 'max' or 'min' per criterion

# Run Modified ARTASI
model = ModifiedARTASI(weights=weights, criteria_types=c_types)
results = model.evaluate(X, alternative_names=data["alternative_names"])

print("Rankings:", dict(zip(data["alternative_names"], results["ranks"])))
# -> {'AJG': 2, 'AON': 4, 'BRO': 3, 'MMC': 1, 'WTW': 5}

# Run all test programs
python test_01_flbwa.py               # Module 1: Weight assessment
python test_02_artasi_ranking.py       # Module 2: MCDM ranking
python test_03_dsi_sensitivity.py      # Module 3: Stability intervals
python test_04_montecarlo_robustness.py # Module 4: Probabilistic robustness
```

---

## Key Results

### Final Rankings (Modified ARTASI, 2024 Data)

| Rank | Broker | Omega Score | Key Strength |
|------|--------|-------------|--------------|
| **1** | **MMC** (Marsh & McLennan) | **85.69** | Superior profitability, scale & cash flows |
| 2 | AJG (Arthur J. Gallagher) | 82.22 | Strong liquidity and leverage control |
| 3 | BRO (Brown & Brown) | 73.04 | Highest EBITDA margin, efficient operations |
| 4 | AON | 69.45 | Large revenue base, moderate risk profile |
| 5 | WTW | 55.10 | Weakest profitability; negative ROA and ROE |

### Monte Carlo Findings (1000 Simulations)

- **MMC** maintains Rank 1 in **93.1%** of all weight scenarios → stochastic dominance confirmed
- **WTW** occupies Rank 5 in **93.8%** of scenarios → consistent last-place positioning
- **Spearman ρ = 0.90** and **Kendall τ = 0.80** between ARTASI and TOPSIS → strong rank agreement

### Stability Insights

- **Most sensitive** criterion: `C15` (Current Assets/Total Assets) — tightest DSI width → small weight changes alter rankings
- **Most robust** criterion: `C29` (Credit Rating) — DSI interval width = 0.5245 → rankings highly stable to this criterion's weight fluctuation
- Large-scale firm attributes (Revenue, Total Assets, Employees) show widest matrix-level stability intervals

---

## Repository Structure

```
fcdm-insurance-mcdm/
├── fcdm/
│   ├── __init__.py                  # Package init
│   ├── dataset.py                   # Tables 4, 5, 7, 8 from the paper
│   ├── flbwa.py                     # Module 1: Fuzzy LBWA
│   ├── modified_artasi.py           # Module 2: Modified ARTASI
│   ├── dsi_sensitivity.py           # Module 3: Decision Stability Intervals
│   └── robustness_montecarlo.py     # Module 4: Monte Carlo Robustness + TOPSIS
├── test_01_flbwa.py                 # Test: F-LBWA weight derivation
├── test_02_artasi_ranking.py        # Test: Broker ranking & Table 10 validation
├── test_03_dsi_sensitivity.py       # Test: DSI weight & matrix sensitivity
├── test_04_montecarlo_robustness.py # Test: MC simulation & ARTASI vs TOPSIS
├── monte_carlo_results.png          # Auto-generated: MC visualization plots
├── requirements.txt
└── README.md
```

---

## Reference

**M. Özçalici et al.** (2026). *A robust hybrid MCDM framework with emphasis on decision stability intervals: Performance evaluation of global insurance brokers using fuzzy LBWA and modified ARTASI*. **Applied Soft Computing**, 190, 114557.  
https://doi.org/10.1016/j.asoc.2026.114557
