"""
Test Program 2: Modified ARTASI Alternative Ranking
===================================================
Executes Module 2 (Modified ARTASI) on the 2024 financial performance dataset
for the top 5 global insurance brokerage firms (AJG, AON, BRO, MMC, WTW).
Validates data preprocessing (Step 0), absolute bounds (Step 2), standardized
matrix (Step 3), usefulness degrees (Step 4), and final utility ranking (Table 10).
"""

import sys
import numpy as np
from fcdm.dataset import get_dataset, INSURANCE_BROKERS, CRITERIA_METADATA
from fcdm.modified_artasi import ModifiedARTASI


def main():
    print("=" * 85)
    print("TEST PROGRAM 2: MODIFIED ARTASI MULTI-CRITERIA EVALUATION")
    print("Case Study: 2024 Performance Evaluation of Global Insurance Brokers")
    print("Reference: Applied Soft Computing Journal 190 (2026) 114557 (Tables 8, 9, 10)")
    print("=" * 85)

    data = get_dataset()
    X = data["X"]
    weights = data["weights"]
    criteria_types = data["criteria_types"]
    criteria_codes = data["criteria_codes"]
    alt_names = data["alternative_names"]

    print(f"\n[1] Problem Dimensions:")
    print(f"    - Alternatives (m = {len(alt_names)}): {', '.join(alt_names)}")
    print(f"    - Criteria (n = {len(criteria_codes)}): 30 Financial, Operational & Market Indicators")
    print(f"    - Standardization Interval: [Psi_l = 1.0, Psi_u = 100.0]")
    print(f"    - Aggregation Parameters: alpha = 0.5, phi = 1.0")

    # Initialize Modified ARTASI
    model = ModifiedARTASI(
        weights=weights,
        criteria_types=criteria_types,
        psi_l=1.0,
        psi_u=100.0,
        alpha=0.5,
        phi=1.0
    )

    results = model.evaluate(X, alternative_names=alt_names)

    # 1. Step 0 & 2 Sample: Verify absolute bounds on selected criteria
    print(f"\n[2] Step 0 & Step 2 Verification (Sample Criteria Reference Bounds):")
    print(f"{'Code':<6} {'Criterion Name':<32} {'Col Min':<10} {'p_min_j':<12} {'Col Max':<10} {'p_max_j':<12}")
    print("-" * 85)
    for j in [0, 1, 5, 6, 16, 28]: # C1, C2, C6, C7, C17, C29
        meta = CRITERIA_METADATA[j]
        c_code = meta["code"]
        c_name = meta["name"]
        raw_min = np.min(X[:, j])
        raw_max = np.max(X[:, j])
        p_min = results["p_min"][j]
        p_max = results["p_max"][j]
        print(f"{c_code:<6} {c_name:<32} {raw_min:<10.2f} {p_min:<12.2f} {raw_max:<10.2f} {p_max:<12.2f}")

    # 2. Aggregated degrees of usefulness & Ranking
    print(f"\n[3] Step 5 & Step 6 Results: Utility Scores and Alternative Rankings:")
    print(f"{'Rank':<6} {'Code':<6} {'Broker Name':<42} {'I+':<10} {'I-':<10} {'f(I+)':<8} {'f(I-)':<8} {'Omega':<10}")
    print("-" * 105)

    broker_dict = {b["code"]: b["name"] for b in INSURANCE_BROKERS}
    rank_order = np.argsort(results["ranks"])

    # Published Table 10 reference rankings for comparison
    published_results = {
        "MMC": {"I_pos": 70.63, "I_neg": 83.41, "Omega": 77.02, "rank": 1},
        "BRO": {"I_pos": 66.98, "I_neg": 82.07, "Omega": 74.53, "rank": 2},
        "AON": {"I_pos": 63.86, "I_neg": 79.01, "Omega": 71.44, "rank": 3},
        "WTW": {"I_pos": 67.58, "I_neg": 75.13, "Omega": 71.36, "rank": 4},
        "AJG": {"I_pos": 59.55, "I_neg": 82.19, "Omega": 70.87, "rank": 5}
    }

    for idx in rank_order:
        code = alt_names[idx]
        name = broker_dict[code]
        r = results["ranks"][idx]
        i_p = results["I_pos"][idx]
        i_n = results["I_neg"][idx]
        f_p = results["f_pos"][idx]
        f_n = results["f_neg"][idx]
        om = results["Omega"][idx]
        print(f"{r:<6} {code:<6} {name:<42} {i_p:<10.2f} {i_n:<10.2f} {f_p:<8.2f} {f_n:<8.2f} {om:<10.2f}")

    print("-" * 105)
    print("\n[4] Comparison with Published Table 10 Rankings:")
    print(f"{'Broker':<8} {'Computed Rank':<15} {'Published Rank':<16} {'Published Omega':<18} {'Status':<10}")
    print("-" * 70)
    for code in ["MMC", "BRO", "AON", "WTW", "AJG"]:
        idx = alt_names.index(code)
        c_rank = results["ranks"][idx]
        pub = published_results[code]
        p_rank = pub["rank"]
        p_om = pub["Omega"]
        status = "MATCH" if c_rank == p_rank else "DIVERGE"
        print(f"{code:<8} Rank {c_rank:<10} Rank {p_rank:<11} {p_om:<18.2f} {status:<10}")

    print("\n>>> Performance Takeaways from the Research Paper:")
    print("    * Marsh & McLennan (MMC) occupies the dominant 1st position across both specifications.")
    print("    * Large-scale operations, superior cash generation, and balance-sheet capacity drive top tier.")
    print("    * Mid-cap agile broker Brown & Brown (BRO) exhibits resilient efficiency and high ROA/EBITDA margins.")
    print("=" * 85)


if __name__ == "__main__":
    main()
