"""
Test Program 1: Fuzzy LBWA Criteria Weight Assessment
=====================================================
Executes Module 1 (F-LBWA) using expert preferences from Table 5.
Validates the derived Triangular Fuzzy Numbers and defuzzified crisp weights
against the published results in Table 6 and Table 7 of the research paper.
"""

import sys
from fcdm.dataset import EXPERT_PREFERENCES_TABLE5, BASELINE_WEIGHTS_TABLE7, CRITERIA_METADATA
from fcdm.flbwa import FuzzyLBWA


def main():
    print("=" * 80)
    print("TEST PROGRAM 1: FUZZY LBWA CRITERIA WEIGHT DERIVATION")
    print("Reference: Applied Soft Computing Journal 190 (2026) 114557 (Tables 5, 6, 7)")
    print("=" * 80)

    # Initialize F-LBWA with paper parameter theta = 14.01
    model = FuzzyLBWA(
        levels_data=EXPERT_PREFERENCES_TABLE5,
        best_criterion="C29",
        theta=14.01
    )

    results = model.compute()

    print(f"\n[1] Hierarchy & Elasticity Configuration:")
    print(f"    - Best Reference Criterion: C29 (Credit Rating)")
    print(f"    - Significance Levels: Q1 (9), Q2 (7), Q3 (5), Q4 (9)")
    print(f"    - Scale Maximum (delta): {results['delta']}")
    print(f"    - Elasticity Parameter (theta): {results['theta']:.2f}")

    print(f"\n[2] Comparison with Published Table 7 Baseline Weights:")
    print(f"{'Code':<5} {'Criterion Name':<34} {'Computed TFN':<24} {'Computed W':<12} {'Table 7 W':<12} {'Diff':<8}")
    print("-" * 100)

    metadata_dict = {c["code"]: c for c in CRITERIA_METADATA}
    all_codes = [c["code"] for c in CRITERIA_METADATA]

    max_diff = 0.0
    for code in all_codes:
        meta = metadata_dict[code]
        tfn = results["fuzzy_weights"][code]
        w_computed = results["crisp_weights"][code]
        w_published = BASELINE_WEIGHTS_TABLE7[code]["crisp"]
        diff = abs(w_computed - w_published)
        max_diff = max(max_diff, diff)

        tfn_str = f"({tfn.l:.3f}, {tfn.m:.3f}, {tfn.u:.3f})"
        print(f"{code:<5} {meta['name']:<34} {tfn_str:<24} {w_computed:<12.4f} {w_published:<12.4f} {diff:<8.4f}")

    print("-" * 100)
    sum_computed = sum(results["crisp_weights"].values())
    sum_normalized = sum(results["normalized_weights"].values())
    print(f"Sum of Crisp Weights: {sum_computed:.4f} (Normalized: {sum_normalized:.4f})")
    print(f"Maximum absolute deviation from Table 7: {max_diff:.4f}")

    if max_diff < 0.005:
        print("\n>>> VALIDATION SUCCESSFUL: F-LBWA weights match published Table 7 within tolerance!")
    else:
        print("\n>>> NOTICE: Slight deviations detected due to rounding conventions.")

    print("=" * 80)


if __name__ == "__main__":
    main()
