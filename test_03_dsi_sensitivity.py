"""
Test Program 3: Decision Stability Intervals (DSI) Sensitivity Analysis
=======================================================================
Executes Module 3 (Decision Stability Intervals) under the Modified ARTASI framework.
Quantifies the numerical tolerance intervals [w_min, w_max] for criteria weights
and [x_min, x_max] for decision matrix entries, comparing with Table 11 & Table 12
from the research paper.
"""

import sys
import numpy as np
from fcdm.dataset import get_dataset, CRITERIA_METADATA, INSURANCE_BROKERS
from fcdm.modified_artasi import ModifiedARTASI
from fcdm.dsi_sensitivity import DecisionStabilityIntervals


def main():
    print("=" * 90)
    print("TEST PROGRAM 3: DECISION STABILITY INTERVALS (DSI) SENSITIVITY ANALYSIS")
    print("Reference: Applied Soft Computing Journal 190 (2026) 114557 (Tables 11 & 12)")
    print("=" * 90)

    data = get_dataset()
    X = data["X"]
    weights = data["weights"]
    criteria_types = data["criteria_types"]
    criteria_codes = data["criteria_codes"]
    alt_names = data["alternative_names"]

    # 1. Base Model Evaluation
    model = ModifiedARTASI(
        weights=weights,
        criteria_types=criteria_types,
        psi_l=1.0,
        psi_u=100.0,
        alpha=0.5,
        phi=1.0
    )
    base_res = model.evaluate(X, alt_names)
    print(f"\n[1] Baseline Top Ranking Alternative: {alt_names[np.argmin(base_res['ranks'])]} (Rank 1)")

    # 2. Compute Criteria Weight DSI (Tolerance bounds for weight perturbations)
    print("\n[2] Computing Criterion Weight Stability Intervals (DSI)...")
    dsi_analyzer = DecisionStabilityIntervals(
        artasi_model=model,
        step_size=0.0005,
        strict_ranking=False  # Top-Rank (Winner) Decision Invariance
    )

    weight_dsi = dsi_analyzer.compute_weight_dsi(X, criteria_codes=criteria_codes)

    print(f"\n{'Code':<6} {'Criterion Name':<32} {'Base W':<10} {'Lower W':<10} {'Upper W':<10} {'Width':<10} {'Rel Width':<10}")
    print("-" * 90)

    metadata_dict = {c["code"]: c for c in CRITERIA_METADATA}
    for item in weight_dsi:
        code = item["criterion"]
        name = metadata_dict[code]["name"]
        w_b = item["original_weight"]
        w_l = item["lower_limit"]
        w_u = item["upper_limit"]
        width = item["interval_width"]
        rel_w = item["relative_width"]
        print(f"{code:<6} {name:<32} {w_b:<10.4f} {w_l:<10.4f} {w_u:<10.4f} {width:<10.4f} {rel_w:<10.2f}")

    print("-" * 90)

    # 3. Identify Top 3 Most Robust and Top 3 Most Sensitive Criteria
    sorted_by_width = sorted(weight_dsi, key=lambda x: x["interval_width"])
    print("\n[3] Stability Spectrum Diagnostics:")
    print("    * Most Sensitive Criteria (Tightest Tolerances - High Fragility):")
    for item in sorted_by_width[:3]:
        name = metadata_dict[item["criterion"]]["name"]
        print(f"      - {item['criterion']} ({name}): Interval Width = {item['interval_width']:.4f} (Max change: +{item['upper_limit'] - item['original_weight']:.4f})")

    print("\n    * Most Robust Criteria (Broadest Tolerances - High Stability):")
    for item in sorted_by_width[-3:]:
        name = metadata_dict[item["criterion"]]["name"]
        print(f"      - {item['criterion']} ({name}): Interval Width = {item['interval_width']:.4f} (Max change: +{item['upper_limit'] - item['original_weight']:.4f})")

    # 4. Matrix Cell Stability Check (Sample from Table 12)
    print("\n[4] Decision-Matrix Stability Intervals Sample (MMC & AJG on selected criteria):")
    matrix_dsi = dsi_analyzer.compute_matrix_dsi(
        X,
        alternative_names=alt_names,
        criteria_codes=criteria_codes,
        max_iterations=100
    )

    print(f"{'Broker':<8} {'Criterion':<6} {'Criterion Name':<28} {'Base Val':<12} {'Lower Bound':<12} {'Upper Bound':<12}")
    print("-" * 80)
    for broker in ["MMC", "AJG"]:
        items = matrix_dsi[broker]
        for idx in [0, 5, 18, 29]: # C1, C6, C19 (Total Assets), C30 (Employees)
            entry = items[idx]
            c_name = metadata_dict[entry["criterion"]]["name"]
            print(f"{broker:<8} {entry['criterion']:<6} {c_name:<28} {entry['original_value']:<12.2f} {entry['lower_limit']:<12.2f} {entry['upper_limit']:<12.2f}")

    print("=" * 90)
    print(">>> DSI Analysis Completed Successfully!")


if __name__ == "__main__":
    main()
