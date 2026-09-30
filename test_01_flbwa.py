from fcdm.dataset import EXPERT_PREFERENCES_TABLE5, BASELINE_WEIGHTS_TABLE7, CRITERIA_METADATA
from fcdm.flbwa import FuzzyLBWA

def main():
    model = FuzzyLBWA(levels_data=EXPERT_PREFERENCES_TABLE5, best_criterion="C29", theta=14.01)
    results = model.compute()

    print("=" * 85)
    print("TEST 1: FUZZY LBWA WEIGHT DERIVATION (Table 6 & Table 7)")
    print("=" * 85)
    print(f"Scale Max (delta): {results['delta']} | Elasticity (theta): {results['theta']:.2f}")
    print(f"\n{'Code':<5} {'Criterion Name':<32} {'Computed TFN':<22} {'Computed W':<12} {'Table 7 W':<12} {'Diff':<8}")
    print("-" * 95)

    meta = {c["code"]: c["name"] for c in CRITERIA_METADATA}
    max_diff = 0.0
    for code, name in meta.items():
        t = results["fuzzy_weights"][code]
        w_c = results["crisp_weights"][code]
        w_p = BASELINE_WEIGHTS_TABLE7[code]["crisp"]
        diff = abs(w_c - w_p)
        max_diff = max(max_diff, diff)
        print(f"{code:<5} {name:<32} ({t.l:.3f}, {t.m:.3f}, {t.u:.3f})   {w_c:<12.4f} {w_p:<12.4f} {diff:<8.4f}")

    print("-" * 95)
    print(f"Max Absolute Deviation: {max_diff:.4f} | Sum of Normalized Weights: {sum(results['normalized_weights'].values()):.4f}")

if __name__ == "__main__":
    main()
