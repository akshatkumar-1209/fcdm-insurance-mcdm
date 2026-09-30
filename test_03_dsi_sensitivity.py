from fcdm.dataset import get_dataset, CRITERIA_METADATA
from fcdm.modified_artasi import ModifiedARTASI
from fcdm.dsi_sensitivity import DecisionStabilityIntervals

def main():
    data = get_dataset()
    model = ModifiedARTASI(weights=data["weights"], criteria_types=data["criteria_types"])
    dsi = DecisionStabilityIntervals(artasi_model=model, step_size=0.0005)

    weight_dsi = dsi.compute_weight_dsi(data["X"], criteria_codes=data["criteria_codes"])

    print("=" * 85)
    print("TEST 3: DECISION STABILITY INTERVALS (DSI) SENSITIVITY (Table 11)")
    print("=" * 85)
    print(f"{'Code':<6} {'Criterion Name':<32} {'Base W':<10} {'Lower W':<10} {'Upper W':<10} {'Width':<10}")
    print("-" * 85)

    meta = {c["code"]: c["name"] for c in CRITERIA_METADATA}
    for item in weight_dsi:
        c = item["criterion"]
        print(f"{c:<6} {meta[c]:<32} {item['original_weight']:<10.4f} {item['lower_limit']:<10.4f} {item['upper_limit']:<10.4f} {item['interval_width']:<10.4f}")
    print("-" * 85)

    sorted_w = sorted(weight_dsi, key=lambda x: x["interval_width"])
    print(f"Most Sensitive Criterion: {sorted_w[0]['criterion']} ({meta[sorted_w[0]['criterion']]}) | Width = {sorted_w[0]['interval_width']:.4f}")
    print(f"Most Robust Criterion:    {sorted_w[-1]['criterion']} ({meta[sorted_w[-1]['criterion']]}) | Width = {sorted_w[-1]['interval_width']:.4f}")

if __name__ == "__main__":
    main()
