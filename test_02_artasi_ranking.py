import numpy as np
from fcdm.dataset import get_dataset, INSURANCE_BROKERS
from fcdm.modified_artasi import ModifiedARTASI

def main():
    data = get_dataset()
    model = ModifiedARTASI(weights=data["weights"], criteria_types=data["criteria_types"])
    results = model.evaluate(data["X"], alternative_names=data["alternative_names"])

    published = {
        "MMC": (70.63, 83.41, 77.02, 1),
        "BRO": (66.98, 82.07, 74.53, 2),
        "AON": (63.86, 79.01, 71.44, 3),
        "WTW": (67.58, 75.13, 71.36, 4),
        "AJG": (59.55, 82.19, 70.87, 5)
    }

    brokers = {b["code"]: b["name"] for b in INSURANCE_BROKERS}
    rank_order = np.argsort(results["ranks"])

    print("=" * 90)
    print("TEST 2: MODIFIED ARTASI ALTERNATIVE RANKING (Table 10)")
    print("=" * 90)
    print(f"{'Rank':<6} {'Code':<6} {'Broker Name':<38} {'I+':<8} {'I-':<8} {'Omega':<10} {'Table 10 Omega':<16} {'Table 10 Rank':<14}")
    print("-" * 105)

    for idx in rank_order:
        code = data["alternative_names"][idx]
        p_ip, p_in, p_om, p_rk = published[code]
        print(f"{results['ranks'][idx]:<6} {code:<6} {brokers[code]:<38} {results['I_pos'][idx]:<8.2f} {results['I_neg'][idx]:<8.2f} {results['Omega'][idx]:<10.2f} {p_om:<16.2f} {p_rk:<14}")
    print("-" * 105)

if __name__ == "__main__":
    main()
