import numpy as np
import matplotlib.pyplot as plt
from fcdm.dataset import get_dataset
from fcdm.robustness_montecarlo import MonteCarloRobustness

def main():
    data = get_dataset()
    alts = data["alternative_names"]
    mc = MonteCarloRobustness(criteria_types=data["criteria_types"])

    sim = mc.run_simulation(X=data["X"], n_simulations=1000, alternative_names=alts, seed=42)
    bench = mc.compare_with_topsis(data["X"], data["weights"], alternative_names=alts)

    print("=" * 80)
    print("TEST 4: MONTE CARLO ROBUSTNESS & COMPARATIVE BENCHMARKING (1000 Runs)")
    print("=" * 80)
    print(f"{'Broker':<8} {'Mean Score':<12} {'Std':<8} {'P(Rank=1)':<12} {'ARTASI Rank':<14} {'TOPSIS Rank':<12}")
    print("-" * 72)

    for i, code in enumerate(alts):
        st = sim["score_stats"][code]
        p_r1 = sim["rank_distribution"][code][0]
        print(f"{code:<8} {st['mean']:<12.2f} {st['std']:<8.2f} {p_r1:<12.3f} {bench['artasi_ranks'][i]:<14} {bench['topsis_ranks'][i]:<12}")
    print("-" * 72)
    print(f"Rank Correlations: Spearman rho = {bench['spearman_rho']:.4f} | Kendall tau = {bench['kendall_tau']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot([sim["scores"][:, i] for i in range(len(alts))])
    axes[0].set_xticks(range(1, len(alts) + 1))
    axes[0].set_xticklabels(alts)
    axes[0].set_title("ARTASI Utility Distributions (1000 MC Runs)")
    axes[0].set_ylabel("Omega Score")

    x = np.arange(len(alts))
    w = 0.35
    axes[1].bar(x - w/2, bench["artasi_ranks"], w, label="Modified ARTASI")
    axes[1].bar(x + w/2, bench["topsis_ranks"], w, label="TOPSIS")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(alts)
    axes[1].set_title(f"Rank Comparison (Spearman rho = {bench['spearman_rho']:.2f})")
    axes[1].set_ylabel("Rank (1=Best)")
    axes[1].invert_yaxis()
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("monte_carlo_results.png", dpi=150)
    plt.close()
    print("Visualization saved to monte_carlo_results.png")

if __name__ == "__main__":
    main()
