"""
Test Program 4: Monte Carlo Robustness Analysis & Comparative Benchmarking
=========================================================================
Executes Module 4 (MonteCarloRobustness) using 1000 Dirichlet-sampled weight
configurations. Assesses probabilistic stability of alternative rankings,
computes pairwise stochastic dominance, and benchmarks Modified ARTASI against
TOPSIS using Spearman and Kendall rank correlations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from fcdm.dataset import get_dataset, INSURANCE_BROKERS
from fcdm.robustness_montecarlo import MonteCarloRobustness, TOPSIS


BROKER_COLORS = {
    "AJG": "#E67E22",   # Orange
    "AON": "#2980B9",   # Blue
    "BRO": "#27AE60",   # Green
    "MMC": "#8E44AD",   # Purple
    "WTW": "#E74C3C"    # Red
}


def print_section(title):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def main():
    print_section("TEST PROGRAM 4: MONTE CARLO ROBUSTNESS ANALYSIS & COMPARATIVE BENCHMARKING")
    print("Reference: Applied Soft Computing Journal 190 (2026) 114557 (Sections 3.3, 4.6, 4.8)")

    data = get_dataset()
    X = data["X"]
    weights = data["weights"]
    criteria_types = data["criteria_types"]
    alt_names = data["alternative_names"]
    broker_dict = {b["code"]: b["name"] for b in INSURANCE_BROKERS}

    mc = MonteCarloRobustness(
        criteria_types=criteria_types,
        psi_l=1.0,
        psi_u=100.0,
        alpha=0.5,
        phi=1.0
    )

    # ─── 1. Run Monte Carlo Simulation (1000 weight configurations) ────────────
    print("\n[1] Running 1000 Monte Carlo Simulations (Dirichlet Simplex Sampling)...")
    sim = mc.run_simulation(
        X=X,
        n_simulations=1000,
        alternative_names=alt_names,
        seed=42
    )
    print("    Done. Each simulation re-runs Modified ARTASI with a fresh weight vector.")

    # ─── 2. Score Distribution Statistics ─────────────────────────────────────
    print(f"\n[2] Simulated Score Statistics (n = {sim['n_simulations']}) per Alternative:")
    print(f"{'Broker':<6} {'Name':<44} {'Mean':<8} {'Std':<8} {'Median':<8} {'Q25':<8} {'Q75':<8}")
    print("-" * 90)
    for code in alt_names:
        s = sim["score_stats"][code]
        name = broker_dict[code]
        print(f"{code:<6} {name:<44} {s['mean']:<8.2f} {s['std']:<8.2f} {s['median']:<8.2f} {s['q25']:<8.2f} {s['q75']:<8.2f}")

    # ─── 3. Rank Frequency Distribution ───────────────────────────────────────
    print(f"\n[3] Rank Frequency Distribution (Empirical P(Rank = r) per Broker):")
    header = f"{'Broker':<6} " + "".join([f"P(Rank={r})"[:9].ljust(11) for r in range(1, 6)])
    print(header)
    print("-" * 65)
    for code in alt_names:
        dist = sim["rank_distribution"][code]
        row = f"{code:<6} " + "".join([f"{p:.3f}".ljust(11) for p in dist])
        print(row)

    # ─── 4. Pairwise Stochastic Dominance ─────────────────────────────────────
    print(f"\n[4] Pairwise Stochastic Dominance Matrix P(A_row > A_col):")
    print(f"{'':>5}", end="")
    for code in alt_names:
        print(f"{code:>8}", end="")
    print()
    print("-" * 50)
    for i, code_i in enumerate(alt_names):
        print(f"{code_i:>5}", end="")
        for j, code_j in enumerate(alt_names):
            prob = sim["dominance_matrix"][i, j]
            print(f"{prob:>8.3f}", end="")
        print()

    # ─── 5. Comparative Benchmarking vs TOPSIS ────────────────────────────────
    print(f"\n[5] Comparative Benchmarking: Modified ARTASI vs. TOPSIS (Expert Weights):")
    bench = mc.compare_with_topsis(X, weights, alternative_names=alt_names)

    print(f"\n{'Broker':<6} {'ARTASI Rank':<14} {'TOPSIS Rank':<14} {'ARTASI Score':<15} {'TOPSIS Closeness':<18}")
    print("-" * 70)
    for i, code in enumerate(alt_names):
        print(f"{code:<6} {bench['artasi_ranks'][i]:<14} {bench['topsis_ranks'][i]:<14} {bench['artasi_scores'][i]:<15.4f} {bench['topsis_scores'][i]:<18.4f}")

    print(f"\n    Rank Correlation Statistics:")
    print(f"    - Spearman rho = {bench['spearman_rho']:.4f}  (p = {bench['spearman_p']:.4f})")
    print(f"    - Kendall tau  = {bench['kendall_tau']:.4f}  (p = {bench['kendall_p']:.4f})")

    # ─── 6. Visualization ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0F1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # ── 6a. Score Distributions (Violin Plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1A1D2E")
    scores_all = [sim["scores"][:, i] for i in range(5)]
    vp = ax1.violinplot(scores_all, positions=range(5), showmedians=True, showextrema=True)
    for pc, code in zip(vp["bodies"], alt_names):
        pc.set_facecolor(BROKER_COLORS[code])
        pc.set_alpha(0.7)
    vp["cmedians"].set_color("white")
    vp["cmins"].set_color("gray")
    vp["cmaxes"].set_color("gray")
    vp["cbars"].set_color("gray")
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(alt_names, color="white")
    ax1.set_title("ARTASI Score Distributions (1000 MC Runs)", color="white", fontsize=11, pad=10)
    ax1.set_ylabel("Omega (Utility Score)", color="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#3D4066")

    # ── 6b. Rank Frequency Stacked Bar
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1A1D2E")
    rank_colors = ["#2ECC71", "#27AE60", "#F39C12", "#E67E22", "#E74C3C"]
    x_pos = np.arange(5)
    bottoms = np.zeros(5)
    for r in range(5):
        vals = [sim["rank_distribution"][code][r] for code in alt_names]
        bars = ax2.bar(x_pos, vals, bottom=bottoms, color=rank_colors[r], label=f"Rank {r+1}", alpha=0.85)
        bottoms += np.array(vals)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(alt_names, color="white")
    ax2.set_title("Rank Frequency Distribution (1000 MC Runs)", color="white", fontsize=11, pad=10)
    ax2.set_ylabel("Proportion", color="white")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", framealpha=0.3, labelcolor="white", fontsize=8)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#3D4066")

    # ── 6c. Stochastic Dominance Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1A1D2E")
    dom = sim["dominance_matrix"]
    im = ax3.imshow(dom, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax3.set_xticks(range(5))
    ax3.set_yticks(range(5))
    ax3.set_xticklabels(alt_names, color="white")
    ax3.set_yticklabels(alt_names, color="white")
    ax3.set_title("Pairwise Stochastic Dominance P(row > col)", color="white", fontsize=11, pad=10)
    for i in range(5):
        for j in range(5):
            ax3.text(j, i, f"{dom[i,j]:.2f}", ha="center", va="center",
                     color="black" if 0.3 < dom[i,j] < 0.7 else "white", fontsize=9)
    plt.colorbar(im, ax=ax3, shrink=0.85)

    # ── 6d. ARTASI vs TOPSIS Rank Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1A1D2E")
    artasi_r = bench["artasi_ranks"]
    topsis_r = bench["topsis_ranks"]
    x = np.arange(5)
    width = 0.35
    b1 = ax4.bar(x - width/2, artasi_r, width, label="ARTASI Rank", color="#8E44AD", alpha=0.85)
    b2 = ax4.bar(x + width/2, topsis_r, width, label="TOPSIS Rank", color="#2980B9", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(alt_names, color="white")
    ax4.set_yticks([1, 2, 3, 4, 5])
    ax4.invert_yaxis()
    ax4.set_title(f"ARTASI vs TOPSIS Rankings\n(Spearman ρ={bench['spearman_rho']:.3f}, Kendall τ={bench['kendall_tau']:.3f})", color="white", fontsize=10, pad=10)
    ax4.set_ylabel("Rank (1=Best)", color="white")
    ax4.legend(framealpha=0.3, labelcolor="white", fontsize=9)
    ax4.tick_params(colors="white")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#3D4066")
    for bar in b1:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{int(bar.get_height())}', ha='center', va='bottom', color='white', fontsize=9)
    for bar in b2:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{int(bar.get_height())}', ha='center', va='bottom', color='white', fontsize=9)

    fig.suptitle("Module 4: Monte Carlo Robustness & Comparative Analysis\n"
                 "F-LBWA + Modified ARTASI | Insurance Brokerage Firms 2024",
                 color="white", fontsize=14, y=0.98)

    out_file = "monte_carlo_results.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[6] Visualization saved to: {out_file}")
    plt.close()

    print("\n>>> Monte Carlo Robustness Analysis Completed Successfully!")
    print("=" * 90)


if __name__ == "__main__":
    main()
