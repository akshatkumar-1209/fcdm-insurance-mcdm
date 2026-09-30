"""
Module 4: Probabilistic Assessment of MCDM Robustness via Monte Carlo Simulation
================================================================================
Implements:
1. Dirichlet simplex sampling of random weight configurations (Uniform on unit simplex)
2. Empirical distribution analysis of alternative scores and rank stability
3. Pairwise Stochastic Dominance Analysis: P(A_i > A_k)
4. Comparative Benchmarking against TOPSIS
5. Rank correlation statistics (Spearman rho and Kendall tau)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import scipy.stats as stats
from .modified_artasi import ModifiedARTASI


class TOPSIS:
    """
    Technique for Order Preference by Similarity to Ideal Solution (TOPSIS).
    Used as an external benchmark method for comparative validation.
    """

    def __init__(self, weights: np.ndarray, criteria_types: List[str]):
        self.weights = np.asarray(weights, dtype=float)
        self.criteria_types = [t.lower() for t in criteria_types]

    def evaluate(self, X: np.ndarray) -> Dict[str, any]:
        m, n = X.shape
        # Vector normalization
        denom = np.sqrt(np.sum(X**2, axis=0))
        denom = np.where(denom == 0, 1e-12, denom)
        R = X / denom

        # Weighted normalized matrix
        V = R * self.weights

        # Ideal and anti-ideal solutions
        ideal_pos = np.zeros(n)
        ideal_neg = np.zeros(n)

        for j in range(n):
            if self.criteria_types[j] == "max":
                ideal_pos[j] = np.max(V[:, j])
                ideal_neg[j] = np.min(V[:, j])
            else:
                ideal_pos[j] = np.min(V[:, j])
                ideal_neg[j] = np.max(V[:, j])

        # Euclidean separation measures
        S_pos = np.sqrt(np.sum((V - ideal_pos)**2, axis=1))
        S_neg = np.sqrt(np.sum((V - ideal_neg)**2, axis=1))

        # Relative closeness
        C = S_neg / (S_pos + S_neg)

        # Ranking
        rank_indices = np.argsort(-C)
        ranks = np.empty(m, dtype=int)
        for r, idx in enumerate(rank_indices):
            ranks[idx] = r + 1

        return {
            "closeness": C,
            "ranks": ranks,
            "S_pos": S_pos,
            "S_neg": S_neg
        }


class MonteCarloRobustness:
    """
    Monte Carlo Simulation framework for MCDM robustness assessment.
    """

    def __init__(
        self,
        criteria_types: List[str],
        psi_l: float = 1.0,
        psi_u: float = 100.0,
        alpha: float = 0.5,
        phi: float = 1.0
    ):
        self.criteria_types = criteria_types
        self.psi_l = psi_l
        self.psi_u = psi_u
        self.alpha = alpha
        self.phi = phi

    def sample_simplex_weights(self, n_criteria: int, n_samples: int = 1000, seed: Optional[int] = 42) -> np.ndarray:
        """
        Samples weight vectors uniformly from the unit simplex using Dirichlet(1,...,1).
        """
        rng = np.random.default_rng(seed)
        raw_weights = rng.exponential(scale=1.0, size=(n_samples, n_criteria))
        simplex_weights = raw_weights / np.sum(raw_weights, axis=1, keepdims=True)
        return simplex_weights

    def run_simulation(
        self,
        X: np.ndarray,
        n_simulations: int = 1000,
        alternative_names: Optional[List[str]] = None,
        seed: Optional[int] = 42
    ) -> Dict[str, any]:
        """
        Executes n_simulations Monte Carlo iterations, computing scores, ranks,
        dominance matrices, and summary statistics.
        """
        m, n = X.shape
        if alternative_names is None:
            alternative_names = [f"A{i+1}" for i in range(m)]

        sim_weights = self.sample_simplex_weights(n, n_samples=n_simulations, seed=seed)

        scores = np.zeros((n_simulations, m))
        ranks = np.zeros((n_simulations, m), dtype=int)

        # Execute Modified ARTASI for each weight configuration
        for s in range(n_simulations):
            w = sim_weights[s, :]
            model = ModifiedARTASI(
                weights=w,
                criteria_types=self.criteria_types,
                psi_l=self.psi_l,
                psi_u=self.psi_u,
                alpha=self.alpha,
                phi=self.phi
            )
            res = model.evaluate(X)
            scores[s, :] = res["Omega"]
            ranks[s, :] = res["ranks"]

        # 1. Score statistics per alternative
        score_stats = {}
        for i, name in enumerate(alternative_names):
            s_arr = scores[:, i]
            score_stats[name] = {
                "mean": float(np.mean(s_arr)),
                "std": float(np.std(s_arr)),
                "median": float(np.median(s_arr)),
                "q25": float(np.percentile(s_arr, 25)),
                "q75": float(np.percentile(s_arr, 75)),
                "min": float(np.min(s_arr)),
                "max": float(np.max(s_arr))
            }

        # 2. Rank frequency distributions: P(Rank == r)
        rank_distribution = {name: np.zeros(m) for name in alternative_names}
        for i, name in enumerate(alternative_names):
            for r in range(1, m + 1):
                rank_distribution[name][r - 1] = np.mean(ranks[:, i] == r)

        # 3. Pairwise Stochastic Dominance Matrix: P(A_i > A_k)
        dominance_matrix = np.zeros((m, m))
        for i in range(m):
            for k in range(m):
                if i != k:
                    dominance_matrix[i, k] = np.mean(scores[:, i] > scores[:, k])
                else:
                    dominance_matrix[i, k] = 0.5

        return {
            "n_simulations": n_simulations,
            "scores": scores,
            "ranks": ranks,
            "score_stats": score_stats,
            "rank_distribution": rank_distribution,
            "dominance_matrix": dominance_matrix,
            "alternative_names": alternative_names
        }

    def compare_with_topsis(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        alternative_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Runs comparative benchmarking between Modified ARTASI and TOPSIS.
        """
        artasi_model = ModifiedARTASI(
            weights=weights,
            criteria_types=self.criteria_types,
            psi_l=self.psi_l,
            psi_u=self.psi_u,
            alpha=self.alpha,
            phi=self.phi
        )
        artasi_res = artasi_model.evaluate(X, alternative_names)

        topsis_model = TOPSIS(weights=weights, criteria_types=self.criteria_types)
        topsis_res = topsis_model.evaluate(X)

        artasi_ranks = artasi_res["ranks"]
        topsis_ranks = topsis_res["ranks"]

        spearman_rho, spearman_p = stats.spearmanr(artasi_ranks, topsis_ranks)
        kendall_tau, kendall_p = stats.kendalltau(artasi_ranks, topsis_ranks)

        return {
            "artasi_ranks": artasi_ranks,
            "topsis_ranks": topsis_ranks,
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "kendall_tau": float(kendall_tau),
            "kendall_p": float(kendall_p),
            "artasi_scores": artasi_res["Omega"],
            "topsis_scores": topsis_res["closeness"]
        }
