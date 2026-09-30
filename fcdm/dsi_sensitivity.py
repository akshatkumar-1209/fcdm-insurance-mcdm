"""
Module 3: Decision Stability Intervals (DSI)
============================================
Implements systematic what-if sensitivity analysis on criterion weights
and decision-matrix entries to quantify decision stability and identify
the numerical tolerance intervals [w_min, w_max] and [x_min, x_max]
within which the rank order remains invariant.

Reference Paper Equations / Procedure:
Let w = (w_1, ..., w_n) be the original vector of criterion weights (sum w_j = 1).
For each criterion k:
  1. Increase w_k by omega, decrease others proportionally by omega / (n - 1)
  2. Re-run MCDM method (Modified ARTASI)
  3. If ranking is unchanged, continue; if ranking changes, stop and record w_max_k.
  4. Mirror test for decreases: decrease w_k by omega, increase others by omega / (n - 1)
     until ranking changes, record w_min_k.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from .modified_artasi import ModifiedARTASI


class DecisionStabilityIntervals:
    """
    Computes Decision Stability Intervals (DSI) for criterion weights
    and decision-matrix entries under the Modified ARTASI framework.
    """

    def __init__(
        self,
        artasi_model: ModifiedARTASI,
        step_size: float = 0.0005,
        matrix_step_ratio: float = 0.005,
        strict_ranking: bool = True
    ):
        """
        Parameters
        ----------
        artasi_model : Initialized ModifiedARTASI instance
        step_size : Step size omega for weight perturbation (default: 0.0005)
        matrix_step_ratio : Relative step size for decision matrix perturbations (default: 0.005)
        strict_ranking : If True, checks that the entire rank order (all positions) is preserved.
                         If False, checks that the top-ranked alternative remains invariant.
        """
        self.model = artasi_model
        self.step_size = float(step_size)
        self.matrix_step_ratio = float(matrix_step_ratio)
        self.strict_ranking = strict_ranking

    def _ranking_matches(self, base_ranks: np.ndarray, new_ranks: np.ndarray) -> bool:
        if self.strict_ranking:
            return np.array_equal(base_ranks, new_ranks)
        else:
            # Top-1 invariance
            return np.argmin(base_ranks) == np.argmin(new_ranks)

    def compute_weight_dsi(
        self,
        X: np.ndarray,
        criteria_codes: Optional[List[str]] = None,
        max_iterations: int = 2000
    ) -> List[Dict[str, any]]:
        """
        Computes Decision Stability Intervals [w_min, w_max] for all criteria weights.
        """
        m, n = X.shape
        base_res = self.model.evaluate(X)
        base_ranks = base_res["ranks"]
        orig_weights = self.model.weights.copy()

        if criteria_codes is None:
            criteria_codes = [f"C{j+1}" for j in range(n)]

        dsi_results = []

        for k in range(n):
            w_orig = orig_weights[k]

            # 1. Test for Maximum Allowable Increase (Upper Bound)
            w_max = w_orig
            curr_w = orig_weights.copy()

            for _ in range(max_iterations):
                test_w = curr_w.copy()
                test_w[k] += self.step_size
                delta_other = self.step_size / (n - 1)
                for j in range(n):
                    if j != k:
                        test_w[j] -= delta_other

                # Check non-negativity constraint
                if np.any(test_w < 0) or test_w[k] > 1.0:
                    break

                # Re-run model
                temp_model = ModifiedARTASI(
                    weights=test_w,
                    criteria_types=self.model.criteria_types,
                    psi_l=self.model.psi_l,
                    psi_u=self.model.psi_u,
                    alpha=self.model.alpha,
                    phi=self.model.phi
                )
                res = temp_model.evaluate(X)

                if self._ranking_matches(base_ranks, res["ranks"]):
                    w_max = test_w[k]
                    curr_w = test_w
                else:
                    break

            # 2. Test for Maximum Allowable Decrease (Lower Bound)
            w_min = w_orig
            curr_w = orig_weights.copy()

            for _ in range(max_iterations):
                test_w = curr_w.copy()
                test_w[k] -= self.step_size
                delta_other = self.step_size / (n - 1)
                for j in range(n):
                    if j != k:
                        test_w[j] += delta_other

                # Check constraints
                if test_w[k] < 0:
                    w_min = max(0.0, w_min)
                    break

                temp_model = ModifiedARTASI(
                    weights=test_w,
                    criteria_types=self.model.criteria_types,
                    psi_l=self.model.psi_l,
                    psi_u=self.model.psi_u,
                    alpha=self.model.alpha,
                    phi=self.model.phi
                )
                res = temp_model.evaluate(X)

                if self._ranking_matches(base_ranks, res["ranks"]):
                    w_min = test_w[k]
                    curr_w = test_w
                else:
                    break

            dsi_results.append({
                "criterion": criteria_codes[k],
                "original_weight": float(w_orig),
                "lower_limit": float(w_min),
                "upper_limit": float(w_max),
                "interval_width": float(w_max - w_min),
                "relative_width": float((w_max - w_min) / w_orig) if w_orig > 0 else 0.0
            })

        return dsi_results

    def compute_matrix_dsi(
        self,
        X: np.ndarray,
        alternative_names: Optional[List[str]] = None,
        criteria_codes: Optional[List[str]] = None,
        max_iterations: int = 500
    ) -> Dict[str, any]:
        """
        Computes stability intervals [x_min, x_max] for each cell in the decision matrix.
        """
        m, n = X.shape
        base_res = self.model.evaluate(X)
        base_ranks = base_res["ranks"]

        if alternative_names is None:
            alternative_names = [f"A{i+1}" for i in range(m)]
        if criteria_codes is None:
            criteria_codes = [f"C{j+1}" for j in range(n)]

        matrix_dsi = {}

        for i in range(m):
            alt_name = alternative_names[i]
            matrix_dsi[alt_name] = []

            for j in range(n):
                val_orig = X[i, j]
                # Determine step size relative to absolute magnitude
                mag = abs(val_orig) if abs(val_orig) > 1e-4 else 1.0
                step = mag * self.matrix_step_ratio

                # 1. Increase test
                ul = val_orig
                curr_X = X.copy()
                for _ in range(max_iterations):
                    curr_X[i, j] += step
                    res = self.model.evaluate(curr_X)
                    if self._ranking_matches(base_ranks, res["ranks"]):
                        ul = curr_X[i, j]
                    else:
                        break

                # 2. Decrease test
                ll = val_orig
                curr_X = X.copy()
                for _ in range(max_iterations):
                    curr_X[i, j] -= step
                    res = self.model.evaluate(curr_X)
                    if self._ranking_matches(base_ranks, res["ranks"]):
                        ll = curr_X[i, j]
                    else:
                        break

                matrix_dsi[alt_name].append({
                    "criterion": criteria_codes[j],
                    "original_value": float(val_orig),
                    "lower_limit": float(ll),
                    "upper_limit": float(ul),
                    "interval_width": float(ul - ll)
                })

        return matrix_dsi
