"""
Module 1: Fuzzy Level-Based Weight Assessment (F-LBWA)
======================================================
Implements Fuzzy LBWA for multi-criteria weighting under epistemic uncertainty.

Key Mathematical Equations:
1. Triangular Fuzzy Number (TFN) aggregation:
   x_j = (min(x_r), mean(x_r), max(x_r))
2. Scale Upper Bound:
   delta = max(|Q_1|, |Q_2|, ..., |Q_k|)
3. Elasticity Coefficient:
   theta > delta (default: delta + 0.01 or empirically 14.01)
4. Fuzzy Influence Function:
   f(C_j) = theta / (j_level * theta + x_j)
   => f_l = theta / (j_level * theta + x_u)
      f_m = theta / (j_level * theta + x_m)
      f_u = theta / (j_level * theta + x_l)
5. Optimal Criterion Fuzzy Weight:
   w_best = 1 / (1 + sum_{k != best} f(C_k))
   => w_l = 1 / (1 + sum f_u)
      w_m = 1 / (1 + sum f_m)
      w_u = 1 / (1 + sum f_l)
6. Non-Optimal Criteria Fuzzy Weights:
   w_j = f(C_j) * w_best
7. Centroid Defuzzification:
   W_j = (w_l + 4 * w_m + w_u) / 6
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class TriangularFuzzyNumber:
    """Represents a Triangular Fuzzy Number (TFN) with bounds (l, m, u)."""

    def __init__(self, l: float, m: float, u: float):
        if not (l <= m <= u):
            # Tolerate tiny float imprecisions
            if abs(l - m) < 1e-9:
                l = min(l, m)
            if abs(m - u) < 1e-9:
                u = max(m, u)
        self.l = float(l)
        self.m = float(m)
        self.u = float(u)

    def __repr__(self) -> str:
        return f"({self.l:.4f}, {self.m:.4f}, {self.u:.4f})"

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.l, self.m, self.u)

    def __add__(self, other: "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
        if isinstance(other, TriangularFuzzyNumber):
            return TriangularFuzzyNumber(self.l + other.l, self.m + other.m, self.u + other.u)
        return TriangularFuzzyNumber(self.l + other, self.m + other, self.u + other)

    def __sub__(self, other: "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
        if isinstance(other, TriangularFuzzyNumber):
            return TriangularFuzzyNumber(self.l - other.u, self.m - other.m, self.u - other.l)
        return TriangularFuzzyNumber(self.l - other, self.m - other, self.u - other)

    def __mul__(self, other) -> "TriangularFuzzyNumber":
        if isinstance(other, TriangularFuzzyNumber):
            # Positive TFN multiplication
            return TriangularFuzzyNumber(self.l * other.l, self.m * other.m, self.u * other.u)
        val = float(other)
        if val >= 0:
            return TriangularFuzzyNumber(self.l * val, self.m * val, self.u * val)
        else:
            return TriangularFuzzyNumber(self.u * val, self.m * val, self.l * val)

    def __truediv__(self, other) -> "TriangularFuzzyNumber":
        if isinstance(other, TriangularFuzzyNumber):
            return TriangularFuzzyNumber(self.l / other.u, self.m / other.m, self.u / other.l)
        val = float(other)
        return TriangularFuzzyNumber(self.l / val, self.m / val, self.u / val)

    def defuzzify(self) -> float:
        """Centroid defuzzification using Simpson's rule: (l + 4m + u) / 6."""
        return (self.l + 4.0 * self.m + self.u) / 6.0


class FuzzyLBWA:
    """
    Fuzzy Level-Based Weight Assessment (F-LBWA) Model.
    """

    def __init__(
        self,
        levels_data: Dict[str, List[Tuple[str, Tuple[float, float, float]]]],
        best_criterion: str = "C29",
        theta: Optional[float] = None
    ):
        """
        Parameters
        ----------
        levels_data : Dict mapping significance level (e.g., 'Q1', 'Q2', ...)
                      to a list of tuples: (criterion_code, (l, m, u))
        best_criterion : Code of the most influential criterion (default: 'C29')
        theta : Elasticity coefficient. If None, set to max(|Q_e|) + 0.01.
        """
        self.levels_data = levels_data
        self.best_criterion = best_criterion
        
        # Determine scale upper bound delta = max(|Q_e|)
        self.delta = max(len(crit_list) for crit_list in levels_data.values())
        
        # Elasticity coefficient theta > delta
        if theta is None:
            self.theta = float(self.delta) + 0.01
        else:
            self.theta = float(theta)
            
        self.influence_functions: Dict[str, TriangularFuzzyNumber] = {}
        self.fuzzy_weights: Dict[str, TriangularFuzzyNumber] = {}
        self.crisp_weights: Dict[str, float] = {}
        self.normalized_weights: Dict[str, float] = {}

    def compute(self) -> Dict[str, any]:
        """
        Executes the F-LBWA algorithm through Steps 1 to 6.
        """
        # Step 5: Compute Fuzzy Influence Functions for each criterion
        for level_idx, (q_name, crit_list) in enumerate(self.levels_data.items(), start=1):
            for code, (xl, xm, xu) in crit_list:
                # Influence function: f(C_jp) = theta / (j * theta + x_jp)
                fl = self.theta / (level_idx * self.theta + xu)
                fm = self.theta / (level_idx * self.theta + xm)
                fu = self.theta / (level_idx * self.theta + xl)
                self.influence_functions[code] = TriangularFuzzyNumber(fl, fm, fu)

        # Step 6: Derive fuzzy weight coefficients
        # 1. Fuzzy weight of the best criterion (C_best)
        sum_fl = sum(f.l for code, f in self.influence_functions.items() if code != self.best_criterion)
        sum_fm = sum(f.m for code, f in self.influence_functions.items() if code != self.best_criterion)
        sum_fu = sum(f.u for code, f in self.influence_functions.items() if code != self.best_criterion)

        w_best_l = 1.0 / (1.0 + sum_fu)
        w_best_m = 1.0 / (1.0 + sum_fm)
        w_best_u = 1.0 / (1.0 + sum_fl)
        
        best_fuzzy_w = TriangularFuzzyNumber(w_best_l, w_best_m, w_best_u)
        self.fuzzy_weights[self.best_criterion] = best_fuzzy_w

        # 2. Fuzzy weights of remaining criteria
        for code, f_val in self.influence_functions.items():
            if code != self.best_criterion:
                w_l = f_val.l * w_best_l
                w_m = f_val.m * w_best_m
                w_u = f_val.u * w_best_u
                self.fuzzy_weights[code] = TriangularFuzzyNumber(w_l, w_m, w_u)

        # 3. Defuzzification (Centroid formula)
        for code, f_weight in self.fuzzy_weights.items():
            self.crisp_weights[code] = f_weight.defuzzify()

        # Normalization (ensure sum to 1.0)
        total_crisp = sum(self.crisp_weights.values())
        for code, w in self.crisp_weights.items():
            self.normalized_weights[code] = w / total_crisp

        return {
            "influence_functions": self.influence_functions,
            "fuzzy_weights": self.fuzzy_weights,
            "crisp_weights": self.crisp_weights,
            "normalized_weights": self.normalized_weights,
            "theta": self.theta,
            "delta": self.delta
        }
