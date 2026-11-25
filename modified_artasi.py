"""
Module 2: Modified ARTASI Multi-Criteria Decision Making Method
==============================================================
Implements Modified ARTASI (Alternative Ranking based on Two-step
Aggregation of Subjective and Objective Information) with negative-value
elimination and reverse sorting for cost criteria.

Key Mathematical Equations:
Step 0: Non-linear translation to remove negative values:
   x_ij = x'_ij + |min_k(x'_kj)|   (for columns containing negative values)
Step 2: Absolute reference bounds (preserving real exponentiation):
   p_max_j = max_i(x_ij) + [max_i(x_ij)]^(1/m)
   p_min_j = min_i(x_ij) - [min_i(x_ij)]^(1/m)
Step 3: Standardization onto [Psi_l, Psi_u] (default [1, 100]):
   phi_ij = ((Psi_u - Psi_l)/(p_max - p_min)) * x_ij + (p_max*Psi_l - p_min*Psi_u)/(p_max - p_min)
   For benefit criteria: zeta_ij = phi_ij
   For cost criteria (reverse-sorting): zeta_ij = -phi_ij + max_k(phi_kj) + min_k(phi_kj)
Step 4: Usefulness degrees:
   Ideal: theta^+_ij = (zeta_ij / max_k(zeta_kj)) * w_j * Psi_u
   Anti-ideal: theta_ij = (min_k(zeta_kj) / zeta_ij) * w_j * Psi_u
               theta^-_ij = -theta_ij + max_k(theta_kj) + min_k(theta_kj)
Step 5: Aggregated usefulness:
   I^+_i = sum_j theta^+_ij
   I^-_i = sum_j theta^-_ij
Step 6: Ultimate utility function & Ranking:
   f(I^+_i) = I^+_i / (I^+_i + I^-_i)
   f(I^-_i) = I^-_i / (I^+_i + I^-_i)
   Omega_i = (I^+_i + I^-_i) * [alpha * f(I^+_i)^phi + (1 - alpha) * f(I^-_i)^phi]^(1/phi)
   (Default: alpha = 0.5, phi = 1.0 => Omega_i = (I^+_i + I^-_i) * 0.5)
"""

from typing import List, Dict, Optional, Union, Tuple
import numpy as np


class ModifiedARTASI:
    """
    Modified ARTASI Multi-Criteria Decision Making Method.
    """

    def __init__(
        self,
        weights: np.ndarray,
        criteria_types: List[str],
        psi_l: float = 1.0,
        psi_u: float = 100.0,
        alpha: float = 0.5,
        phi: float = 1.0
    ):
        """
        Parameters
        ----------
        weights : 1D array of criteria weights (shape: n_criteria, sum=1)
        criteria_types : List of 'max' (benefit) or 'min' (cost) for each criterion
        psi_l : Lower bound of standardization interval (default: 1.0)
        psi_u : Upper bound of standardization interval (default: 100.0)
        alpha : Decision-making impact parameter in [0, 1] (default: 0.5)
        phi : Stabilization parameter in [1, infinity) (default: 1.0)
        """
        self.weights = np.asarray(weights, dtype=float)
        self.criteria_types = [t.lower() for t in criteria_types]
        self.psi_l = float(psi_l)
        self.psi_u = float(psi_u)
        self.alpha = float(alpha)
        self.phi = float(phi)

    def step0_eliminate_negatives(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocesses raw decision matrix: for every criterion containing negative entries,
        translates the column upwards by adding |min(column)| so min becomes 0.
        """
        X_trans = X.astype(float).copy()
        n_crit = X_trans.shape[1]
        for j in range(n_crit):
            col_min = np.min(X_trans[:, j])
            if col_min < 0:
                X_trans[:, j] += abs(col_min)
        return X_trans

    def step2_absolute_bounds(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes absolute maximum and minimum reference bounds:
        p_max_j = max(x_j) + [max(x_j)]^(1/m)
        p_min_j = min(x_j) - [min(x_j)]^(1/m)
        """
        m = X.shape[0]
        n = X.shape[1]
        p_max = np.zeros(n)
        p_min = np.zeros(n)
        
        for j in range(n):
            c_max = np.max(X[:, j])
            c_min = np.min(X[:, j])
            
            # Avoid complex numbers if c_min is slightly negative due to floating point
            c_max_safe = max(0.0, c_max)
            c_min_safe = max(0.0, c_min)
            
            p_max[j] = c_max + (c_max_safe ** (1.0 / m))
            p_min[j] = c_min - (c_min_safe ** (1.0 / m))
            
        return p_max, p_min

    def step3_standardize(
        self, X: np.ndarray, p_max: np.ndarray, p_min: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardizes decision matrix elements to [psi_l, psi_u] and applies
        reverse sorting for cost ('min') criteria.
        """
        m, n = X.shape
        phi = np.zeros((m, n))
        zeta = np.zeros((m, n))

        denom = p_max - p_min
        # Protect against division by zero
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

        for j in range(n):
            phi[:, j] = ((self.psi_u - self.psi_l) / denom[j]) * X[:, j] + \
                        (p_max[j] * self.psi_l - p_min[j] * self.psi_u) / denom[j]

            if self.criteria_types[j] == "max":
                zeta[:, j] = phi[:, j]
            else:
                # Reverse sorting algorithm
                max_phi = np.max(phi[:, j])
                min_phi = np.min(phi[:, j])
                zeta[:, j] = -phi[:, j] + max_phi + min_phi

        return phi, zeta

    def step4_usefulness_degrees(self, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes degrees of usefulness relative to ideal (theta+) and anti-ideal (theta-).
        """
        m, n = zeta.shape
        theta_pos = np.zeros((m, n))
        theta_neg = np.zeros((m, n))

        for j in range(n):
            w_j = self.weights[j]
            max_z = np.max(zeta[:, j])
            min_z = np.min(zeta[:, j])

            max_z_safe = max_z if abs(max_z) > 1e-12 else 1e-12

            # Ideal usefulness
            theta_pos[:, j] = (zeta[:, j] / max_z_safe) * w_j * self.psi_u

            # Anti-ideal intermediate usefulness
            zeta_safe = np.where(np.abs(zeta[:, j]) < 1e-12, 1e-12, zeta[:, j])
            th_temp = (min_z / zeta_safe) * w_j * self.psi_u
            
            # Anti-ideal usefulness with reverse transformation
            max_th = np.max(th_temp)
            min_th = np.min(th_temp)
            theta_neg[:, j] = -th_temp + max_th + min_th

        return theta_pos, theta_neg

    def evaluate(
        self,
        X: np.ndarray,
        alternative_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Full evaluation pipeline returning all intermediate steps, utility scores, and rankings.
        """
        X = np.asarray(X, dtype=float)
        m, n = X.shape

        if alternative_names is None:
            alternative_names = [f"A{i+1}" for i in range(m)]

        # Step 0: Eliminate negative values
        X_trans = self.step0_eliminate_negatives(X)

        # Step 2: Absolute bounds
        p_max, p_min = self.step2_absolute_bounds(X_trans)

        # Step 3: Standardization & reverse sorting
        phi, zeta = self.step3_standardize(X_trans, p_max, p_min)

        # Step 4: Usefulness degrees
        theta_pos, theta_neg = self.step4_usefulness_degrees(zeta)

        # Step 5: Aggregated degrees
        I_pos = np.sum(theta_pos, axis=1)
        I_neg = np.sum(theta_neg, axis=1)

        # Step 6: Utility aggregation
        denom = I_pos + I_neg
        denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        f_pos = I_pos / denom_safe
        f_neg = I_neg / denom_safe

        # Aggregation formula
        term = self.alpha * (f_pos ** self.phi) + (1.0 - self.alpha) * (f_neg ** self.phi)
        Omega = denom * (term ** (1.0 / self.phi))

        # Ranking (higher Omega is better)
        rank_indices = np.argsort(-Omega)
        ranks = np.empty(m, dtype=int)
        for r, idx in enumerate(rank_indices):
            ranks[idx] = r + 1

        return {
            "X_trans": X_trans,
            "p_max": p_max,
            "p_min": p_min,
            "phi": phi,
            "zeta": zeta,
            "theta_pos": theta_pos,
            "theta_neg": theta_neg,
            "I_pos": I_pos,
            "I_neg": I_neg,
            "f_pos": f_pos,
            "f_neg": f_neg,
            "Omega": Omega,
            "ranks": ranks,
            "alternative_names": alternative_names
        }
