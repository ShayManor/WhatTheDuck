from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Callable

import numpy as np

TailMode = Literal["pnl_leq", "loss_geq"]  # defines the tail event convention


@dataclass(frozen=True)
class TailQuery:
    """
    A single probability query specification:
    estimate p = P(event holds) for a given threshold.

    Use either:
      - threshold_value for continuous-valued distributions, or
      - index for discretized grid-based distributions.
    """
    alpha_target: float  # e.g., 0.07 (used by VaR search logic)
    tail_mode: TailMode  # "pnl_leq" or "loss_geq"
    threshold_value: Optional[float] = None
    index: Optional[int] = None

    # Optional context so one function can handle either continuous or discretized
    grid_points: Optional[np.ndarray] = None  # shape (K,)


@dataclass(frozen=True)
class ProbEstimate:
    """
    Returned by both classical and quantum estimators.
    """
    p_hat: float  # point estimate of tail probability
    ci_low: float  # lower confidence bound
    ci_high: float  # upper confidence bound
    cost: int  # cost in "probability queries" (samples for classical; query count for quantum)
    meta: Dict[str, Any]  # implementation-specific info (shots, grover powers, seed, etc.)


class ClassicalProbEstimator:
    def estimate_tail_prob(
            self,
            query: TailQuery,
            *,
            budget: int,  # number of samples N allowed (hard cap)
            confidence: float = 0.99,  # confidence level for CI (e.g., 0.99)
            **kwargs: Any,
    ) -> ProbEstimate:
        """
        Estimate tail probability p = P(event) under a classical sampling method.

        Must:
          - use <= budget samples (cost == used samples)
          - return a valid CI at requested confidence
          - support multiple methods via kwargs (tilting params, stratification, etc.)
        """
        ...


class QuantumProbEstimator:
    def estimate_tail_prob(
            self,
            query: TailQuery,
            *,
            epsilon: float,  # target additive error on probability (IQAE parameter)
            alpha: float,  # failure probability (IQAE parameter; e.g., 0.01)
            max_total_queries: Optional[int] = None,  # optional hard cap for tuning/sweeps
            seed: Optional[int] = None,
            **kwargs: Any,
    ) -> ProbEstimate:
        """
        Estimate tail probability p = P(event) using quantum amplitude estimation.

        Must:
          - return p_hat and a confidence interval
          - return cost as total "probability queries" (your chosen accounting)
          - record enough meta to audit accounting (shots, Grover powers, etc.)
        """
        ...


@dataclass(frozen=True)
class VarResult:
    var_value: float
    var_index: Optional[int]
    ci_low: float  # CI for VaR value (derived from probability CI + bracket)
    ci_high: float
    total_cost: int
    steps: int
    meta: Dict[str, Any]


def solve_var(
        estimator: Callable[..., ProbEstimate],
        *,
        alpha_target: float,
        tail_mode: TailMode,
        grid_points: np.ndarray,
        # bracket over index space
        lo_index: int,
        hi_index: int,
        # stopping
        value_tol: float,  # tolerance in asset-value units
        prob_tol: float,  # optional tolerance in probability space
        max_steps: int = 64,
        # estimator parameters forwarded (budget or epsilon/alpha)
        estimator_params: Dict[str, Any],
) -> VarResult:
    """
    CI-driven bisection (or other bracketing) to find VaR threshold/index.
    """
    ...
