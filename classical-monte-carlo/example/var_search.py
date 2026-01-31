# var_search.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from structures import TailMode, TailQuery, ProbEstimate, VarResult


def _tighten_estimator_params(
    estimator_params: Dict[str, Any],
    refine_mult: float,
) -> Dict[str, Any]:
    """
    Generic refinement rule:
      - If classical (has 'budget'): multiply budget.
      - If quantum (has 'epsilon'): divide epsilon.
    """
    new_params = dict(estimator_params)
    if "budget" in new_params and new_params["budget"] is not None:
        new_params["budget"] = int(max(1, round(new_params["budget"] * refine_mult)))
    if "epsilon" in new_params and new_params["epsilon"] is not None:
        new_params["epsilon"] = float(new_params["epsilon"]) / float(refine_mult)
    return new_params


def solve_var(
    estimator: Callable[..., ProbEstimate],
    *,
    alpha_target: float,
    tail_mode: TailMode,
    grid_points: np.ndarray,
    lo_index: int,
    hi_index: int,
    value_tol: float,
    prob_tol: float,
    max_steps: int = 64,
    estimator_params: Dict[str, Any],
) -> VarResult:
    """
    CI-driven bisection to find the smallest index i such that CDF(i) >= alpha_target,
    where the estimator answers tail probability queries p = P(event).

    Assumes discretized grid + index queries.
    """
    if lo_index < 0 or hi_index >= len(grid_points) or lo_index > hi_index:
        raise ValueError("Invalid [lo_index, hi_index] bracket.")

    total_cost = 0
    steps = 0
    trace: List[Dict[str, Any]] = []

    # Optional refinement knobs (sweepable)
    max_refinements: int = int(estimator_params.pop("max_refinements", 2))
    refine_mult: float = float(estimator_params.pop("refine_mult", 2.0))

    lo, hi = lo_index, hi_index

    def bracket_value_width(a: int, b: int) -> float:
        return float(grid_points[b] - grid_points[a])

    while lo < hi and steps < max_steps:
        mid = (lo + hi) // 2

        params = dict(estimator_params)
        est: Optional[ProbEstimate] = None

        for refine_round in range(max_refinements + 1):
            q = TailQuery(
                alpha_target=alpha_target,
                tail_mode=tail_mode,
                index=mid,
                grid_points=grid_points,
            )
            est = estimator(q, **params)
            steps += 1
            total_cost += int(est.cost)

            trace.append(
                {
                    "mid": mid,
                    "p_hat": est.p_hat,
                    "ci_low": est.ci_low,
                    "ci_high": est.ci_high,
                    "cost": est.cost,
                    "params": dict(params),
                    "refine_round": refine_round,
                    "bracket": (lo, hi),
                }
            )

            # CI-decision:
            # - if alpha_target is above CI => CDF(mid) too small => move right
            # - if alpha_target is below CI => CDF(mid) too large => move left
            # - else CI overlaps alpha_target => ambiguous; refine or fall back to p_hat
            if alpha_target > est.ci_high + prob_tol:
                lo = mid + 1
                break
            if alpha_target < est.ci_low - prob_tol:
                hi = mid
                break

            # Ambiguous region
            if refine_round < max_refinements:
                params = _tighten_estimator_params(params, refine_mult)
                continue

            # Final fallback: use point estimate direction
            if est.p_hat < alpha_target:
                lo = mid + 1
            else:
                hi = mid
            break

        if lo < hi and bracket_value_width(lo, hi) <= value_tol:
            break

    var_index = lo
    var_value = float(grid_points[var_index])

    # VaR CI from bracket endpoints (coarse but honest)
    ci_low = float(grid_points[lo])
    ci_high = float(grid_points[hi]) if hi < len(grid_points) else float(grid_points[-1])

    return VarResult(
        var_value=var_value,
        var_index=var_index,
        ci_low=min(ci_low, ci_high),
        ci_high=max(ci_low, ci_high),
        total_cost=int(total_cost),
        steps=int(steps),
        meta={"trace": trace, "final_bracket": (lo, hi)},
    )
