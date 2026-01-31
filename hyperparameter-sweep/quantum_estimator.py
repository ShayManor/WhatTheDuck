from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class ProbEstimate:
    p_hat: float
    ci_low: float
    ci_high: float
    cost: int
    meta: Dict[str, Any]


class QuantumIQAECDF:
    def __init__(self, probs_list, num_qubits: int, max_width: int = 28):
        from classiq import qfunc, qperm, QArray, QBit, QNum, Const, inplace_prepare_state, Constraints, Preferences
        from classiq.applications.iqae.iqae import IQAE

        self.IQAE = IQAE
        self.Constraints = Constraints
        self.Preferences = Preferences
        self.num_qubits = int(num_qubits)
        self.max_width = int(max_width)
        self.PROBS = list(map(float, probs_list))
        self.GLOBAL_INDEX = 0

        @qfunc(synthesize_separately=True)
        def state_preparation(asset: QArray[QBit], ind: QBit):
            inplace_prepare_state(self.PROBS, bound=0, target=asset)
            payoff(asset=asset, ind=ind)

        @qperm
        def payoff(asset: Const[QNum], ind: QBit):
            ind ^= asset < self.GLOBAL_INDEX

        self.state_preparation = state_preparation

    def synthesize_to_qasm(self, threshold_index: int) -> str:
        """Synthesize circuit to OpenQASM for given threshold."""
        from classiq_to_cuda import synthesize_qfunc_to_qasm
        return synthesize_qfunc_to_qasm(
            qfunc=self.state_preparation,
            num_qubits=self.num_qubits,
            probs=self.PROBS,
            threshold_index=threshold_index,
            max_width=self.max_width
        )

    def _cost_from_iqae(self, res) -> int:
        iters = getattr(res, "iterations_data", None)
        if not iters:
            return 0
        total = 0
        for it in iters:
            k = int(getattr(it, "grover_iterations", 0))
            sr = getattr(it, "sample_results", None)
            shots = int(getattr(sr, "num_shots", 0)) if sr is not None else 0
            total += shots * (2 * k + 1)
        return int(total)

    def estimate_tail_prob(self, query, *, epsilon: float, alpha: float,
                           max_total_queries: Optional[int] = None, seed: Optional[int] = None, **_) -> ProbEstimate:
        if query.index is None:
            raise ValueError("query.index required")
        self.GLOBAL_INDEX = int(query.index)

        iqae = self.IQAE(
            state_prep_op=self.state_preparation,
            problem_vars_size=self.num_qubits,
            constraints=self.Constraints(max_width=self.max_width),
            preferences=self.Preferences(machine_precision=self.num_qubits),
        )
        res = iqae.run(epsilon=float(epsilon), alpha=float(alpha))
        p_hat = float(res.estimation)
        ci = list(res.confidence_interval)
        cost = self._cost_from_iqae(res)
        if max_total_queries is not None:
            cost = min(cost, int(max_total_queries))
        return ProbEstimate(p_hat=p_hat, ci_low=float(ci[0]), ci_high=float(ci[1]), cost=int(cost),
                            meta={"epsilon": epsilon, "alpha": alpha, "seed": seed})