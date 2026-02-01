"""
Algorithm-agnostic Classiq -> OpenQASM -> CUDA-Q converter.

Usage:
    from classiq_to_cudaq import classiq_to_qasm, qasm_to_cudaq

    # From a synthesized qprog
    qasm = classiq_to_qasm(qprog)
    kernel = qasm_to_cudaq(qasm)

    # Or from a @qfunc directly
    qasm = synthesize_to_qasm(my_qfunc)
"""

"""
Algorithm-agnostic Classiq -> OpenQASM -> CUDA-Q converter.
"""


def synthesize_qfunc_to_qasm(
        qfunc,
        num_qubits: int,
        probs: list = None,
        threshold_index: int = 0,
        max_width: int = 28
) -> str:
    """
    Wrap a Classiq @qfunc in a main() and synthesize to OpenQASM.

    Args:
        qfunc: The @qfunc to export (ignored - we rebuild it)
        num_qubits: Number of qubits for asset register
        probs: Probability distribution
        threshold_index: Comparator threshold
        max_width: Circuit width constraint

    Returns:
        OpenQASM string
    """
    from classiq import qfunc as qfunc_decorator, qperm, QArray, QBit, QNum, Const, Output
    from classiq import inplace_prepare_state, allocate, synthesize
    from classiq import Constraints, Preferences, set_constraints, set_preferences
    from classiq import create_model, QuantumProgram

    PROBS = list(map(float, probs))
    THRESHOLD = int(threshold_index)
    n = int(num_qubits)

    @qperm
    def payoff(asset: Const[QNum], ind: QBit):
        ind ^= asset < THRESHOLD

    @qfunc_decorator
    def main(asset: Output[QArray[QBit]], ind: Output[QBit]):
        allocate(n, asset)
        allocate(1, ind)
        inplace_prepare_state(PROBS, bound=0, target=asset)
        payoff(asset=asset, ind=ind)

    model = create_model(main)
    model = set_constraints(model, Constraints(max_width=max_width))

    qprog = synthesize(model)
    return qprog.qasm


def qasm_to_cudaq(qasm_str: str):
    """Convert OpenQASM string to CUDA-Q kernel."""
    from qbraid import transpile
    return transpile(qasm_str, "cudaq")


def classiq_to_qasm(qprog, qasm3: bool = False) -> str:
    """
    Extract OpenQASM from a Classiq quantum program.

    Args:
        qprog: Result from classiq.synthesize()
        qasm3: If True, try to get QASM 3.0 format

    Returns:
        OpenQASM string
    """
    from classiq import QuantumProgram
    qp = qprog.qasm
    return qp.qasm


def synthesize_to_qasm(qfunc_or_model, qasm3: bool = False, **constraints) -> str:
    """
    Synthesize a Classiq @qfunc or model and return OpenQASM.

    Args:
        qfunc_or_model: A @qfunc decorated function or a Classiq model
        qasm3: Request QASM 3.0 output
        **constraints: max_width, max_depth, etc.

    Returns:
        OpenQASM string
    """
    from classiq import synthesize, create_model, set_constraints, set_preferences
    from classiq import Constraints, Preferences

    # Handle both qfunc and pre-created model
    if callable(qfunc_or_model):
        model = create_model(qfunc_or_model)
    else:
        model = qfunc_or_model

    # Apply constraints
    if constraints:
        c = Constraints(**{k: v for k, v in constraints.items()
                           if k in ['max_width', 'max_depth', 'max_gate_count']})
        model = set_constraints(model, c)

    # Set QASM3 preference if requested
    if qasm3:
        model = set_preferences(model, Preferences(qasm3=True))

    qprog = synthesize(model)
    return classiq_to_qasm(qprog, qasm3=qasm3)


def qasm_to_cudaq(qasm_str: str):
    """
    Convert OpenQASM string to CUDA-Q kernel.

    Args:
        qasm_str: OpenQASM 2 or 3 string

    Returns:
        cudaq.PyKernel
    """
    from qbraid import transpile
    return transpile(qasm_str, "cudaq")


def classiq_qprog_to_cudaq(qprog):
    """
    One-liner: Classiq qprog -> CUDA-Q kernel.

    Args:
        qprog: Result from classiq.synthesize()

    Returns:
        cudaq.PyKernel
    """
    qasm = classiq_to_qasm(qprog)
    return qasm_to_cudaq(qasm)
