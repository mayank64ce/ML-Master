"""FHE interpreters for ML-Master."""

from .base import BaseInterpreter, ExecutionResult, ValidationResult
from .black_box import BlackBoxInterpreter
from .white_box import WhiteBoxInterpreter

from fhe.challenge_parser import ChallengeType, FHEChallengeSpec


def create_interpreter(
    spec: FHEChallengeSpec,
    workspace_dir,
    build_timeout: int = 600,
    run_timeout: int = 600,
) -> BaseInterpreter:
    """Create the appropriate interpreter for a challenge type."""
    interpreters = {
        ChallengeType.BLACK_BOX: BlackBoxInterpreter,
        ChallengeType.WHITE_BOX_OPENFHE: WhiteBoxInterpreter,
    }

    interpreter_class = interpreters.get(spec.challenge_type, WhiteBoxInterpreter)
    return interpreter_class(
        spec=spec,
        workspace_dir=workspace_dir,
        build_timeout=build_timeout,
        run_timeout=run_timeout,
    )


__all__ = [
    "BaseInterpreter",
    "ExecutionResult",
    "ValidationResult",
    "BlackBoxInterpreter",
    "WhiteBoxInterpreter",
    "create_interpreter",
]
