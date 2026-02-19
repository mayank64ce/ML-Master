"""
FHE Interpreter adapter for ML-Master.

Wraps FHE interpreters (BlackBox/WhiteBox) to match ML-Master's
Interpreter.run(code, id, reset_session) interface, converting between
FHE ExecutionResult and ML-Master ExecutionResult.
"""

import logging
import time
from pathlib import Path

from interpreter.interpreter_parallel import ExecutionResult as MLExecutionResult
from fhe.challenge_parser import FHEChallengeSpec, ChallengeType
from fhe.interpreters import create_interpreter

logger = logging.getLogger("ml-master")


class FHEInterpreter:
    """
    Adapter that wraps FHE interpreters to match ML-Master's Interpreter interface.

    ML-Master calls: interpreter.run(code, id, reset_session)
    FHE interpreters call: interpreter.execute(code, testcase_path)

    This adapter bridges the two interfaces.
    """

    def __init__(self, spec: FHEChallengeSpec, cfg):
        self.spec = spec
        self.cfg = cfg
        self.workspace_dir = Path(cfg.workspace_dir)
        self.build_timeout = cfg.fhe.build_timeout
        self.run_timeout = cfg.fhe.run_timeout

    def run(self, code: str, node_id: str = None, reset_session: bool = True) -> MLExecutionResult:
        """
        Execute FHE code and return ML-Master's ExecutionResult.

        Args:
            code: Generated eval() function body (C++)
            node_id: Node identifier (used for workspace isolation)
            reset_session: Ignored (FHE interpreter creates fresh workspace each time)

        Returns:
            ML-Master's ExecutionResult with term_out, exec_time, exc_type, exc_info
        """
        # Create a per-node workspace to avoid conflicts in parallel execution
        node_workspace = self.workspace_dir / f"node_{node_id}" if node_id else self.workspace_dir / "default"

        # Create interpreter for this execution
        interpreter = create_interpreter(
            spec=self.spec,
            workspace_dir=node_workspace,
            build_timeout=self.build_timeout,
            run_timeout=self.run_timeout,
        )

        start_time = time.time()

        try:
            # Determine testcase path
            testcase_path = None
            if self.spec.challenge_type == ChallengeType.BLACK_BOX:
                if self.spec.testcase_dirs:
                    testcase_path = self.spec.testcase_dirs[0]
            elif self.spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
                test_case_json = self.spec.challenge_dir / "tests" / "test_case.json"
                if test_case_json.exists():
                    testcase_path = test_case_json

            # Execute
            fhe_result = interpreter.execute(code, testcase_path)
            exec_time = time.time() - start_time

            # Convert FHE ExecutionResult → ML-Master ExecutionResult
            return self._convert_result(fhe_result, exec_time)

        except Exception as e:
            exec_time = time.time() - start_time
            logger.exception(f"FHE interpreter error for node {node_id}")
            return MLExecutionResult(
                term_out=[f"FHE interpreter error: {e}"],
                exec_time=exec_time,
                exc_type=type(e).__name__,
                exc_info={"error": str(e)},
            )

    def _convert_result(self, fhe_result, exec_time: float) -> MLExecutionResult:
        """Convert FHE ExecutionResult to ML-Master ExecutionResult."""
        # Build combined terminal output
        term_out = []

        if not fhe_result.build_success:
            term_out.append("=== BUILD FAILED ===")
            if fhe_result.error_type:
                term_out.append(f"Error type: {fhe_result.error_type}")
            if fhe_result.error_message:
                term_out.append(f"Error: {fhe_result.error_message}")
            term_out.append("")
            term_out.extend(fhe_result.build_output[-50:])
        elif not fhe_result.run_success:
            term_out.append("=== RUNTIME FAILED ===")
            if fhe_result.error_type:
                term_out.append(f"Error type: {fhe_result.error_type}")
            if fhe_result.error_message:
                term_out.append(f"Error: {fhe_result.error_message}")
            term_out.append("")
            term_out.extend(fhe_result.run_output[-50:])
        else:
            term_out.append("=== EXECUTION SUCCESSFUL ===")
            term_out.append(f"Build time: {fhe_result.build_time:.2f}s")
            term_out.append(f"Run time: {fhe_result.run_time:.2f}s")

            if fhe_result.validation:
                v = fhe_result.validation
                term_out.append(f"Validation: {'PASSED' if v.passed else 'FAILED'}")
                if v.accuracy is not None:
                    term_out.append(f"Accuracy: {v.accuracy:.4f}")
                if v.mean_error > 0:
                    term_out.append(f"Mean error: {v.mean_error:.6f}")
                if v.max_error > 0:
                    term_out.append(f"Max error: {v.max_error:.6f}")
                if v.total_slots > 0:
                    term_out.append(f"Total slots: {v.total_slots}")

        # Determine exception type
        exc_type = None
        exc_info = None
        if not fhe_result.build_success:
            exc_type = fhe_result.error_type or "BUILD_ERROR"
            exc_info = {"error": fhe_result.error_message or "Build failed"}
        elif not fhe_result.run_success:
            exc_type = fhe_result.error_type or "RUNTIME_ERROR"
            exc_info = {"error": fhe_result.error_message or "Runtime error"}

        # Store FHE-specific data in exc_info for the agent to use
        if exc_info is None:
            exc_info = {}
        exc_info["fhe_result"] = {
            "build_success": fhe_result.build_success,
            "run_success": fhe_result.run_success,
            "output_generated": fhe_result.output_generated,
            "accuracy": fhe_result.accuracy,
            "build_time": fhe_result.build_time,
            "run_time": fhe_result.run_time,
        }
        if fhe_result.validation:
            exc_info["fhe_result"]["validation"] = {
                "passed": fhe_result.validation.passed,
                "accuracy": fhe_result.validation.accuracy,
                "mean_error": fhe_result.validation.mean_error,
                "max_error": fhe_result.validation.max_error,
                "total_slots": fhe_result.validation.total_slots,
            }

        return MLExecutionResult(
            term_out=term_out,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
        )

    def cleanup_session(self, session_id):
        """Cleanup (compatibility with ML-Master's Interpreter interface)."""
        pass
