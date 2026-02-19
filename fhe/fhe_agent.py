"""
FHE Agent for ML-Master.

Extends MCTSAgent with FHE-specific:
- Prompt generation (draft/improve/debug) for C++ eval() body
- Code extraction (C++ instead of Python)
- Execution result parsing (FHE validation metrics instead of CSV checks)
- Template-aware variable documentation

Reuses ML-Master's MCTS tree search, node management, and LLM backends.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, cast

from backend import FunctionSpec, compile_prompt_to_md, query, r1_query, gpt_query
from interpreter.interpreter_parallel import ExecutionResult
from search.mcts_node import MCTSNode
from agent.mcts_agent import MCTSAgent
from utils.config_mcts import Config
from utils.metric import MetricValue, WorstMetricValue
from utils.response import wrap_code, extract_review

from fhe.challenge_parser import FHEChallengeSpec, ChallengeType, Scheme, Library

logger = logging.getLogger("ml-master")


# FHE-specific feedback function spec
fhe_review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the build failed, runtime error occurred, or no valid output was generated.",
            },
            "has_output": {
                "type": "boolean",
                "description": "true if the solution produced valid encrypted output and validation metrics.",
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence summary of build/run status and accuracy results.",
            },
            "metric": {
                "type": "number",
                "description": "Accuracy/correctness score (0.0-1.0). null if execution failed.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "false (higher accuracy is better for FHE challenges).",
            },
        },
        "required": ["is_bug", "has_output", "summary", "metric", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the FHE solution execution.",
)


def extract_cpp_code(text: str) -> str:
    """Extract C++ code blocks from LLM response text.

    Handles:
    - ```cpp ... ``` blocks
    - ```c++ ... ``` blocks
    - ``` ... ``` blocks with C++ content
    - ### CONFIG ### and ### CODE ### sections for white-box challenges
    """
    # For white-box: check for CONFIG and CODE sections
    if "### CONFIG ###" in text or "### CODE ###" in text:
        return _extract_whitebox_code(text)

    # Find code blocks
    parsed_codes = []

    # Match ```cpp, ```c++, or ``` blocks
    matches = re.findall(r"```(?:cpp|c\+\+)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        if match.strip():
            parsed_codes.append(match.strip())

    if parsed_codes:
        # Return the longest code block (most likely the main implementation)
        return max(parsed_codes, key=len)

    # Fallback: look for C++-like content without code fences
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if any(kw in line for kw in ["#include", "m_cc->", "EvalMult", "EvalAdd",
                                      "Ciphertext", "m_OutputC", "m_InputC"]):
            in_code = True
        if in_code:
            code_lines.append(line)

    return "\n".join(code_lines) if code_lines else text


def _extract_whitebox_code(text: str) -> str:
    """Extract code and optional CONFIG section for white-box challenges."""
    # Find all code blocks
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        # Try incomplete code blocks (missing closing ```)
        incomplete_pattern = r"```(?:\w+)?\n(.*?)(?=```|\Z)"
        matches = re.findall(incomplete_pattern, text, re.DOTALL)

    config_json = None
    main_code = None

    config_keys = ['"indexes_for_rotation_key"', '"mult_depth"', '"ring_dimension"',
                   '"poly_degree"', '"plaintext_modulus"', '"scheme"', '"batch_size"']

    for block in matches:
        block_lower = block.lower()
        if "### config" in block_lower or any(key in block for key in config_keys):
            config_json = block.strip()
            if config_json.lower().startswith("### config"):
                config_json = "\n".join(config_json.split("\n")[1:]).strip()
        elif "### code" in block_lower:
            main_code = block.strip()
            if main_code.lower().startswith("### code"):
                main_code = "\n".join(main_code.split("\n")[1:]).strip()
        elif any(kw in block for kw in ["EvalMult", "EvalAdd", "m_cc->", "Ciphertext", "m_OutputC"]):
            main_code = block.strip()
        elif not main_code and len(block.strip()) > 50:
            main_code = block.strip()

    result_parts = []
    if config_json:
        result_parts.append("### CONFIG ###")
        result_parts.append(config_json)
    if main_code:
        result_parts.append("\n### CODE ###")
        result_parts.append(main_code)
    elif matches:
        result_parts.append("\n### CODE ###")
        result_parts.append(max(matches, key=len).strip())

    if result_parts:
        return "\n".join(result_parts)

    return text


class FHEAgent(MCTSAgent):
    """
    FHE-specific MCTS agent.

    Overrides MCTSAgent methods to:
    - Generate C++ eval() body instead of Python scripts
    - Use FHE-specific prompts (template variables, constraints, scheme)
    - Parse FHE execution results (accuracy from Docker output)
    - Skip CSV submission checks
    """

    def __init__(self, task_desc, cfg: Config, journal, spec: FHEChallengeSpec):
        super().__init__(task_desc=task_desc, cfg=cfg, journal=journal)
        self.spec = spec

        # Load template context
        self._template_context = self._load_template_context()

        # Extract template variables
        self._template_vars = self._extract_template_variables()

    # ==================== Template Loading ====================

    def _load_template_context(self) -> dict[str, str]:
        """Load relevant template files for prompts."""
        context = {}

        if not self.spec.template_dir or not self.spec.template_dir.exists():
            return context

        if self.spec.challenge_type == ChallengeType.BLACK_BOX:
            for name in ["yourSolution.cpp", "yourSolution.h"]:
                path = self.spec.template_dir / name
                if path.exists():
                    context[name] = path.read_text()

        elif self.spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
            for name in ["yourSolution.cpp", "yourSolution.h", "config.json"]:
                path = self.spec.template_dir / name
                if path.exists():
                    context[name] = path.read_text()

        return context

    def _extract_template_variables(self) -> dict:
        """Extract C++ template variables from header files."""
        result = {
            "context": "m_cc",
            "inputs": ["m_InputC"],
            "output": "m_OutputC",
            "public_key": "m_PublicKey",
        }

        header_content = None
        for name in ["yourSolution.h"]:
            if name in self._template_context:
                header_content = self._template_context[name]
                break

        if not header_content and self.spec.template_dir:
            header_path = self.spec.template_dir / "yourSolution.h"
            if header_path.exists():
                header_content = header_path.read_text()

        if not header_content:
            return result

        # Extract Ciphertext member variables
        ciphertext_vars = re.findall(r"Ciphertext<DCRTPoly>\s+(m_\w+)", header_content)
        inputs = []
        output = None
        for var in ciphertext_vars:
            if "Output" in var:
                output = var
            else:
                inputs.append(var)

        if inputs:
            result["inputs"] = inputs
        if output:
            result["output"] = output

        cc_match = re.search(r"CryptoContext<DCRTPoly>\s+(m_\w+)", header_content)
        if cc_match:
            result["context"] = cc_match.group(1)

        pk_match = re.search(r"PublicKey<DCRTPoly>\s+(m_\w+)", header_content)
        if pk_match:
            result["public_key"] = pk_match.group(1)

        return result

    def _format_variable_docs(self) -> list[str]:
        """Format C++ variable documentation for prompts."""
        v = self._template_vars
        lines = [
            "IMPORTANT - Use these EXACT variable names (class members):",
            f"  {v['context']}       - CryptoContext<DCRTPoly> (NOT 'cc')",
        ]

        if len(v["inputs"]) == 1:
            lines.append(f"  {v['inputs'][0]}   - Input Ciphertext<DCRTPoly>")
        else:
            for inp in v["inputs"]:
                desc = inp.replace("m_", "").replace("C", " Ciphertext")
                lines.append(f"  {inp}  - {desc} (Ciphertext<DCRTPoly>)")

        lines.extend([
            f"  {v['output']}  - Output Ciphertext (ASSIGN to this, don't return)",
            f"  {v['public_key']} - PublicKey<DCRTPoly>",
            "",
            f"The eval() function is void - assign result to {v['output']}:",
            f"  {v['output']} = result;  // CORRECT",
            "  return result;       // WRONG - eval() is void!",
        ])

        return lines

    # ==================== Prompt Building ====================

    def _build_challenge_prompt(self) -> dict:
        """Build challenge specification section for prompts."""
        spec = self.spec
        prompt = {
            "Task": spec.task,
            "Description": spec.task_description,
            "Scheme": spec.scheme.value if spec.scheme else "CKKS",
            "Library": spec.library.value if spec.library else "OpenFHE",
        }

        if spec.output_format:
            prompt["Expected Output"] = spec.output_format

        if spec.constraints:
            prompt["Constraints"] = {
                "Multiplicative Depth": spec.constraints.depth,
                "Batch Size": spec.constraints.batch_size,
                "Input Range": f"[{spec.constraints.input_range[0]}, {spec.constraints.input_range[1]}]",
            }

        if spec.keys:
            prompt["Available Keys"] = {
                "Public Key": spec.keys.public,
                "Multiplication Key": spec.keys.multiplication,
                "Rotation Indices": spec.keys.rotation_indices,
            }

        if spec.useful_links:
            links_info = []
            for link in spec.useful_links:
                link_str = f"- {link['name']}: {link['url']}"
                if link.get("description"):
                    link_str += f" ({link['description']})"
                links_info.append(link_str)
            prompt["Useful Resources"] = links_info

        return {"Challenge Specification": prompt}

    def _build_template_prompt(self) -> dict:
        """Build template context section showing key template files."""
        if not self._template_context:
            return {}

        templates = {}
        for name, content in self._template_context.items():
            if len(content) > 3000:
                # Truncate but keep important sections
                important_keywords = [
                    "class ", "struct ", "void ", "eval(",
                    "Ciphertext", "CryptoContext", "EvalMult", "EvalAdd",
                    "#include", "public:", "private:",
                ]
                lines = content.split("\n")
                relevant_sections = []
                i = 0
                while i < len(lines):
                    if any(kw in lines[i] for kw in important_keywords):
                        start = max(0, i - 2)
                        end = min(len(lines), i + 20)
                        relevant_sections.append("\n".join(lines[start:end]))
                        i = end
                    else:
                        i += 1
                if relevant_sections:
                    content = "\n\n// ... (truncated) ...\n\n".join(relevant_sections)
                else:
                    content = content[:1500] + "\n\n// ... (truncated) ...\n\n" + content[-500:]
            templates[name] = content

        return {"Template Files": templates}

    def _build_eval_instructions(self) -> dict:
        """Build instructions for what to implement."""
        spec = self.spec

        if spec.challenge_type == ChallengeType.BLACK_BOX:
            return {
                "What to Implement": [
                    "Implement ONLY the body of the eval() function.",
                    "The eval() function is in yourSolution.cpp.",
                    "Input ciphertexts are already deserialized and available.",
                    f"Output must be assigned to {self._template_vars['output']}.",
                    "Do NOT modify main.cpp, Dockerfile, or CMakeLists.txt.",
                ],
                "C++ Template Variables": self._format_variable_docs(),
            }

        elif spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
            config_content = self._template_context.get("config.json", "")

            result = {
                "What to Implement": [
                    "Implement the body of the eval() function.",
                    "Template handles CryptoContext creation and key generation.",
                    "fherma-validator will build, run, and validate your solution.",
                    "",
                    "You MAY also modify config.json parameters if needed:",
                    "- Add rotation key indexes if your algorithm requires them",
                    "- Adjust mult_depth if you need more levels",
                ],
                "C++ Template Variables": self._format_variable_docs(),
            }

            if config_content:
                result["Current config.json"] = config_content
                result["Config Modification (Optional)"] = [
                    "To modify config, provide the COMPLETE config.json:",
                    "",
                    "### CONFIG ###",
                    "<your complete modified config.json here>",
                    "",
                    "Common modifications:",
                    "- indexes_for_rotation_key: add rotation indices you need",
                    "- mult_depth: increase if you get 'depth exhausted' error",
                ]

            return result

        return {}

    def _build_response_format(self) -> dict:
        """Build expected response format instructions."""
        if self.spec.challenge_type == ChallengeType.BLACK_BOX:
            return {
                "Response Format": (
                    "Provide your implementation in this format:\n\n"
                    "1. Brief explanation of your approach (2-3 sentences)\n\n"
                    "2. The eval() function body:\n"
                    "```cpp\n"
                    "// Your implementation here\n"
                    "// Do NOT include function signature or braces\n"
                    "```\n\n"
                    "IMPORTANT: Only provide the code that goes INSIDE eval(), "
                    "not the function signature or surrounding code."
                )
            }

        elif self.spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
            return {
                "Response Format": (
                    "Provide BOTH sections (config.json controls crypto parameters):\n\n"
                    "### CONFIG ###\n"
                    "{\n"
                    '  "mult_depth": 20,\n'
                    '  "indexes_for_rotation_key": [1, 2, 4],\n'
                    '  "scheme": "CKKS"\n'
                    "}\n\n"
                    "### CODE ###\n"
                    "// Your eval() function body here\n\n"
                    "IMPORTANT: Both sections are required for white-box challenges."
                )
            }

        return {}

    # ==================== Override MCTSAgent Methods ====================

    def update_data_preview(self):
        """Override: show template context instead of data file preview."""
        preview_parts = []
        preview_parts.append(f"Challenge: {self.spec.challenge_name}")
        preview_parts.append(f"Type: {self.spec.challenge_type.value}")
        preview_parts.append(f"Scheme: {self.spec.scheme.value}")
        preview_parts.append(f"Task: {self.spec.task}")
        if self.spec.constraints:
            preview_parts.append(f"Depth: {self.spec.constraints.depth}")
            preview_parts.append(f"Batch Size: {self.spec.constraints.batch_size}")

        if self._template_context:
            preview_parts.append("\nTemplate files available:")
            for name in self._template_context:
                preview_parts.append(f"  - {name}")

        self.data_preview = "\n".join(preview_parts)

    def _draft(self) -> MCTSNode:
        """Generate initial FHE solution draft."""
        logger.info("Starting FHE Draft.")

        introduction = (
            "You are an expert C++ programmer specializing in Fully Homomorphic Encryption (FHE). "
            "Implement a solution for the given challenge. "
            "You must generate ONLY the eval() function body in C++."
        )

        prompt: Any = {
            "Introduction": introduction,
        }
        prompt |= self._build_challenge_prompt()
        prompt |= self._build_template_prompt()
        prompt |= self._build_eval_instructions()

        prompt["Memory"] = self.virtual_root.fetch_child_memory()

        prompt["CRITICAL"] = [
            "NEVER decrypt inputs or use secret keys.",
            "NEVER return dummy/placeholder output - implement real FHE computation.",
            "Derive optimal polynomial approximation parameters analytically before coding.",
        ]

        prompt |= self._build_response_format()

        instructions = "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt, 2)

        # Build final prompt using the model-appropriate format
        if self.acfg.steerable_reasoning and "deepseek" in self.acfg.code.model:
            user_prompt = instructions
            prompt_complete = (
                f"<｜begin▁of▁sentence｜>\n{introduction}\n"
                f"<｜User｜>{user_prompt}<｜Assistant｜><think>\n"
                f"I need to implement an FHE eval() function. Let me analyze the challenge.\n"
                f"Challenge info: {self.data_preview}"
            )
        elif self.acfg.steerable_reasoning and "qwen3" in self.acfg.code.model:
            user_prompt = instructions
            prompt_complete = (
                f"<|im_start|>system\n{introduction}<|im_end|>\n"
                f"<|im_start|>user{user_prompt}<|im_end|>"
                f"<|im_start|>assistant\n<think>"
                f"I need to implement an FHE eval() function. Let me analyze the challenge.\n"
                f"Challenge info: {self.data_preview}"
            )
        else:
            # Chat format for GPT/Claude/other closed-source models
            user_prompt = f"""
{instructions}

# Challenge Preview
{self.data_preview}
"""
            prompt_complete = [
                {"role": "system", "content": introduction},
                {"role": "user", "content": user_prompt},
            ]

        self.virtual_root.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(
            plan=plan, code=code, parent=self.virtual_root,
            stage="draft", local_best_node=self.virtual_root,
        )
        logger.info(f"FHE Draft: node {new_node.id}")
        return new_node

    def _improve(self, parent_node: MCTSNode) -> MCTSNode:
        """Improve working FHE solution."""
        logger.info(f"Starting FHE Improve on Node {parent_node.id}.")

        introduction = (
            "You are improving a working FHE solution. "
            "Analyze the current results and make improvements to increase accuracy. "
            "Provide the complete eval() function body with your improvements."
        )

        prompt: Any = {
            "Introduction": introduction,
        }
        prompt |= self._build_challenge_prompt()
        prompt |= self._build_template_prompt()
        prompt |= self._build_eval_instructions()

        prompt["Memory"] = parent_node.fetch_child_memory()

        prompt["Current Solution"] = {
            "Code": wrap_code(parent_node.code, lang="cpp"),
            "Execution Output": wrap_code(parent_node.term_out, lang=""),
        }

        if parent_node.analysis:
            prompt["Current Results"] = parent_node.analysis
        if parent_node.metric and parent_node.metric.value is not None:
            prompt["Current Accuracy"] = f"{parent_node.metric.value:.4f}"

        prompt["Improvement Focus"] = [
            "Identify what limits the current accuracy.",
            "Consider better polynomial approximations or more iterations.",
            "Ensure depth budget is used efficiently.",
            "Try to reduce numerical errors in the computation.",
        ]

        prompt |= self._build_response_format()

        instructions = "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt, 2)

        if self.acfg.steerable_reasoning and "deepseek" in self.acfg.code.model:
            user_prompt = instructions
            prompt_complete = (
                f"<｜begin▁of▁sentence｜>{introduction}"
                f"<｜User｜>{user_prompt}<｜Assistant｜><think>\n"
                f"I need to improve this FHE solution. Current accuracy: "
                f"{parent_node.metric.value if parent_node.metric and parent_node.metric.value else 'unknown'}.\n"
                f"Previous code:\n{parent_node.code[:500]}"
            )
        elif self.acfg.steerable_reasoning and "qwen3" in self.acfg.code.model:
            user_prompt = instructions
            prompt_complete = (
                f"<|im_start|>system\n{introduction}<|im_end|>\n"
                f"<|im_start|>user{user_prompt}<|im_end|>"
                f"<|im_start|>assistant\n<think>"
                f"I need to improve this FHE solution. Current accuracy: "
                f"{parent_node.metric.value if parent_node.metric and parent_node.metric.value else 'unknown'}.\n"
                f"Previous code:\n{parent_node.code[:500]}"
            )
        else:
            user_prompt = instructions
            prompt_complete = [
                {"role": "system", "content": introduction},
                {"role": "user", "content": user_prompt},
            ]

        parent_node.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(
            plan=plan, code=code, parent=parent_node,
            stage="improve", local_best_node=parent_node.local_best_node,
        )
        logger.info(f"FHE Improve: {parent_node.id} -> {new_node.id}")
        return new_node

    def _debug(self, parent_node: MCTSNode) -> MCTSNode:
        """Fix buggy FHE solution."""
        logger.info(f"Starting FHE Debug on Node {parent_node.id}.")

        introduction = (
            "Debug this FHE solution. Carefully read the error output, "
            "understand what went wrong, and fix the code. "
            "Provide the complete fixed eval() function body."
        )

        prompt: Any = {
            "Introduction": introduction,
        }
        prompt |= self._build_challenge_prompt()
        prompt |= self._build_template_prompt()
        prompt |= self._build_eval_instructions()

        prompt["Buggy Solution"] = {
            "Code": wrap_code(parent_node.code, lang="cpp"),
        }
        prompt["Error Output"] = wrap_code(parent_node.term_out[-5000:] if parent_node.term_out else "", lang="")

        prompt["Debugging Instructions"] = [
            "Read the error message carefully and fix the specific issue.",
            "If accuracy is low, recompute parameters; if depth exceeded, reduce complexity.",
            "Common fixes: depth exceeded -> reduce polynomial degree; "
            "missing rotation key [k] -> add k to config.json indexes_for_rotation_key.",
            "CRITICAL: Do NOT fall back to dummy/placeholder output.",
        ]

        prompt |= self._build_response_format()

        instructions = "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt, 2)

        if self.acfg.steerable_reasoning and "deepseek" in self.acfg.code.model:
            user_prompt = instructions
            prompt_complete = (
                f"<｜begin▁of▁sentence｜>{introduction}"
                f"<｜User｜>{user_prompt}<｜Assistant｜><think>\n"
                f"I need to debug this FHE solution. The error was:\n"
                f"{parent_node.term_out[-500:] if parent_node.term_out else 'unknown'}\n"
                f"Root cause analysis: {parent_node.analysis or 'N/A'}"
            )
        elif self.acfg.steerable_reasoning and "qwen3" in self.acfg.code.model:
            user_prompt = instructions
            prompt_complete = (
                f"<|im_start|>system\n{introduction}<|im_end|>\n"
                f"<|im_start|>user{user_prompt}<|im_end|>"
                f"<|im_start|>assistant\n<think>"
                f"I need to debug this FHE solution. The error was:\n"
                f"{parent_node.term_out[-500:] if parent_node.term_out else 'unknown'}\n"
                f"Root cause analysis: {parent_node.analysis or 'N/A'}"
            )
        else:
            user_prompt = instructions
            prompt_complete = [
                {"role": "system", "content": introduction},
                {"role": "user", "content": user_prompt},
            ]

        parent_node.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(
            plan=plan, code=code, parent=parent_node,
            stage="debug", local_best_node=parent_node.local_best_node,
        )
        logger.info(f"FHE Debug: {parent_node.id} -> {new_node.id}")
        return new_node

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Override: extract C++ code instead of Python."""
        completion_text = None
        for _ in range(retries):
            if "gpt-5" in self.acfg.code.model or self.acfg.steerable_reasoning == False:
                completion_text = gpt_query(
                    prompt=prompt,
                    temperature=self.acfg.code.temp,
                    model=self.acfg.code.model,
                    cfg=self.cfg,
                )
            else:
                completion_text = r1_query(
                    prompt=prompt,
                    temperature=self.acfg.code.temp,
                    model=self.acfg.code.model,
                    cfg=self.cfg,
                )

            code = extract_cpp_code(completion_text)
            # Extract natural language text before code
            nl_text = ""
            if "```" in completion_text:
                nl_text = completion_text[:completion_text.find("```")].strip()

            if code and nl_text:
                return nl_text, code

            logger.info("FHE plan + code extraction failed, retrying...")
        logger.info("Final FHE plan + code extraction attempt failed.")
        return "", completion_text or ""

    # ==================== Execution Result Parsing ====================

    def parse_exec_result(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        """
        Parse FHE execution results.

        Uses the FHE-specific result data stored in exc_info by FHEInterpreter,
        plus the feedback LLM to extract accuracy metrics.
        """
        try:
            logger.info(f"FHE Agent parsing results for node {node.id}")

            node.absorb_exec_result(exec_result)

            # Check if we have FHE-specific result data from the interpreter
            fhe_data = None
            if exec_result.exc_info and isinstance(exec_result.exc_info, dict):
                fhe_data = exec_result.exc_info.get("fhe_result")

            if fhe_data:
                return self._parse_fhe_result_direct(node, fhe_data)

            # Fallback: use feedback LLM (like original MCTSAgent but with FHE-specific prompt)
            return self._parse_fhe_result_with_llm(node, exec_result)

        except Exception as e:
            logger.warning(f"FHE parse result error: {e}")
            node.is_buggy = True
            node.metric = WorstMetricValue()
            node.analysis = f"Parse error: {e}"
            return node

    def _parse_fhe_result_direct(self, node: MCTSNode, fhe_data: dict) -> MCTSNode:
        """Parse FHE result directly from interpreter data (no LLM needed)."""
        build_success = fhe_data.get("build_success", False)
        run_success = fhe_data.get("run_success", False)
        output_generated = fhe_data.get("output_generated", False)
        accuracy = fhe_data.get("accuracy")

        validation = fhe_data.get("validation", {})
        if validation:
            accuracy = validation.get("accuracy", accuracy)

        # Determine if buggy
        node.is_buggy = not (build_success and run_success and output_generated)

        # Build analysis
        analysis_parts = []
        if node.is_buggy:
            node.metric = WorstMetricValue()
            if not build_success:
                analysis_parts.append("Build failed - compilation errors")
            elif not run_success:
                analysis_parts.append("Runtime error during execution")
            elif not output_generated:
                analysis_parts.append("No output generated")
        else:
            if accuracy is not None:
                node.metric = MetricValue(accuracy, maximize=True)
                analysis_parts.append(f"Accuracy: {accuracy:.4f}")
                if validation:
                    if validation.get("mean_error") is not None:
                        analysis_parts.append(f"Mean error: {validation['mean_error']:.6f}")
                    if validation.get("max_error") is not None:
                        analysis_parts.append(f"Max error: {validation['max_error']:.6f}")
            else:
                node.metric = MetricValue(0.0, maximize=True)
                analysis_parts.append("Executed but no accuracy metric")

        node.analysis = "; ".join(analysis_parts) if analysis_parts else "No analysis"

        logger.info(f"FHE Node {node.id}: buggy={node.is_buggy}, metric={node.metric.value if node.metric else None}")
        return node

    def _parse_fhe_result_with_llm(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        """Fallback: use feedback LLM to parse FHE results."""
        introduction = (
            "You are an FHE expert evaluating the output of an FHE solution execution. "
            "Determine if the build succeeded, if there were runtime errors, "
            "and extract the accuracy metric if available."
        )
        prompt = {
            "Introduction": introduction,
            "Task": f"FHE Challenge: {self.spec.task}",
            "Implementation": wrap_code(node.code, lang="cpp"),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=fhe_review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
                cfg=self.cfg,
            ),
        )

        if not isinstance(response.get("metric"), (int, float)):
            response["metric"] = None

        node.analysis = response.get("summary", "")
        node.is_buggy = (
            response.get("is_bug", True)
            or node.exc_type is not None
            or response.get("metric") is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
            logger.info(f"FHE Node {node.id} is buggy")
        else:
            node.metric = MetricValue(
                response["metric"],
                maximize=not response.get("lower_is_better", False),
            )
            logger.info(f"FHE Node {node.id}: metric={node.metric.value}")

        return node

    # ==================== Override step() ====================

    def step(self, node: MCTSNode, exec_callback) -> MCTSNode:
        """Override step to skip CSV submission checks."""
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
            self.search_start_time = time.time()

        if not node or node.stage == "root":
            node = self.select(self.virtual_root)

        _root, result_node = self._step_search_fhe(node, exec_callback=exec_callback)

        if result_node and result_node.metric.value is not None:
            if self.best_node is None or self.best_node.metric < result_node.metric:
                logger.info(f"FHE Node {result_node.id} is the best node so far")
                self.best_node = result_node
                best_solution_dir = self.cfg.workspace_dir / "best_solution"
                with self.save_node_lock:
                    best_solution_dir.mkdir(exist_ok=True, parents=True)
                    with open(best_solution_dir / "solution.cpp", "w") as f:
                        f.write(result_node.code)
                    with open(best_solution_dir / "node_id.txt", "w") as f:
                        f.write(str(result_node.id))
            else:
                logger.info(f"FHE Node {result_node.id} not better than best ({self.best_node.id})")
        elif not result_node:
            logger.info("FHE: Result node is None.")
        else:
            logger.info("FHE: Result node has bug or no metric.")

        if self.best_node:
            logger.info(f"FHE Best metric: {self.best_node.metric.value}")

        self.current_step = len(self.journal)
        if _root or result_node is None:
            return self.virtual_root
        else:
            return result_node

    def _step_search_fhe(self, parent_node: MCTSNode, exec_callback):
        """FHE version of _step_search - skips CSV submission checks."""
        logger.info(f"[FHE _step_search] Processing node: {parent_node.id}")
        result_node = None
        _root = False

        if not parent_node.is_terminal:
            try:
                if self.is_root(parent_node):
                    result_node = self._draft()
                    result_node.lock = True
                elif parent_node.is_buggy:
                    result_node = self._debug(parent_node)
                elif parent_node.is_buggy is False:
                    result_node = self._improve(parent_node)
                else:
                    logger.warning(f"[FHE] node {parent_node.id} is_buggy is None.")

                if result_node:
                    exe_res = exec_callback(result_node.code, result_node.id, True)
                    result_node = self.parse_exec_result(
                        node=result_node,
                        exec_result=exe_res,
                    )

                    # Skip CSV check (not applicable to FHE)
                    logger.info(f"FHE Node {result_node.id} metric: {result_node.metric.value}")

                    if not self.check_metric_valid(node=result_node):
                        result_node.metric = WorstMetricValue()
                        logger.info(f"FHE node {result_node.id}: invalid metric.")

                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

                    if parent_node.is_buggy and result_node.is_buggy is False:
                        parent_node.is_debug_success = True

                    _root = self.check_improvement(result_node, parent_node)

                    with self.journal_lock:
                        if (self.best_node and result_node.metric.maximize
                                and self.best_node.metric.maximize != result_node.metric.maximize):
                            logger.warning("FHE: Metric direction inconsistency.")
                            raise ValueError("Metric direction inconsistency")
                        else:
                            self.journal.append(result_node)

            except Exception as e:
                logger.warning(f"FHE: Node generation failed, rolling back. Error: {e}")
                self.backpropagate(node=parent_node, value=0, add_to_tree=False)
                parent_node.sub_expected_child_count()
                raise e
        else:
            logger.info("FHE: Current node is terminal, backpropagating.")
            self.backpropagate(node=parent_node, value=0)
            _root = True

        return _root, result_node
