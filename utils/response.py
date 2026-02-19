import json
import re

import black


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))

def extract_review(text):
    """Extract json code blocks from the text."""
    parsed_codes = []

    # When code is in a text or json block
    matches = re.findall(r"```(json)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(json)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    review = json.loads(parsed_codes[0])
    return review

def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def extract_cpp_code(text):
    """Extract C++ code blocks from LLM response text.

    Handles ```cpp, ```c++, and ``` blocks with C++ content.
    Also handles ### CONFIG ### and ### CODE ### sections for white-box FHE challenges.
    """
    # For white-box: check for CONFIG and CODE sections
    if "### CONFIG ###" in text or "### CODE ###" in text:
        return _extract_whitebox_code(text)

    parsed_codes = []

    # Match ```cpp, ```c++, or ``` blocks
    matches = re.findall(r"```(?:cpp|c\+\+)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        if match.strip():
            parsed_codes.append(match.strip())

    if parsed_codes:
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


def _extract_whitebox_code(text):
    """Extract code and optional CONFIG section for white-box FHE challenges."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
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
