"""

Helper functions for Cogito Code to manage user settings, extract text and code from model responses,
parse test results, and check for failed tests.

functions:
- ensure_config_dir: Ensures the configuration directory exists.
- extract_text: Extracts plain text from the model's response.

- code_extractor: Extracts code blocks from the model's response.

- extract_test_results: Parses test results from the model's output.

- has_failed_tests: Checks if any tests have failed based on the test results.


"""

import re
import json
from colorama import Fore
from pathlib import Path

from ui_elements import Spinner


def ensure_config_dir():
    """Ensure the configuration directory exists."""
    config_dir = Path.home() / ".Cogito Code"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def extract_text(result):
    """Extract plain text explanation from the result."""
    if hasattr(result, "text"):
        result = result.text

    if not isinstance(result, str):
        result = str(result)

    return result.strip()


def code_extractor(result):
    """
    Extracts code blocks from the result.
    If the result is not a string, it tries to use its .text attribute.
    """
    if hasattr(result, "text"):
        result = result.text

    if not isinstance(result, str):
        result = str(result)

    # Regex to extract code inside triple backticks, optionally with a language specifier.
    pattern = re.compile(r"```(?:[a-zA-Z]+)?\n(.*?)\n```", re.DOTALL)
    match = pattern.search(result)
    if match:
        code = match.group(1)
        return code.strip()
    else:
        return result.strip()


def extract_test_results(test_output):
    """
    Extract test results from the model's output and convert to dictionary format.
    Handles PredictionResult objects and converts them to strings first.
    """
    # Ensure we're working with a string
    if hasattr(test_output, "text"):
        test_output = test_output.text

    if not isinstance(test_output, str):
        test_output = str(test_output)

    spinner = Spinner(message="Parsing test results", color=Fore.BLUE)
    spinner.start()

    try:
        # Try to find and parse a JSON or dictionary-like structure in the output
        dict_pattern = re.compile(r"\{[\s\S]*\}")
        dict_match = dict_pattern.search(test_output)

        if dict_match:
            # Found a dictionary-like structure
            try:
                # Try to parse it as JSON
                result = json.loads(dict_match.group(0))
                spinner.stop()
                return result
            except json.JSONDecodeError:
                pass

        # If we can't find a dictionary directly, try to parse the markdown table
        # Look for markdown tables with |---|---|
        rows = re.findall(r"^\|(.*)\|$", test_output, re.MULTILINE)
        if rows and len(rows) >= 3:
            # Extract headers
            headers = [h.strip() for h in rows[0].split("|")]

            # Skip the separator row (row[1]) and process data rows
            test_cases = []
            for i in range(2, len(rows)):
                values = [v.strip() for v in rows[i].split("|")]
                test_case = {
                    headers[j]: values[j]
                    for j in range(len(headers))
                    if j < len(values)
                }
                test_cases.append(test_case)

            spinner.stop()
            return {"test_cases": test_cases}
    except Exception as e:
        spinner.stop()
        print(Fore.RED + f"Error parsing test results: {str(e)}")

    # If all else fails, return the raw text
    spinner.stop()
    return {"raw_output": test_output}


def has_failed_tests(test_results):
    """
    Check if any tests have failed based on the test results.
    """
    # Convert to string for simple checks if it's not already a properly parsed result
    test_results_str = str(test_results).lower()

    # Simple check for failure keywords in the entire result
    if "fail" in test_results_str or "error" in test_results_str:
        # If the parsed result is a dictionary, do more detailed checks
        if isinstance(test_results, dict):
            if "test_cases" in test_results:
                for test in test_results["test_cases"]:
                    status_key = next(
                        (
                            k
                            for k in test
                            if "status" in k.lower() or "pass" in k.lower()
                        ),
                        None,
                    )
                    if status_key and (
                        "fail" in test[status_key].lower()
                        or "error" in test[status_key].lower()
                    ):
                        return True
            elif "raw_output" in test_results:
                # Already checked for failure keywords in the string conversion above
                return True
        else:
            # If not a dictionary, fall back to the string check we already did
            return True

    # If no failure keywords found in the entire output, assume all tests passed
    return False
