"""
This module contains functions to interact with the LLM model for various tasks.
It includes functions to analyze and fix code, generate test cases, and understand problems.


Functions:
- analyze_and_fix_code: Analyzes failed tests, explains issues, and fixes the code.
- test_case_gen_checker: Generates comprehensive test cases, executes the code, evaluates results, and fixes code if needed.
- understand_problem: Uses the LLM to understand the problem, explain it in plain English, and suggest an algorithm or approach to solve it.



"""

import time
import traceback
import pickle
from colorama import Fore, Style
from pathlib import Path
from huggingface_hub import InferenceClient
from ui_elements import (
    Spinner,
    ProgressBar,
    print_section_header,
    print_banner,
    type_text_effect,
)
from ui_elements import ensure_config_dir, figlet_format
from helper_func import (
    extract_text,
    code_extractor,
    extract_test_results,
    has_failed_tests,
)
import json


def analyze_and_fix_code(model, code, test_results, user_prompt):
    """
    Analyze failed tests, explain issues, and fix the code.
    """
    # Ensure test_results is serializable for the prompt
    if not isinstance(test_results, dict):
        test_results = {"raw_output": str(test_results)}

    print_section_header("Analyzing Failed Tests and Updating Code", Fore.BLUE)
    type_text_effect(
        Fore.YELLOW + "Thinking about what went wrong and how to fix it...", delay=0.03
    )

    # Create an animated thinking spinner
    spinner = Spinner(message="Analyzing code issues", color=Fore.MAGENTA)
    spinner.start()

    fix_prompt = f"""
You are a code optimization expert. You need to analyze the following code and its test results, identify issues, 
and provide an improved version of the code that will pass all tests.

ORIGINAL CODE:
```python
{code}
```

TEST RESULTS:
{str(test_results)}

ORIGINAL PROBLEM STATEMENT:
{user_prompt}

Please:
1. Analyze the test failures and identify the specific issues in the code
2. Explain your thought process and what needs to be fixed
3. Provide an updated and corrected version of the code

Your response should include:
- Analysis of failures (what went wrong)
- Your thought process on how to fix the issues
- Complete updated code that will pass all tests (enclosed in triple backticks)
"""

    fix_result = model.respond(fix_prompt)

    # Ensure we get a string from the result
    if hasattr(fix_result, "text"):
        fix_result_text = fix_result.text
    else:
        fix_result_text = str(fix_result)

    spinner.stop()

    print(fix_result_text)

    # Extract the updated code
    progress = ProgressBar(
        prefix="Extracting updated code", suffix="Complete", length=40, color=Fore.CYAN
    )
    progress.start()
    updated_code = code_extractor(fix_result_text)
    time.sleep(1)  # Simulate processing time
    progress.stop()

    return updated_code


def test_case_gen_checker(model, code, prompt, max_iterations=3):
    """
    This function takes the generated code and the user's problem prompt, then:
      1. Generates comprehensive test cases
      2. Executes the code against these test cases
      3. Evaluates the results
      4. If tests fail, analyzes and fixes the code
      5. Repeats until all tests pass or max iterations reached
      6. Returns the results in dictionary format
    """

    test_prompt = """
You are a specialized AI assistant designed to generate comprehensive test cases and evaluate the code provided by the user. Your tasks are as follows:

1. Analyze the provided code and generate a set of test cases covering the following aspects:
   - Edge cases
   - Corner cases
   - Boundary conditions
   - Invalid inputs
   - Valid inputs
   - Case sensitivity (if applicable)
   - Performance testing
   - Runtime errors
   - Memory leaks
   - Security vulnerabilities
   - Concurrency issues

2. Execute the code against these test cases.

3. Evaluate the results, determining whether each test case passes or fails.

4. Present the outcomes in a JSON dictionary format with the following structure:
   ```
   {
     "test_cases": [
       {
         "description": "Test case description",
         "input": "Input value(s)",
         "expected": "Expected outcome",
         "actual": "Actual outcome",
         "status": "PASS or FAIL"
       },
       ...
     ]
   }
   ```

Make sure to wrap the JSON output in triple backticks to make it easy to extract.
"""

    current_code = code
    iteration = 0
    all_tests_pass = False
    final_test_results = {"test_cases": [], "status": "incomplete"}

    while not all_tests_pass and iteration < max_iterations:
        iteration += 1
        print_section_header(
            f"Test Case Evaluation (Iteration {iteration}/{max_iterations})",
            Fore.YELLOW,
        )

        # Build the complete prompt
        full_prompt = (
            test_prompt
            + "\n"
            + prompt
            + "\n"
            + current_code
            + "\n"
            + "# Generate and evaluate test cases for this code\n"
        )

        try:
            # Show an animation while generating test cases
            spinner = Spinner(
                message=f"Generating and evaluating test cases", color=Fore.CYAN
            )
            spinner.start()

            # Generate and evaluate test cases
            test_result = model.respond(full_prompt)

            # Ensure we get a string from the result
            if hasattr(test_result, "text"):
                test_result_text = test_result.text
            else:
                test_result_text = str(test_result)

            spinner.stop()

            print(test_result_text)

            # Extract and parse test results
            progress = ProgressBar(
                prefix="Processing test results",
                suffix="Complete",
                length=40,
                color=Fore.BLUE,
            )
            progress.start()
            test_results = extract_test_results(test_result_text)
            time.sleep(1)  # Simulate processing time
            progress.stop()

            # Store the latest test results
            final_test_results = test_results

            # Check if any tests failed
            if not has_failed_tests(test_results):
                print(Fore.GREEN + "\n✓ All tests passed successfully!")
                all_tests_pass = True
                final_test_results["status"] = "complete_success"
            else:
                print(
                    Fore.RED
                    + "\n✗ Some tests failed. Starting code correction process..."
                )
                # Analyze failures and update code
                current_code = analyze_and_fix_code(
                    model, current_code, test_results, prompt
                )
                print_section_header("Updated Code", Fore.GREEN)
                print(current_code)
        except Exception as e:
            print(Fore.RED + f"An error occurred: {str(e)}")
            traceback.print_exc()
            final_test_results["status"] = "error"
            final_test_results["error_message"] = str(e)
            break

    if not all_tests_pass and iteration >= max_iterations:
        print(
            Fore.YELLOW
            + f"\nReached maximum iterations ({max_iterations}). Some tests may still fail."
        )
        final_test_results["status"] = "max_iterations_reached"

    # Ensure final_test_results is serializable
    if not isinstance(final_test_results, dict):
        final_test_results = {"raw_output": str(final_test_results)}

    return current_code, final_test_results


def understand_problem(model, prompt):
    """
    Uses the LLM to understand the problem, explain it in plain English,
    and suggest an algorithm or approach to solve it.
    """
    understanding_prompt = f"""
You are an expert programmer and teacher. Given the following programming problem, please:

1. Explain what the problem is asking in simple terms
2. Break down the key requirements and constraints
3. Suggest a clear algorithm or approach to solve it
4. Explain the time and space complexity of your suggested approach
5. Mention any potential edge cases or challenges to consider

Problem: {prompt}

Your response should be educational and easy to understand, explaining everything in plain English before any code is written.
"""

    # Show a spinner while generating the explanation
    spinner = Spinner(message="Understanding the problem", color=Fore.CYAN)
    spinner.start()
    understanding_result = model.respond(understanding_prompt)
    time.sleep(1)  # Slight delay for effect
    spinner.stop()

    # Extract the explanation
    explanation = extract_text(understanding_result)

    return explanation
