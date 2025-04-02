"""
This is the main module of the Cogito Code application.
This module provides the main functionality for the application, including user interaction, model initialization, and problem-solving capabilities.
It includes functions to load user settings, manage LLM providers, and handle user input for programming problems.

The functions are imported from other modules, including ui_elements, helper_func, and wrapper.

"""

import lmstudio as lms
import os
import time
import pickle
import traceback
import json
import re
import sys
import threading

from colorama import Fore, Style
from pathlib import Path
from huggingface_hub import InferenceClient

from ui_elements import (
    Spinner,
    print_section_header,
    print_banner,
    type_text_effect,
    ensure_config_dir,
    ProgressBar,
    input_with_border,
)
from helper_func import (
    extract_text,
    code_extractor,
    extract_test_results,
    has_failed_tests,
)
from prompts import understand_problem, test_case_gen_checker
from wrapper import (
    HuggingFaceModel,
    save_user_settings,
    load_user_settings,
    manage_huggingface_settings,
    display_settings_menu,
)


def main():
    # Display the stylish banner with improved formatting
    print_banner()

    # Load user settings
    settings = load_user_settings() or {}

    # Ask for LLM provider preference if not already set
    if "provider" not in settings:
        print_section_header("LLM Provider Selection", Fore.CYAN)
        print(Fore.GREEN + "Please select your preferred LLM provider:")
        print(Fore.YELLOW + "1. Local LLM (via LM Studio)")
        print(Fore.YELLOW + "2. Hugging Face")

        while True:
            choice = input_with_border("Enter your choice (1 or 2):")
            if choice in ["1", "2"]:
                settings["provider"] = "local" if choice == "1" else "huggingface"
                if settings["provider"] == "local":
                    print_section_header("Local Model Selection", Fore.CYAN)
                    print(
                        Fore.GREEN
                        + "Please enter the name of the local model you want to use."
                    )
                    print(
                        Fore.YELLOW
                        + "This should be a model you have available in LM Studio."
                    )
                    settings["local_model_name"] = input_with_border(
                        "Enter model name:"
                    )
                save_user_settings(settings)
                break
            print(Fore.RED + "Invalid choice. Please enter 1 or 2.")

    # Initialize model
    model = None

    while True:
        if model is None:
            if settings["provider"] == "local":
                spinner = Spinner(message="Initializing Local LLM", color=Fore.CYAN)
                spinner.start()
                try:
                    client = lms.Client()
                    model_name = settings.get("local_model_name")
                    if not model_name:
                        spinner.stop()
                        print_section_header("Local Model Selection", Fore.CYAN)
                        print(
                            Fore.GREEN
                            + "Please enter the name of the local model you want to use."
                        )
                        model_name = input_with_border("Enter model name:")
                        settings["local_model_name"] = model_name
                        save_user_settings(settings)
                        spinner = Spinner(
                            message="Initializing Local LLM", color=Fore.CYAN
                        )
                        spinner.start()

                    model = client.llm.model(model_name)
                    time.sleep(1)  # Simulate loading time
                    spinner.stop()
                    print(
                        Fore.GREEN
                        + f"✓ Local model '{model_name}' loaded successfully!"
                    )
                except Exception as e:
                    spinner.stop()
                    print(Fore.RED + f"Error initializing local model: {str(e)}")
                    print(
                        Fore.YELLOW
                        + "Please check your LM Studio setup and make sure the model name is correct."
                    )
                    retry = input_with_border(
                        "Would you like to try a different model name? (y/n):"
                    )
                    if retry.lower() == "y":
                        settings["local_model_name"] = input_with_border(
                            "Enter the new local model name:"
                        )
                        save_user_settings(settings)
                        continue
                    else:
                        model = None
                        break

            else:  # Hugging Face
                if "hf_api_key" not in settings or "hf_model_name" not in settings:
                    print_section_header("Hugging Face Credentials", Fore.CYAN)
                    settings["hf_api_key"] = input_with_border(
                        "Enter your Hugging Face API key:"
                    )
                    settings["hf_model_name"] = input_with_border(
                        "Enter the Hugging Face model name:"
                    )
                    save_user_settings(settings)

                spinner = Spinner(
                    message="Initializing Hugging Face model", color=Fore.CYAN
                )
                spinner.start()
                try:
                    model = HuggingFaceModel(
                        settings["hf_api_key"], settings["hf_model_name"]
                    )
                    time.sleep(1)  # Simulate loading time
                    spinner.stop()
                    print(
                        Fore.GREEN
                        + f"✓ Hugging Face model '{settings['hf_model_name']}' initialized successfully!"
                    )
                except Exception as e:
                    spinner.stop()
                    print(Fore.RED + f"Error initializing Hugging Face model: {str(e)}")
                    retry = input_with_border(
                        "Would you like to switch to a local LLM? (y/n):"
                    )
                    if retry.lower() == "y":
                        print(Fore.YELLOW + "Switching to local LLM...")
                        print_section_header("Local Model Selection", Fore.CYAN)
                        print(
                            Fore.GREEN
                            + "Please enter the name of the local model you want to use."
                        )
                        model_name = input_with_border("Enter model name:")
                        try:
                            spinner = Spinner(
                                message="Initializing Local LLM", color=Fore.CYAN
                            )
                            spinner.start()
                            client = lms.Client()
                            model = client.llm.model(model_name)
                            settings["provider"] = "local"
                            settings["local_model_name"] = model_name
                            save_user_settings(settings)
                            time.sleep(1)
                            spinner.stop()
                            print(
                                Fore.GREEN
                                + f"✓ Local model '{model_name}' loaded successfully as fallback!"
                            )
                        except Exception as e2:
                            spinner.stop()
                            print(
                                Fore.RED
                                + f"Error initializing fallback model: {str(e2)}"
                            )
                            model = None
                            break
                    else:
                        model = None
                        break

        # Main interaction loop for problem solving
        while model is not None:
            print("\n" + Fore.CYAN + "=" * 80)
            print(Fore.GREEN + "Commands: ")
            print(Fore.YELLOW + "1. Solve a programming problem")
            print(Fore.YELLOW + "2. Manage settings")
            print(Fore.YELLOW + "3. Exit")

            cmd = input_with_border("Enter your choice (1-3):")
            if cmd == "1":
                print_section_header("Programming Problem Solver", Fore.GREEN)
                user_prompt = input_with_border(
                    "Enter your programming problem or coding challenge:"
                )
                if not user_prompt.strip():
                    print(Fore.RED + "Problem description cannot be empty.")
                    continue

                print_section_header("Problem Analysis", Fore.BLUE)
                explanation = understand_problem(model, user_prompt)
                print(explanation)

                proceed = input_with_border(
                    "Would you like to generate code for this problem? (y/n):"
                )
                if proceed.lower() != "y":
                    continue

                print_section_header("Generating Code Solution", Fore.MAGENTA)
                code_gen_prompt = f"""
You are an expert programmer tasked with solving the following programming problem:

{user_prompt}

Please provide a complete, efficient, and well-documented solution in Python.
Your code should:
1. Be fully functional and ready to run
2. Include detailed comments explaining the logic
3. Handle edge cases and potential errors
4. Follow best practices for Python coding style (PEP 8)
5. Be optimized for performance where applicable

Return ONLY the code solution, enclosed in triple backticks.
"""
                spinner = Spinner(message="Generating solution", color=Fore.GREEN)
                spinner.start()
                code_result = model.respond(code_gen_prompt)
                spinner.stop()
                code = code_extractor(code_result)
                print(Fore.GREEN + "Generated Solution:\n")
                print(code)

                test_code = input_with_border(
                    "Would you like to test this code? (y/n):"
                )
                if test_code.lower() == "y":
                    final_code, test_results = test_case_gen_checker(
                        model, code, user_prompt
                    )
                    if final_code != code:
                        print_section_header("Final Optimized Code", Fore.GREEN)
                        print(final_code)
                    save_code = input_with_border(
                        "Would you like to save this code to a file? (y/n):"
                    )
                    if save_code.lower() == "y":
                        filename = input_with_border(
                            "Enter filename to save (default: solution.py):"
                        )
                        if not filename:
                            filename = "solution.py"
                        if not filename.endswith(".py"):
                            filename += ".py"
                        try:
                            with open(filename, "w") as f:
                                f.write(final_code)
                            print(
                                Fore.GREEN + f"✓ Code successfully saved to {filename}"
                            )
                        except Exception as e:
                            print(Fore.RED + f"Error saving file: {str(e)}")
            elif cmd == "2":
                if display_settings_menu(settings):
                    print(Fore.YELLOW + "Settings updated. Reinitializing model...")
                    model = None
                    break  # Break inner loop to reinitialize model
            elif cmd == "3":
                print(Fore.GREEN + "Thank you for using Cogito Code! Goodbye.")
                return
            else:
                print(Fore.RED + "Invalid command. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nExiting Cogito Code. Goodbye!")
    except Exception as e:
        print(Fore.RED + f"\nAn unexpected error occurred: {str(e)}")
        traceback.print_exc()
