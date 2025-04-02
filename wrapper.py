"""
This is a module of the CogitoCode application.
This module provides a wrapper for the Hugging Face model and manages user settings.
It includes functions to save and load user settings, manage Hugging Face API keys and model names, and handle local model settings.

functions:
- save_user_settings: Saves user settings to a configuration file.
- load_user_settings: Loads user settings from a configuration file.
- manage_huggingface_settings: Manages Hugging Face API key and model name settings.
- manage_local_model_settings: Manages local model settings.
- display_settings_menu: Displays the settings menu and handles user input.

- get_huggingface_model: Returns a Hugging Face model wrapper object.
- get_local_model: Returns a local model wrapper object.
- get_model: Returns the appropriate model wrapper based on user settings.
- get_model_response: Returns the model response based on user input and settings.
- get_test_results: Returns test results based on user input and settings.
- get_code_extraction: Returns code extraction based on user input and settings.
- get_text_extraction: Returns text extraction based on user input and settings.
- get_code_extraction: Returns code extraction based on user input and settings.
- get_test_results: Returns test results based on user input and settings.


"""

import os
import time
import pickle
import traceback
from colorama import Fore, Style
from pathlib import Path
from huggingface_hub import InferenceClient

from ui_elements import Spinner, print_section_header, print_banner, type_text_effect
from ui_elements import ensure_config_dir, figlet_format
from helper_func import (
    extract_text,
    code_extractor,
    extract_test_results,
    has_failed_tests,
)
import json


# Hugging Face Model Wrapper
class HuggingFaceModel:
    def __init__(self, api_key, model_name):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name

    def respond(self, prompt):
        response = self.client.text_generation(
            prompt,
            model=self.model_name,
            max_new_tokens=1024,
            temperature=0.7,
            return_full_text=False,
        )
        return response


# User settings management
def save_user_settings(settings):
    config_dir = ensure_config_dir()
    settings_file = config_dir / "settings.pkl"

    with open(settings_file, "wb") as f:
        pickle.dump(settings, f)


def load_user_settings():
    config_dir = ensure_config_dir()
    settings_file = config_dir / "settings.pkl"

    if settings_file.exists():
        try:
            with open(settings_file, "rb") as f:
                return pickle.load(f)
        except:
            return {}
    return {}


# Function to manage Hugging Face settings
def manage_huggingface_settings(settings):
    """Manage Hugging Face API key and model name"""
    print_section_header("Hugging Face Settings Management", Fore.CYAN)

    # Show current settings if they exist
    if "hf_api_key" in settings and "hf_model_name" in settings:
        print(Fore.GREEN + f"Current model: {settings['hf_model_name']}")
        masked_key = (
            settings["hf_api_key"][:4]
            + "*" * (len(settings["hf_api_key"]) - 8)
            + settings["hf_api_key"][-4:]
        )
        print(Fore.GREEN + f"Current API key: {masked_key}")

    print(Fore.YELLOW + "What would you like to update?")
    print(Fore.YELLOW + "1. API Key")
    print(Fore.YELLOW + "2. Model Name")
    print(Fore.YELLOW + "3. Both")
    print(Fore.YELLOW + "4. Back to main menu")

    choice = input(Fore.CYAN + "Enter your choice (1-4): " + Style.RESET_ALL)

    if choice == "1" or choice == "3":
        settings["hf_api_key"] = input(
            Fore.GREEN + "Enter your new Hugging Face API key: " + Style.RESET_ALL
        )

    if choice == "2" or choice == "3":
        settings["hf_model_name"] = input(
            Fore.GREEN + "Enter the new Hugging Face model name: " + Style.RESET_ALL
        )

    if choice in ["1", "2", "3"]:
        save_user_settings(settings)
        print(Fore.GREEN + "Settings updated successfully!")
        return True  # Indicate settings were changed

    return False  # No changes made


# Function to manage local model settings
def manage_local_model_settings(settings):
    """Manage local model settings"""
    print_section_header("Local Model Settings", Fore.CYAN)

    if "local_model_name" in settings:
        print(Fore.GREEN + f"Current local model: {settings['local_model_name']}")

    print(Fore.YELLOW + "1. Change local model")
    print(Fore.YELLOW + "2. Back to main menu")

    choice = input(Fore.CYAN + "Enter your choice (1-2): " + Style.RESET_ALL)

    if choice == "1":
        print(Fore.GREEN + "Available models depend on what you have in LM Studio.")
        new_model = input(
            Fore.GREEN + "Enter the new local model name: " + Style.RESET_ALL
        )
        settings["local_model_name"] = new_model
        save_user_settings(settings)
        print(Fore.GREEN + f"Local model updated successfully to '{new_model}'!")
        return True  # Indicate settings were changed

    return False  # No changes made


# Function to display settings menu
def display_settings_menu(settings):
    """Display and handle the settings menu"""
    print_section_header("Settings Menu", Fore.CYAN)

    print(Fore.YELLOW + "1. Change LLM provider (Local/Hugging Face)")

    if settings.get("provider") == "huggingface":
        print(Fore.YELLOW + "2. Manage Hugging Face settings")
    else:
        print(Fore.YELLOW + "2. Manage local model settings")

    print(Fore.YELLOW + "3. Back to main menu")

    choice = input(Fore.CYAN + "Enter your choice (1-3): " + Style.RESET_ALL)

    settings_changed = False

    if choice == "1":
        current = (
            "Hugging Face" if settings.get("provider") == "huggingface" else "Local LLM"
        )
        print(Fore.YELLOW + f"Currently using: {current}")
        print(Fore.GREEN + "Switch to:")
        print(Fore.YELLOW + "1. Local LLM (via LM Studio)")
        print(Fore.YELLOW + "2. Hugging Face")

        provider_choice = input(
            Fore.CYAN + "Enter your choice (1 or 2): " + Style.RESET_ALL
        )
        if provider_choice == "1" and settings.get("provider") != "local":
            settings["provider"] = "local"
            if "local_model_name" not in settings:
                # Ask for model name instead of using default
                print_section_header("Local Model Selection", Fore.CYAN)
                settings["local_model_name"] = input(
                    Fore.GREEN
                    + "Enter the name of the local model to use: "
                    + Style.RESET_ALL
                )
            save_user_settings(settings)
            settings_changed = True
        elif provider_choice == "2" and settings.get("provider") != "huggingface":
            settings["provider"] = "huggingface"
            if "hf_api_key" not in settings or "hf_model_name" not in settings:
                print_section_header("Hugging Face Credentials", Fore.CYAN)
                settings["hf_api_key"] = input(
                    Fore.GREEN + "Enter your Hugging Face API key: " + Style.RESET_ALL
                )
                settings["hf_model_name"] = input(
                    Fore.GREEN + "Enter the Hugging Face model name: " + Style.RESET_ALL
                )
            save_user_settings(settings)
            settings_changed = True

    elif choice == "2":
        if settings.get("provider") == "huggingface":
            settings_changed = manage_huggingface_settings(settings)
        else:
            settings_changed = manage_local_model_settings(settings)

    return settings_changed
