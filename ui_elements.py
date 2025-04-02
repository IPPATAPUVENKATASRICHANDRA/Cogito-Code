"""
This is a module of the Cogito Code application that provides various UI elements and utilities.
It includes a loading spinner, progress bar, text effects, and a banner display.

functions:
- ensure_config_dir: Ensures the configuration directory exists.
- Spinner: A class that provides a loading spinner animation.
- ProgressBar: A class that provides a simple progress bar animation.
- type_text_effect: Types out text with a typewriter effect.
- print_banner: Displays a stylish, centered banner for the application.
- print_section_header: Prints a stylish section header, centered in the terminal.
- input_with_border: Displays an input prompt with a border.


"""

import sys
import time
import threading
from pathlib import Path
import textwrap
import shutil  # For getting terminal size

try:
    from huggingface_hub import InferenceClient
except ImportError:
    print(
        "huggingface_hub not installed. Run 'pip install huggingface_hub' to use Hugging Face models."
    )

try:
    from pyfiglet import figlet_format
except ImportError:
    # Fallback if pyfiglet is not installed
    def figlet_format(text, font="slant"):
        return f"=== {text} ===\n"


try:
    from colorama import init, Fore, Style

    init(autoreset=True)
except ImportError:
    # Fallback definitions if colorama is not installed
    class Fore:
        CYAN = ""
        GREEN = ""
        RED = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""

    class Style:
        BRIGHT = ""
        RESET_ALL = ""


def ensure_config_dir():
    config_dir = Path.home() / ".Cogito Code"
    config_dir.mkdir(exist_ok=True)
    return config_dir


class Spinner:
    """A simple loading spinner animation"""

    def __init__(self, message="Loading", color=Fore.CYAN):
        self.message = message
        self.color = color
        self.spinning = False
        self.spinner_chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        self.spinner_thread = None

    def spin(self):
        i = 0
        while self.spinning:
            sys.stdout.write(
                f"\r{self.color}{self.message} {self.spinner_chars[i % len(self.spinner_chars)]}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        self.spinning = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self):
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()


class ProgressBar:
    """A simple progress bar animation"""

    def __init__(
        self,
        total=100,
        prefix="Progress",
        suffix="Complete",
        length=50,
        fill="█",
        color=Fore.GREEN,
    ):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.color = color
        self.progress = 0
        self.running = False
        self.thread = None

    def print_progress(self):
        while self.running and self.progress < self.total:
            percent = self.progress / self.total
            filled_length = int(self.length * percent)
            bar = self.fill * filled_length + "-" * (self.length - filled_length)
            sys.stdout.write(
                f"\r{self.color}{self.prefix} |{bar}| {int(percent * 100)}% {self.suffix}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            if self.progress < self.total:
                self.progress += 1

    def start(self):
        self.running = True
        self.progress = 0
        self.thread = threading.Thread(target=self.print_progress)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, completed=True):
        self.running = False
        if completed:
            self.progress = self.total
            percent = 1.0
            filled_length = int(self.length * percent)
            bar = self.fill * filled_length + "-" * (self.length - filled_length)
            sys.stdout.write(
                f"\r{self.color}{self.prefix} |{bar}| {int(percent * 100)}% {self.suffix}"
            )
            sys.stdout.flush()
        sys.stdout.write("\n")
        if self.thread:
            self.thread.join()


def type_text_effect(text, delay=0.01, color=Fore.GREEN):
    """Types out text with a typewriter effect"""
    for char in text:
        sys.stdout.write(f"{color}{char}")
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")


def print_banner():
    """Display a stylish, centered banner for the application."""
    # Use a bigger or different font if you prefer (e.g., "standard", "big", "doom", etc.)
    banner_text = figlet_format("Cogito Code", font="big")

    # Determine the terminal width
    terminal_width = shutil.get_terminal_size((80, 20)).columns

    # Center each line of the banner text
    centered_lines = []
    for line in banner_text.split("\n"):
        centered_lines.append(line.center(terminal_width))

    # Print the centered banner with a typewriter effect
    type_text_effect(Fore.CYAN + Style.BRIGHT + "\n".join(centered_lines), delay=0.001)

    # Print a horizontal line and subtitle, also centered
    separator = "=" * terminal_width
    print(Fore.GREEN + Style.BRIGHT + separator)
    subtitle = "Welcome to Cogito Code - Your AI Code Assistant"
    # Center the subtitle as well
    print(Fore.YELLOW + Style.BRIGHT + subtitle.center(terminal_width))
    print(Fore.GREEN + Style.BRIGHT + separator + Style.RESET_ALL)


def print_section_header(text, color=Fore.YELLOW):
    """Print a stylish section header, centered in the terminal."""
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    line = "=" * terminal_width
    print("\n" + color + line)
    centered_title = f" {text} ".center(terminal_width, "=")
    print(color + centered_title)
    print(color + line + Style.RESET_ALL)


def input_with_border(prompt_text, border_char="*", padding=2, color=Fore.CYAN):
    """
    Display an input prompt with a border.
    """
    prompt_lines = prompt_text.splitlines()
    max_length = max(len(line) for line in prompt_lines)
    border = border_char * (max_length + padding * 2 + 2)
    print(color + border)
    for line in prompt_lines:
        print(
            color
            + border_char
            + " " * padding
            + line.ljust(max_length)
            + " " * padding
            + border_char
        )
    print(color + border + Style.RESET_ALL)
    return input(color + "> " + Style.RESET_ALL)
