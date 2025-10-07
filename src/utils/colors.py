#!/usr/bin/env python3

"""
Color utilities for CASM
Provides colored output for better user experience
"""

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'      # Error messages
    ORANGE = '\033[38;5;208m'  # System commands (proper orange)
    GREEN = '\033[92m'    # Success messages
    BLUE = '\033[94m'     # Info messages
    YELLOW = '\033[93m'   # Warning messages
    PURPLE = '\033[95m'   # Debug messages
    CYAN = '\033[96m'     # Special info
    WHITE = '\033[97m'    # Emphasis
    RESET = '\033[0m'     # Reset to default
    
    # Styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_error(message):
    """Print error message with red [x]"""
    print(f"{Colors.RED}[x]{Colors.RESET} {message}")

def print_system(message):
    """Print system command message with orange [*]"""
    print(f"{Colors.ORANGE}[*]{Colors.RESET} {message}")

def print_success(message):
    """Print success message with green [+]"""
    print(f"{Colors.GREEN}[+]{Colors.RESET} {message}")

def print_info(message):
    """Print info message with blue [-]"""
    print(f"{Colors.BLUE}[-]{Colors.RESET} {message}")

def print_warning(message):
    """Print warning message with yellow [!]"""
    print(f"{Colors.YELLOW}[!]{Colors.RESET} {message}")

def print_debug(message):
    """Print debug message with purple [#]"""
    print(f"{Colors.PURPLE}[#]{Colors.RESET} {message}")

def print_special(message):
    """Print special message with cyan [~]"""
    print(f"{Colors.CYAN}[~]{Colors.RESET} {message}")

def colorize(text: str, color: str) -> str:
    """Colorize text with given color"""
    return f"{color}{text}{Colors.RESET}"

def bold(text: str) -> str:
    """Make text bold"""
    return f"{Colors.BOLD}{text}{Colors.RESET}"

def underline(text: str) -> str:
    """Underline text"""
    return f"{Colors.UNDERLINE}{text}{Colors.RESET}"