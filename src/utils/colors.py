#!/usr/bin/env python3
class Colors:
    """ANSI color codes for terminal output"""
    # By default use colored output
    MINIMAL = False

    # ANSI color codes
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
    # Control whether non-final success messages are shown. Leave False to
    # suppress intermediate successes and show only the final success message
    SHOW_SUCCESS = False

def print_error(message):
    """Print error message (always shown)"""
    if getattr(Colors, 'MINIMAL', False):
        print(f"[*] {message}")
    else:
        print(f"{Colors.RED}[*]{Colors.RESET} {message}")

def print_system(message):
    """Print system message (visible by default unless MINIMAL and replaced by debug)"""
    # In minimal mode, system messages are quiet unless explicit
    if getattr(Colors, 'MINIMAL', False):
        # Treat as debug-level in minimal mode
        if getattr(Colors, 'VERBOSE', False):
            print(f"[-] {message}")
    else:
        print(f"{Colors.YELLOW}[-]{Colors.RESET} {message}")

def print_success(message):
    """Print success message (shown)"""
    # Only print success if configured to show success messages, or if this
    # is a fatal/terminal success (caller can still use print_error on failure).
    if not getattr(Colors, 'SHOW_SUCCESS', False):
        return

    if getattr(Colors, 'MINIMAL', False):
        print(f"[+] {message}")
    else:
        print(f"{Colors.GREEN}[+]{Colors.RESET} {message}")

def print_info(message):
    """Print informational messages.

    Info messages are suppressed by default. Set Colors.VERBOSE = True
    to enable them.
    """
    if getattr(Colors, 'VERBOSE', False):
        if getattr(Colors, 'MINIMAL', False):
            print(f"[-] {message}")
        else:
            print(f"{Colors.BLUE}[-]{Colors.RESET} {message}")

def print_warning(message):
    """Print warning message (always shown)"""
    if getattr(Colors, 'MINIMAL', False):
        print(f"[!] {message}")
    else:
        print(f"{Colors.ORANGE}[!]{Colors.RESET} {message}")

def print_debug(message):
    """Print debug messages when verbose.

    Debug messages are suppressed by default. Set Colors.VERBOSE = True
    to enable them.
    """
    if getattr(Colors, 'VERBOSE', False):
        if getattr(Colors, 'MINIMAL', False):
            print(f"[#] {message}")
        else:
            print(f"{Colors.PURPLE}[#]{Colors.RESET} {message}")

def print_special(message):
    """Print special message"""
    if getattr(Colors, 'MINIMAL', False):
        print(f"[~] {message}")
    else:
        print(f"{Colors.CYAN}[~]{Colors.RESET} {message}")

def print_final_success(message):
    """Always print the final success message regardless of SHOW_SUCCESS."""
    if getattr(Colors, 'MINIMAL', False):
        print(f"[+] {message}")
    else:
        print(f"{Colors.GREEN}[+]{Colors.RESET} {message}")

def colorize(text: str, color: str) -> str:
    """Colorize text with given color"""
    return f"{color}{text}{Colors.RESET}"

def bold(text: str) -> str:
    """Make text bold"""
    return f"{Colors.BOLD}{text}{Colors.RESET}"

def underline(text: str) -> str:
    """Underline text"""
    return f"{Colors.UNDERLINE}{text}{Colors.RESET}"