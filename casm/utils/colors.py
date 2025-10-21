#!/usr/bin/env python3
class Colors:
    """ANSI color codes for terminal output"""
    MINIMAL = False
    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    SHOW_SUCCESS = False

def print_error(message):
    if getattr(Colors, 'MINIMAL', False):
        print(f"[*] {message}")
    else:
        print(f"{Colors.RED}[*]{Colors.RESET} {message}")

def print_system(message):
    if getattr(Colors, 'MINIMAL', False):
        if getattr(Colors, 'VERBOSE', False):
            print(f"[-] {message}")
    else:
        print(f"{Colors.YELLOW}[-]{Colors.RESET} {message}")

def print_success(message):
    if not getattr(Colors, 'SHOW_SUCCESS', False):
        return
    if getattr(Colors, 'MINIMAL', False):
        print(f"[+] {message}")
    else:
        print(f"{Colors.GREEN}[+]{Colors.RESET} {message}")

def print_info(message):
    if getattr(Colors, 'VERBOSE', False):
        if getattr(Colors, 'MINIMAL', False):
            print(f"[-] {message}")
        else:
            print(f"{Colors.BLUE}[-]{Colors.RESET} {message}")

def print_warning(message):
    if getattr(Colors, 'MINIMAL', False):
        print(f"[!] {message}")
    else:
        print(f"{Colors.ORANGE}[!]{Colors.RESET} {message}")

def print_debug(message):
    if getattr(Colors, 'VERBOSE', False):
        if getattr(Colors, 'MINIMAL', False):
            print(f"[#] {message}")
        else:
            print(f"{Colors.PURPLE}[#]{Colors.RESET} {message}")

def print_special(message):
    if getattr(Colors, 'MINIMAL', False):
        print(f"[~] {message}")
    else:
        print(f"{Colors.CYAN}[~]{Colors.RESET} {message}")

def print_final_success(message):
    if getattr(Colors, 'MINIMAL', False):
        print(f"[+] {message}")
    else:
        print(f"{Colors.GREEN}[+]{Colors.RESET} {message}")

def colorize(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"

def bold(text: str) -> str:
    return f"{Colors.BOLD}{text}{Colors.RESET}"

def underline(text: str) -> str:
    return f"{Colors.UNDERLINE}{text}{Colors.RESET}"
