#!/usr/bin/env python3

"""
CASM - Clean Assembly Language Compiler
Main entry point with clean command-line interface
"""

import sys
import os

# Prevent Python bytecode generation
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.compiler import processor, compiler
from src.utils.colors import print_info, print_success, print_error, print_warning, print_system, Colors, bold

def print_help():
    """Print usage information"""
    print(f"{Colors.CYAN}{bold('CASM - Clean Assembly Language Compiler')}{Colors.RESET}")
    print("=" * 50)
    print("A clean, focused assembly language with high-level constructs")
    print("")
    print(f"{bold('Usage:')} python3 casm.py <command> <file> [options]")
    print("")
    print(f"{Colors.ORANGE}{bold('Commands:')}{Colors.RESET}")
    print("  compile <file.casm>  - Compile to executable only") 
    print("  asm <file.casm>      - Generate assembly code only")
    print("  help                 - Show this help")
    print("")
    print(f"{Colors.GREEN}{bold('Language Features:')}{Colors.RESET}")
    print("  %var name value      - Declare variable")
    print("  %if condition        - If statement")
    print("  %while condition     - While loop")
    print("  %for var in range(n) - For loop")
    print("  %println message     - Print line")
    print("  %scanf format var    - Read input")
    print(f"  {Colors.YELLOW}%!{Colors.RESET}                   - Embed C code block")
    print("")
    print(f"{Colors.BLUE}{bold('Examples:')}{Colors.RESET}")
    print("  python3 casm.py compile program.casm")
    print("  python3 casm.py asm test.casm")
    print("")
    print(f"{Colors.PURPLE}{bold('C Code Integration:')}{Colors.RESET}")
    print("  %!")
    print("  int add(int a, int b) {")
    print("      return a + b;")
    print("  }")
    print("")
    print(f"{bold('Files:')}")
    print("  Input: .casm files")
    print("  Output: .exe executables, .asm assembly files")

def validate_file(file_path: str) -> bool:
    """Validate input file"""
    if not os.path.exists(file_path):
        print_error(f"File not found: {file_path}")
        return False
    
    if not file_path.endswith('.casm'):
        print_warning(f"Expected .casm file, got: {file_path}")
        print_info("Continuing anyway...")
    
    return True

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command in ['help', '-h', '--help']:
        print_help()
        return
    
    if len(sys.argv) < 3:
        print_error("File argument required")
        print_help()
        sys.exit(1)
    
    input_file = sys.argv[2]
    
    if not validate_file(input_file):
        sys.exit(1)
    
    # Execute command
    success = False
    
    if command == 'compile':
        print_system(f"Compiling {input_file}...")
        success = compiler.compile_to_executable(input_file, run_after=False)
    
    elif command == 'asm':
        print_system(f"Generating assembly for {input_file}...")
        output_file = os.path.join("output", Path(input_file).stem + ".asm")
        success = compiler.compile_to_assembly(input_file, output_file)
    
    else:
        print_error(f"Unknown command: {command}")
        print_help()
        sys.exit(1)
    
    if not success:
        sys.exit(1)
    
    print_success("Operation completed successfully!")

if __name__ == "__main__":
    main()