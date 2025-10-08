#!/usr/bin/env python3
import sys
import os

# Prevent Python bytecode generation
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.compiler import processor, compiler
from src.utils.colors import print_info, print_success, print_final_success, print_error, print_warning, print_system, Colors, bold

def print_help():
    """Print minimal usage information"""
    print("python3 casm.py <command> <file> [options]")
    print("")
    print("Commands:")
    print("  compile <file.casm>  - Compile to executable")
    print("  asm <file.casm>      - Generate assembly")
    print("  help                 - Show this help")
    print("")
    print("")
def validate_file(file_path: str) -> bool:
    """Validate input file"""
    if not os.path.exists(file_path):
        print_error(f"File not found: {file_path}")
        return False
    
    if not file_path.endswith('.asm'):
        print_warning(f"Expected .asm file, got: {file_path}")
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
        try:
            success = compiler.compile_to_executable(input_file, run_after=False)
        except Exception as e:
            print_error(str(e))
            sys.exit(1)
    
    elif command == 'asm':
        # quiet: don't print a duplicate system message here (compiler will handle higher-level messaging)
        output_file = os.path.join("output", Path(input_file).stem + ".asm")
        try:
            success = compiler.compile_to_assembly(input_file, output_file)
        except Exception as e:
            # Single consolidated error message with details
            print_error(str(e))
            sys.exit(1)
    
    else:
        print_error(f"Unknown command: {command}")
        print_help()
        sys.exit(1)
    
    if not success:
        sys.exit(1)

    # Single final success message that includes the produced path depending on the mode
    if command == 'asm':
        print_final_success(f"Assembly generated: {output_file}")
    elif command == 'compile':
        # Match compiler default: output executable is placed in 'output/<stem>.exe'
        exe_name = Path(input_file).stem + ".exe"
        exe_path = os.path.join("output", exe_name)
        print_final_success(f"Executable generated: {exe_path}")

if __name__ == "__main__":
    main()