#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Prefer local project modules for compiler and asm transpiler
try:
    from utils.transpiler.transpiler import transpile_file as _transpile_asm_file
    HAS_ASM_TRANSPILER = True
except Exception:
    HAS_ASM_TRANSPILER = False

try:
    # CASM compiler pipeline
    from src.core.pipeline import compile_file as _casm_compile_file
    HAS_CASM_COMPILER = True
except Exception:
    HAS_CASM_COMPILER = False

import argparse
import subprocess

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='casm.py', description='CASM/ASM helper CLI')
    parser.add_argument('input', help='Input file (.casm source or .asm assembly)')
    parser.add_argument('-o', '--output', help='Output file (for asm output). If omitted uses ./output/<name>.asm')
    parser.add_argument('--type', choices=['asm', 'exe', 'obj'], default='asm', help='What to produce from the input')
    parser.add_argument('--target', default='linux', help='Target platform for asm transpiler (linux|windows|macos|bsd)')
    parser.add_argument('--target-os', choices=['linux', 'windows', 'macos'], default='linux', help='Target OS for CASM compiler (affects calling convention)')
    parser.add_argument('--arch', default='x86_64', help='Target architecture for asm transpiler')
    parser.add_argument('--extract-c', action='store_true', help='If assembly contains embedded C passthrough comments, extract them to a companion .c file')
    parser.add_argument('--compile-embedded-c', action='store_true', help='If embedded C is extracted, attempt to compile it to an object file using system C compiler')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return 2

    output_type = args.type
    out_path = Path(args.output) if args.output else Path('output') / (input_path.stem + '.asm')

    # If input is a raw .asm file - use the assembly transpiler if available
    if input_path.suffix.lower() == '.asm':
        if not HAS_ASM_TRANSPILER:
            print("ASM transpiler not available in this environment. Please ensure 'utils.transpiler.transpiler' is importable.")
            return 3
        try:
            # transpile_file returns the generated assembly as a string (and writes to output if output arg provided)
            asm = _transpile_asm_file(str(input_path), target=args.target, arch=args.arch, opt_level='basic', output=None)
        except Exception as e:
            print(f"Error transpiling .asm file: {e}")
            return 4
    else:
        # Assume it's a CASM source file to be compiled by the project's compiler
        if not HAS_CASM_COMPILER:
            print("CASM compiler pipeline not available. Ensure 'src.core.pipeline' is importable.")
            return 5
        try:
            asm = _casm_compile_file(str(input_path), verbose=args.verbose, target_os=args.target_os, extract_c=args.extract_c, compile_embedded_c=args.compile_embedded_c)
            if asm is None:
                print("Compilation failed — no assembly produced.")
                return 6
        except Exception as e:
            print(f"Error compiling CASM source: {e}")
            return 7

    # At this point we have assembly text in `asm`
    if output_type == 'asm':
        try:
            write_output(out_path, asm)
            print(f"Assembly written to: {out_path}")
            return 0
        except Exception as e:
            print(f"Failed to write assembly: {e}")
            return 8

    # Producing executable/object is not implemented in this script; write asm and instruct the user
    try:
        # write the assembly to a temporary location so user can assemble/link
        write_output(out_path, asm)
        print(f"Assembly written to: {out_path}")
        print("Note: building 'exe' or 'obj' is not implemented by casm.py. Use your assembler/linker (nasm/gcc/clang) to create binaries from the produced .asm file.")
        return 0
    except Exception as e:
        print(f"Failed to write assembly while preparing binary: {e}")
        return 9


if __name__ == '__main__':
    sys.exit(main())
