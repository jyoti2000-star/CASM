#!/usr/bin/env python3
import sys
import os
import time
import shutil
import io
import contextlib
from pathlib import Path
from src.compiler import compiler

try:
    from src.utils.asm_transpiler import transpile_file as _transpile_asm_file
    HAS_ASM_TRANSPILER = True
except Exception:
    HAS_ASM_TRANSPILER = False

from src.utils.colors import Colors
from src.utils.term import print_stage, print_error as term_error, print_info as term_info

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

def main():
    if len(sys.argv) < 2:
        print("Usage: casm.py <file.asm> [--type asm|exe|obj]")
        sys.exit(1)

    input_file = sys.argv[1]
    options = sys.argv[2:]
    requested_type = None
    if '--type' in options:
        try:
            requested_type = options[options.index('--type') + 1]
        except Exception:
            requested_type = None

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    output_type = requested_type or 'asm'

    if output_type == 'asm':
        out = os.path.join('output', Path(input_file).stem + '.asm')
        os.makedirs('output', exist_ok=True)
        ok = compiler.compile_to_assembly(input_file, out, quiet=False)
        if not ok:
            sys.exit(1)
        print(f"Assembly written to: {out}")
    else:
        ok = compiler.compile_to_executable(input_file, output_executable=None, run_after=False, quiet=False)
        if not ok:
            sys.exit(1)

if __name__ == '__main__':
    main()
