# Smoke test: generate assembly for example ufunc_nou.asm
from src.compiler import compiler
import os
from pathlib import Path

SRC = 'examples/ufunc_nou.asm'

# Test x86_64 assembly generation
ok64 = compiler.compile_to_assembly(SRC, output_file='output/ufunc_nou.x86_64.asm', quiet=True, target='linux', arch='x86_64')
print('x86_64 asm generation:', ok64)

# Test x86 (32-bit) assembly generation
ok32 = compiler.compile_to_assembly(SRC, output_file='output/ufunc_nou.x86.asm', quiet=True, target='linux', arch='x86')
print('x86 (32-bit) asm generation:', ok32)

# Ensure files were created
print('Files created:')
for f in ('output/ufunc_nou.x86_64.asm','output/ufunc_nou.x86.asm'):
    print(f, os.path.exists(f))
