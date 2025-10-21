#!/usr/bin/env python3
"""
build_asm.py - assemble and link a .asm output into an executable

Usage: build_asm.py INPUT.asm [-o OUTPUT] [--target TARGET] [--arch ARCH] [--dry-run]

This script detects the host platform and chooses reasonable toolchains:
- Linux x86_64: nasm (elf64) -> ld or cc
- macOS x86_64: nasm (macho64) -> ld or cc
- Windows x86_64 (mingw): nasm (win64) -> x86_64-w64-mingw32-gcc or clang
- ARM64: prefer clang/cc on macOS, gcc/cc on Linux

The script prefers to use system `cc`/`clang` to link because it handles CRT/driver flags.
"""
import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path


def which(cmd):
    return shutil.which(cmd)


def detect_host():
    system = platform.system().lower()
    machine = platform.machine().lower()
    return system, machine


def choose_toolchain(system, machine, target=None, arch=None):
    # Normalize requested arch
    arch = arch or machine
    target = target or system
    tool = {
        'nasm': which('nasm'),
        'cc': which('cc') or which('clang') or which('gcc'),
        'ld': which('ld'),
        'mingw-gcc': which('x86_64-w64-mingw32-gcc') or which('i686-w64-mingw32-gcc'),
    }
    return tool, target, arch


def assemble_and_link(input_asm: Path, output_exe: Path, target: str, arch: str, dry_run: bool = False):
    tool, target, arch = choose_toolchain(*detect_host(), target=target, arch=arch)
    nasm = tool.get('nasm')
    cc = tool.get('cc')
    ld = tool.get('ld')
    mingw = tool.get('mingw-gcc')

    if not input_asm.exists():
        raise FileNotFoundError(f"Input asm file not found: {input_asm}")

    tmp_obj = input_asm.with_suffix('.o')

    cmds = []
    # Pick NASM format
    if target in ('linux',):
        format = 'elf64' if arch.startswith('x86_64') else 'elf64'
        if nasm:
            cmds.append([nasm, '-f', format, str(input_asm), '-o', str(tmp_obj)])
        else:
            raise RuntimeError('nasm not found; please install nasm')
        # Link using cc if available
        if cc:
            cmds.append([cc, str(tmp_obj), '-o', str(output_exe)])
        elif ld:
            cmds.append([ld, str(tmp_obj), '-o', str(output_exe)])
        else:
            raise RuntimeError('no linker (cc or ld) found')

    elif target in ('darwin', 'macos'):
        # macOS
        format = 'macho64'
        if nasm:
            cmds.append([nasm, '-f', format, str(input_asm), '-o', str(tmp_obj)])
        else:
            raise RuntimeError('nasm not found; please install nasm')
        if cc:
            cmds.append([cc, str(tmp_obj), '-o', str(output_exe)])
        else:
            raise RuntimeError('no cc/clang found')

    elif target in ('windows', 'mingw'):
        # Use mingw if available
        format = 'win64' if arch.startswith('x86_64') else 'win32'
        if nasm:
            cmds.append([nasm, '-f', format, str(input_asm), '-o', str(tmp_obj)])
        else:
            raise RuntimeError('nasm not found; please install nasm')
        if mingw:
            cmds.append([mingw, str(tmp_obj), '-o', str(output_exe)])
        elif cc:
            cmds.append([cc, str(tmp_obj), '-o', str(output_exe)])
        else:
            raise RuntimeError('no mingw/gcc/cc found to link on Windows')

    else:
        # generic fallback
        if nasm:
            cmds.append([nasm, '-f', 'elf64', str(input_asm), '-o', str(tmp_obj)])
        if cc:
            cmds.append([cc, str(tmp_obj), '-o', str(output_exe)])

    # run commands
    for c in cmds:
        print('RUN:', ' '.join(c))
        if dry_run:
            continue
        res = subprocess.run(c)
        if res.returncode != 0:
            raise RuntimeError(f'Command failed: {c}')

    print('Built:', output_exe)
    return output_exe


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('input', help='.asm file to build')
    p.add_argument('-o', '--output', help='output executable name', default=None)
    p.add_argument('--target', help='target os (linux, macos, windows)', default=None)
    p.add_argument('--arch', help='target arch (x86_64, arm64)', default=None)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    input_asm = Path(args.input)
    out = Path(args.output) if args.output else input_asm.with_suffix('')
    target = args.target
    arch = args.arch

    try:
        assemble_and_link(input_asm, out, target, arch, dry_run=args.dry_run)
    except Exception as e:
        print('ERROR:', e)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
