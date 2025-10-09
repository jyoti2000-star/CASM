#!/usr/bin/env python3
import os
import sys
import tempfile
import subprocess
import shutil
import re
from typing import Optional
from pathlib import Path

from .core.lexer import CASMLexer
from .core.parser import CASMParser
from .core.codegen import AssemblyCodeGenerator
from .stdlib.minimal import stdlib
from .utils.c_processor import c_processor
from .utils.formatter import formatter
from .utils.colors import print_info, print_success, print_warning, print_error, print_system
from .utils.syntax import check_syntax, format_errors

class CASMProcessor:
    """Core CASM language processor"""
    
    def __init__(self):
        self.lexer = CASMLexer()
        self.parser = CASMParser()
        self.codegen = AssemblyCodeGenerator()
    
    def process_file(self, input_file: str) -> str:
        """Process CASM file and return assembly code"""
        try:
            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Reduce noisy system output; keep as debug-level
            from .utils.colors import print_debug
            print_debug(f"Processing {input_file}...")
            
            # Legacy explicit embedded C indicator removed; lexer now detects C lines
            # Run syntax checks (lexer + parser) before proceeding to codegen
            with open(input_file, 'r', encoding='utf-8') as _f:
                raw_content = _f.read()

            syntax_errors = check_syntax(raw_content, filename=input_file)
            if syntax_errors:
                # Print errors and abort processing
                err_text = format_errors(syntax_errors, filename=input_file)
                print_error(err_text)
                raise Exception('Syntax errors detected; aborting compilation')

            # Tokenize
            tokens = self.lexer.tokenize(content)
            non_eof_tokens = [t for t in tokens if t.type.value != 'EOF']
            print_info(f"Generated {len(non_eof_tokens)} tokens")
            
            # Parse
            ast = self.parser.parse(tokens)
            print_info(f"Parsed AST with {len(ast.statements)} statements")
            
            # Generate assembly
            assembly_code = self.codegen.generate(ast)
            line_count = len(assembly_code.split(chr(10)))
            print_success(f"Generated {line_count} lines of optimized assembly")
            
            return assembly_code
            
        except Exception as e:
            # Allow the exception to propagate so the top-level CLI prints a single
            # consolidated error message. Do not print here to avoid duplicates.
            raise
        
        finally:
            # Cleanup C processor
            c_processor.cleanup()

    # Note: implicit C detection is now handled in the lexer (C_INLINE tokens)

class CASMCompiler:
    """Complete CASM compiler with build pipeline"""
    
    def __init__(self):
        self.processor = CASMProcessor()
        self.temp_dir = None
    
    def compile_to_assembly(self, input_file: str, output_file: str = None) -> bool:
        """Compile CASM to assembly file"""
        try:
            # The higher-level CLI now controls visible messages; avoid duplicate system prints
            from .utils.colors import print_debug
            print_debug(f"Assembly generation for {input_file}...")

            # Generate assembly
            assembly_code = self.processor.process_file(input_file)

            # Determine output file
            if output_file is None:
                output_file = str(Path(input_file).with_suffix('.asm'))

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

            # Add debug info and format
            final_assembly = formatter.add_debug_info(assembly_code, input_file)

            # Write assembly file (no automatic asm prelude; prettifier will
            # emit grouped externs when appropriate)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_assembly)

            print_success(f"Assembly generated: {output_file}")
            return True

        except Exception as e:
            print(f"[ERROR] Assembly generation failed: {e}")
            return False
    
    def compile_to_executable(self, input_file: str, output_executable: str = None, run_after: bool = False) -> bool:
        """Compile CASM to executable"""
        try:
            # Check dependencies first
            if not self._check_build_dependencies():
                return False
            
            # Set default output name
            if output_executable is None:
                output_executable = Path(input_file).stem
            
            # Ensure output directory
            os.makedirs("output", exist_ok=True)
            if not os.path.dirname(output_executable):
                output_executable = os.path.join("output", output_executable)
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix='casm_')
            
            print(f"[1/4] Processing {input_file}...")
            # Process CASM file
            assembly_code = self.processor.process_file(input_file)

            # Automatically generate a small driver if assembly references external
            # CASM variables (V#) or Winsock imports (__imp_*) so linking succeeds.
            v_symbols, imp_symbols = self._collect_external_symbols(assembly_code)
            # Detect labels/data already defined in assembly so we don't redefine them
            defined_vs = set()
            for m in re.finditer(r'\n\s*([A-Za-z_][A-Za-z0-9_]*)\s+(db|dq|dd|resb|times)\b', '\n' + assembly_code):
                defined_vs.add(m.group(1))

            # Detect if assembly already provides a main symbol
            has_main = bool(re.search(r'(?:^|\n)\s*(?:global\s+main|main:)\b', assembly_code, re.I | re.M))

            # Only request driver definitions for V# symbols not already defined
            filtered_v_symbols = [v for v in v_symbols if v not in defined_vs]
            driver_obj = None
            extra_libs = []
            if filtered_v_symbols or any(s.startswith('__imp_') for s in imp_symbols):
                # Create a driver C file in the temp dir and compile it
                driver_obj = self._create_and_compile_driver(self.temp_dir, filtered_v_symbols, imp_symbols, create_main=(not has_main))
                # Infer libs from saved combined C includes (if present)
                inferred = self._infer_libs_from_includes(self.temp_dir)
                extra_libs.extend(inferred)
                # Fallback: if we still have imp symbols that look like winsock, add ws2_32
                if not any('ws2_32' == l for l in extra_libs) and any('wsacleanup' in s.lower() or 'wsastartup' in s.lower() or 'socket' in s.lower() for s in imp_symbols):
                    extra_libs.append('ws2_32')
            
            # Write assembly to temp file
            asm_file = os.path.join(self.temp_dir, 'output.asm')
            # Prepend extern declarations for V# and __imp_* symbols so NASM treats them as external
            asm_prelude_lines = []
            for v in v_symbols:
                asm_prelude_lines.append(f'extern {v}')
            for imp in imp_symbols:
                asm_prelude_lines.append(f'extern {imp}')

            # Also detect any direct function calls in the assembly (e.g. "call printf")
            # and add them as externs unless they are defined in the assembly itself.
            call_targets = set(re.findall(r"\bcall\s+([A-Za-z_][A-Za-z0-9_]*)\b", assembly_code))

            # Collect symbols already defined in assembly so we don't extern them
            defined_syms = set()
            # NASM/GAS global directives
            for m in re.finditer(r'(?m)^\s*(?:global|\.globl)\s+([A-Za-z_][A-Za-z0-9_]*)', assembly_code):
                defined_syms.add(m.group(1))
            # Label definitions (e.g. "casm_main:")
            for m in re.finditer(r'(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', assembly_code):
                defined_syms.add(m.group(1))
            # .def entries (from GCC output)
            for m in re.finditer(r'\.def\s+([A-Za-z_][A-Za-z0-9_]*)', assembly_code):
                defined_syms.add(m.group(1))

            # Exclude a few known internal entry points
            internal_excludes = {'casm_main', 'main'}

            for name in sorted(call_targets):
                if name in defined_syms or name in internal_excludes:
                    continue
                # Avoid duplicating entries already added
                entry = f'extern {name}'
                if entry not in asm_prelude_lines:
                    asm_prelude_lines.append(entry)

            asm_prelude = ''
            if asm_prelude_lines:
                asm_prelude = '\n'.join(f"{l}" for l in asm_prelude_lines) + '\n\n'

            with open(asm_file, 'w', encoding='utf-8') as f:
                f.write(asm_prelude + assembly_code)
            
            print("[2/4] Assembling with NASM...")
            # Assemble with NASM
            obj_file = os.path.join(self.temp_dir, 'output.o')
            if not self._run_nasm(asm_file, obj_file):
                return False
            
            print("[3/4] Linking...")
            # Link executable (include driver object and any extra libs)
            exe_file = f"{output_executable}.exe"
            extra_objs = [driver_obj] if driver_obj else None
            if not self._run_linker(obj_file, exe_file, extra_objs=extra_objs, extra_libs=extra_libs):
                return False
            
            print(f"[SUCCESS] Executable created: {exe_file}")
            
            if run_after:
                print("[4/4] Running executable...")
                return self._run_executable(exe_file)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Compilation failed: {e}")
            return False
        
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _check_build_dependencies(self) -> bool:
        """Check if required build tools are available"""
        required = ['nasm', 'x86_64-w64-mingw32-gcc']
        missing = []
        
        for tool in required:
            if not shutil.which(tool):
                missing.append(tool)
        
        if missing:
            print(f"[ERROR] Missing build tools: {', '.join(missing)}")
            print("[INFO] Install with: brew install nasm mingw-w64")
            return False
        
        return True

    def _collect_external_symbols(self, assembly_code: str):
        """Find referenced external symbols (V#, __imp_*) in assembly."""
        refs = set()
        # look for [rel SYMBOL] and plain SYMBOL references
        for m in re.finditer(r'\[rel\s+(\w+)\]', assembly_code):
            refs.add(m.group(1))
        for m in re.finditer(r'__imp_\w+', assembly_code):
            refs.add(m.group(0))

        v_symbols = sorted([s for s in refs if re.match(r'^V\d+$', s)])
        imp_symbols = sorted([s for s in refs if s.startswith('__imp_')])
        return v_symbols, imp_symbols

    def _infer_needed_libs(self, assembly_code: str, imp_symbols) -> list:
        """Infer which system libraries are needed from assembly/function names.

        This is heuristic-based: maps common networking/math/thread functions to
        their usual libraries. Users can still provide additional libs via
        project configuration in the future.
        """
        libs = set()

        # Winsock / sockets
        if re.search(r'\b(send|recv|socket|closesocket|WSAStartup|WSACleanup|inet_addr|inet_pton|getaddrinfo|htons|connect)\b', assembly_code, re.I):
            libs.add('ws2_32')

        # Math functions
        if re.search(r'\b(sin|cos|tan|sqrt|pow|log|exp)\b', assembly_code, re.I):
            libs.add('m')

        # Pthreads / threading
        if re.search(r'\b(pthread_create|pthread_join|pthread_mutex)\b', assembly_code, re.I):
            libs.add('pthread')

        # If __imp_ symbols exist, try to map obvious ones
        for sym in imp_symbols:
            low = sym.lower()
            if 'wsacleanup' in low or 'wsastartup' in low or 'socket' in low or 'send' in low or 'recv' in low:
                libs.add('ws2_32')

        return sorted(list(libs))

    def _infer_libs_from_includes(self, temp_dir: str) -> list:
        """Look for a combined C file saved by the C processor and map included
        headers to likely libraries. Returns a list of lib names (without -l).
        """
        includes_file = os.path.join(os.getcwd(), 'output', 'combined_c_from_cprocessor.c')
        libs = set()

        if not os.path.exists(includes_file):
            return []

        try:
            with open(includes_file, 'r', encoding='utf-8') as f:
                content = f.read()

            include_matches = re.findall(r'#include\s+[<"]([^>\"]+)[>\"]', content)
            for inc in include_matches:
                name = inc.lower()
                # Common mappings
                if 'winsock2.h' in name or 'ws2tcpip.h' in name:
                    libs.add('ws2_32')
                elif 'math.h' in name:
                    libs.add('m')
                elif 'pthread.h' in name:
                    libs.add('pthread')
                elif 'wininet' in name or 'windows.h' in name:
                    libs.add('user32')
                # Add additional header-to-lib heuristics here as needed

        except Exception:
            return []

        return sorted(list(libs))

    def _create_and_compile_driver(self, temp_dir: str, v_symbols, imp_symbols, create_main: bool = True) -> Optional[str]:
        """Create a small driver C file that defines V# variables and initializes Winsock if needed.

        Returns path to compiled object file or None on failure.
        """
        try:
            driver_c = os.path.join(temp_dir, 'casm_driver.c')
            driver_o = os.path.join(temp_dir, 'casm_driver.o')

            # Build driver content
            lines = [
                '#include <stdio.h>',
                '#include <stdlib.h>',
                '#include <stdint.h>',
            ]

            # If winsock imports present, include headers
            if any('__imp_' in s for s in imp_symbols):
                lines.append('#include <winsock2.h>')
                lines.append('#include <ws2tcpip.h>')

            lines.append('')

            # Define V# symbols with conservative types
            for v in v_symbols:
                # Make V2 buffer if used as a pointer name heuristic - but default to int
                if v == 'V2':
                    lines.append('char V2[1024];')
                else:
                    lines.append(f'int {v} = 0;')

            lines.append('')
            lines.append('extern void casm_main(void);')
            lines.append('')
            if create_main:
                lines.append('int main(int argc, char** argv) {')

                if any('__imp_' in s for s in imp_symbols):
                    lines.append('    WSADATA wsaData;')
                    lines.append('    if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {')
                    # Use escaped backslash-n so the C string contains \n rather than a real newline
                    lines.append('        fprintf(stderr, "WSAStartup failed\\n");')
                    lines.append('        return 1;')
                    lines.append('    }')

                lines.append('    casm_main();')

                if any('__imp_' in s for s in imp_symbols):
                    lines.append('    WSACleanup();')

                lines.append('    return 0;')
                lines.append('}')

            with open(driver_c, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            # Compile driver to object using cross-compiler
            cmd = ['x86_64-w64-mingw32-gcc', '-c', '-O0', '-masm=intel', driver_c, '-o', driver_o]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[ERROR] Failed to compile automatic driver: {result.stderr}")
                return None

            return driver_o
        except Exception as e:
            print(f"[ERROR] Exception creating driver: {e}")
            return None
    
    def _run_nasm(self, asm_file: str, obj_file: str) -> bool:
        """Run NASM assembler"""
        cmd = ['nasm', '-f', 'win64', asm_file, '-o', obj_file]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] NASM failed:")
            print(result.stderr)
            return False
        
        return True
    
    def _run_linker(self, obj_file: str, exe_file: str, extra_objs: Optional[list] = None, extra_libs: Optional[list] = None) -> bool:
        """Run linker, optionally adding extra object files and libraries."""
        cmd = ['x86_64-w64-mingw32-gcc']
        cmd.append(obj_file)
        if extra_objs:
            for eo in extra_objs:
                if eo:
                    cmd.append(eo)

        cmd.extend(['-o', exe_file, '-mconsole', '-lkernel32', '-lmsvcrt'])
        if extra_libs:
            for lib in extra_libs:
                cmd.append(f'-l{lib}')

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] Linking failed:")
            print(result.stderr)
            return False

        return True
    
    def _run_executable(self, exe_file: str) -> bool:
        """Run the compiled executable"""
        wine_cmd = shutil.which('wine') or shutil.which('wine64')
        
        if wine_cmd:
            print("-" * 40)
            try:
                result = subprocess.run([wine_cmd, exe_file], capture_output=False)
                print("-" * 40)
                print(f"[INFO] Program exited with code: {result.returncode}")
                return result.returncode == 0
            except KeyboardInterrupt:
                print("\n[INFO] Program interrupted by user")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to run executable: {e}")
                return False
        else:
            print("[WARNING] Wine not found, cannot run Windows executable")
            print(f"[INFO] Executable saved as: {exe_file}")
            print("[INFO] Install Wine to run: brew install --cask wine-stable")
            return True

# Global instances
processor = CASMProcessor()
compiler = CASMCompiler()