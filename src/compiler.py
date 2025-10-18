#!/usr/bin/env python3
import os
import sys
import tempfile
import subprocess
import shutil
import re
import platform
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
from .utils.term import print_stage, print_error as term_error, print_info as term_info
from .utils.asm_transpiler import transpile_text

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
            # Extract top-level bracketed assembler directives (e.g. [BITS 16])
            try:
                directives = []
                for line in raw_content.splitlines():
                    s = line.strip()
                    if s.startswith('[') and s.endswith(']'):
                        directives.append(s)
                # Attach to codegen so it can emit these at the top of the file
                setattr(self.codegen, 'top_directives', directives)
            except Exception:
                setattr(self.codegen, 'top_directives', [])

            # Provide target hints to the code generator so it can emit
            # platform-appropriate section headers and conventions when
            # generating unified assembly. Default to host detection.
            try:
                host = platform.system().lower()
                if 'darwin' in host:
                    tgt = 'macos'
                elif 'windows' in host:
                    tgt = 'windows'
                else:
                    tgt = 'linux'
                setattr(self.codegen, 'target_platform', tgt)
                arch = platform.machine().lower()
                if 'aarch64' in arch or 'arm' in arch:
                    setattr(self.codegen, 'target_arch', 'arm64')
                elif 'riscv' in arch:
                    setattr(self.codegen, 'target_arch', 'riscv64')
                else:
                    setattr(self.codegen, 'target_arch', 'x86_64')
            except Exception:
                setattr(self.codegen, 'target_platform', None)
                setattr(self.codegen, 'target_arch', None)

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
        self._quiet_mode = False
        self._last_error = None
        self._last_info = None
        # detect host once
        self.host = self._detect_host()
        # detect arch once
        self.arch = self._detect_arch()

    def _detect_arch(self) -> str:
        """Return normalized architecture id: 'x86_64', 'arm64', or 'riscv64'."""
        m = platform.machine().lower()
        if 'aarch64' in m or 'arm' in m:
            return 'arm64'
        if 'riscv' in m:
            return 'riscv64'
        if 'x86_64' in m or 'amd64' in m:
            return 'x86_64'
        # default
        return 'x86_64'

    def _host_to_target(self) -> str:
        """Map detected host to asm_transpiler target names."""
        if self.host == 'darwin':
            return 'macos'
        if self.host == 'windows':
            return 'windows'
        return 'linux'

    def _detect_host(self) -> str:
        """Return a normalized host id: 'linux', 'darwin', or 'windows'."""
        p = platform.system().lower()
        if 'darwin' in p:
            return 'darwin'
        if 'windows' in p:
            return 'windows'
        return 'linux'
    
    def compile_to_assembly(self, input_file: str, output_file: str = None, quiet: bool = False, target: Optional[str] = None, arch: Optional[str] = None) -> bool:
        """Compile CASM to assembly file"""
        try:
            # The higher-level CLI now controls visible messages; avoid duplicate system prints
            from .utils.colors import print_debug
            print_debug(f"Assembly generation for {input_file}...")

            # Generate assembly
            # Generate assembly (unified/transpiler-friendly assembly)
            assembly_code = self.processor.process_file(input_file)

            # Transpile unified assembly into platform-specific assembly
            target = target or self._host_to_target()
            arch = arch or self.arch
            try:
                assembly_code = transpile_text(assembly_code, target=target, arch=arch, opt_level='none')
                # The transpiler may inject platform-specific section headers
                # and reorder sections. Run the AssemblyFixer to normalize the
                # output and then ensure string literals are placed under
                # section .data via the codegen helper.
                try:
                    from .core.assembly_fixer import assembly_fixer
                    assembly_code = assembly_fixer.fix_assembly(assembly_code)
                except Exception:
                    # Non-fatal: continue with transpiled code
                    pass

                try:
                    # Ensure any STR... db lines are in .data
                    assembly_code = self.processor.codegen._move_strings_to_data(assembly_code)
                except Exception:
                    pass
            except Exception:
                # If transpilation fails, fall back to original assembly
                pass

            # Determine output file
            if output_file is None:
                output_file = str(Path(input_file).with_suffix('.asm'))

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

            # Add debug info and format
            final_assembly = formatter.add_debug_info(assembly_code, input_file)

            # Final sanitization: ensure any duplicates accidentally
            # reintroduced later in the pipeline are removed. This is
            # defensive and uses the pretty printer's cleanup helpers to
            # collapse repeated section headers and remove consecutive
            # duplicate instruction lines (e.g., duplicated 'leave').
            try:
                from .core.pretty import pretty_printer
                final_assembly = pretty_printer._collapse_duplicate_section_headers(final_assembly)
                final_assembly = pretty_printer._remove_consecutive_duplicate_instructions(final_assembly)
            except Exception:
                # Non-fatal: if the pretty printer helpers are unavailable
                # or fail for some reason, continue with the current
                # assembly content rather than aborting the compilation.
                pass

            # Write assembly file (no automatic asm prelude; prettifier will
            # emit grouped externs when appropriate)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_assembly)

            if not quiet:
                print_success(f"Assembly generated: {output_file}")
            return True

        except Exception as e:
            term_error(f"Assembly generation failed: {e}")
            return False
    
    def compile_to_executable(self, input_file: str, output_executable: str = None, run_after: bool = False, quiet: bool = False, target: Optional[str] = None, arch: Optional[str] = None) -> bool:
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
            
            if not quiet:
                # set quiet mode for internal emits
                self._quiet_mode = bool(quiet)
                self._last_error = None
                self._last_info = None
                print_stage(1, 4, f"Processing {input_file}...")
            # Process CASM file
            assembly_code = self.processor.process_file(input_file)

            # Transpile unified assembly into platform-specific assembly
            target = target or self._host_to_target()
            arch = arch or self.arch
            try:
                assembly_code = transpile_text(assembly_code, target=target, arch=arch, opt_level='none')
            except Exception:
                # If transpilation fails, continue with the generated assembly
                pass

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
            
            if not quiet:
                print_stage(2, 4, "Assembling with NASM...")
            # Assemble with NASM
            obj_file = os.path.join(self.temp_dir, 'output.o')
            if not self._run_nasm(asm_file, obj_file, quiet=quiet):
                return False
            
            if not quiet:
                print_stage(3, 4, "Linking...")
            # Link executable (include driver object and any extra libs)
            exe_file = f"{output_executable}.exe"
            extra_objs = [driver_obj] if driver_obj else None
            if not self._run_linker(obj_file, exe_file, extra_objs=extra_objs, extra_libs=extra_libs, quiet=quiet):
                return False
            
            if not quiet:
                term_info(f"Executable created: {exe_file}")
            
            if run_after:
                if not quiet:
                    print_stage(4, 4, "Running executable...")
                return self._run_executable(exe_file)
            
            return True
            
        except Exception as e:
            # Capture error for outer display when quiet
            if self._quiet_mode:
                self._last_error = str(e)
            else:
                term_error(f"Compilation failed: {e}")
            return False
        
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            # reset quiet mode
            self._quiet_mode = False
    
    def _check_build_dependencies(self) -> bool:
        """Check if required build tools are available"""
        # Choose default toolchain depending on host. For Windows host prefer
        # MinGW cross/gcc; for Linux/macOS prefer system gcc/clang.
        if self.host == 'windows':
            required = ['nasm', 'x86_64-w64-mingw32-gcc']
        else:
            # On Linux prefer gcc; on macOS prefer clang but gcc shim is often ok
            required = ['nasm', shutil.which('gcc') and 'gcc' or 'clang']
        missing = []
        
        for tool in required:
            # some entries may be None if we used shutil.which during selection
            if not tool:
                continue
            if not shutil.which(tool):
                missing.append(tool)
        
        if missing:
            term_error(f"Missing build tools: {', '.join(missing)}")
            term_info("Install with: brew install nasm mingw-w64")
            return False
        
        return True

    def _nasm_format_for_host(self) -> str:
        """Return NASM output format appropriate for the host."""
        if self.host == 'darwin':
            return 'macho64'
        if self.host == 'windows':
            return 'win64'
        return 'elf64'

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
                elif 'sdl2/SDL.h' in name or 'SDL2/SDL.h'.lower() in name:
                    # Map SDL2 headers to SDL2 linking
                    libs.add('SDL2')
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
                term_error(f"Failed to compile automatic driver: {result.stderr}")
                return None

            return driver_o
        except Exception as e:
            term_error(f"Exception creating driver: {e}")
            return None
    
    def _run_nasm(self, asm_file: str, obj_file: str, quiet: bool = False) -> bool:
        """Run NASM assembler"""
        fmt = self._nasm_format_for_host()
        cmd = ['nasm', '-f', fmt, asm_file, '-o', obj_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if quiet or getattr(self, '_quiet_mode', False):
                # capture single-line error for outer display
                self._last_error = result.stderr.strip() or 'NASM failed'
            else:
                term_error("NASM failed:")
                term_info(result.stderr)
            return False

        return True
    
    def _run_linker(self, obj_file: str, exe_file: str, extra_objs: Optional[list] = None, extra_libs: Optional[list] = None, quiet: bool = False) -> bool:
        """Run linker, optionally adding extra object files and libraries."""
        # Choose linker based on host
        if self.host == 'windows':
            linker = 'x86_64-w64-mingw32-gcc'
        else:
            # Prefer system gcc/clang
            linker = shutil.which('gcc') or shutil.which('clang') or 'gcc'

        cmd = [linker]
        cmd.append(obj_file)
        if extra_objs:
            for eo in extra_objs:
                if eo:
                    cmd.append(eo)

        # Default linking flags depend on host
        if self.host == 'windows':
            cmd.extend(['-o', exe_file, '-mconsole', '-lkernel32', '-lmsvcrt'])
        else:
            # Linux and macOS: produce a native executable
            cmd.extend(['-o', exe_file])
            # On Linux ensure no PIE for simple execs if desired
            if self.host == 'linux':
                cmd.append('-no-pie')

        if extra_libs:
            for lib in extra_libs:
                cmd.append(f'-l{lib}')

        # Gather extra linker flags from C processor (e.g., SDL2 pkg-config)
        try:
            pkg_link_flags = []
            # If user provided ldflags via CLI, prefer those
            if getattr(c_processor, 'user_ldflags', None):
                # user_ldflags may be string or list
                if isinstance(c_processor.user_ldflags, str):
                    import shlex
                    pkg_link_flags = shlex.split(c_processor.user_ldflags)
                else:
                    pkg_link_flags = list(c_processor.user_ldflags)
            elif hasattr(c_processor, '_gather_pkg_link_flags_for_headers'):
                pkg_link_flags = c_processor._gather_pkg_link_flags_for_headers()

            # If pkg-config returned flags, try to validate that any -L paths
            # actually contain MinGW-compatible import libraries (libSDL2.dll.a or libSDL2.a).
            def _libs_present_in_dir(d: str):
                """Return:
                - 'dll_a' if libSDL2.dll.a present (preferred MinGW import lib)
                - 'static_a' if libSDL2.a present (likely host static lib — may be incompatible)
                - False if none found
                """
                import os
                dll_a = os.path.join(d, 'libSDL2.dll.a')
                static_a = os.path.join(d, 'libSDL2.a')
                if os.path.exists(dll_a):
                    return 'dll_a'
                if os.path.exists(static_a):
                    return 'static_a'
                return False

            valid_flags = []
            if pkg_link_flags:
                # Collect -L dirs and other flags
                L_dirs = [f[2:] for f in pkg_link_flags if f.startswith('-L')]
                has_libSDL2 = any(f == '-lSDL2' or f == ' -lSDL2' for f in pkg_link_flags) or ('SDL2' in (extra_libs or []))

                usable = False
                found_static_only = False
                for d in L_dirs:
                    res = _libs_present_in_dir(d)
                    if res == 'dll_a':
                        usable = True
                        break
                    if res == 'static_a':
                        found_static_only = True

                # If no usable import libs found in pkg-config dirs, check environment override
                if not usable and has_libSDL2:
                    env_dir = os.environ.get('CASM_MINGW_LIB_DIR')
                    if env_dir:
                        res = _libs_present_in_dir(env_dir)
                        if res == 'dll_a':
                            if quiet or getattr(self, '_quiet_mode', False):
                                self._last_info = f"Using CASM_MINGW_LIB_DIR for MinGW SDL2 libs: {env_dir}"
                            else:
                                term_info(f"Using CASM_MINGW_LIB_DIR for MinGW SDL2 libs: {env_dir}")
                            cmd.extend([f'-L{env_dir}', '-lSDL2'])
                            usable = True
                        elif res == 'static_a':
                            # Found a host static lib — don't accept it for mingw
                            if quiet or getattr(self, '_quiet_mode', False):
                                self._last_error = f"CASM_MINGW_LIB_DIR contains libSDL2.a which is likely a host static library and not a MinGW import library: {env_dir}"
                                self._last_info = "You still need a MinGW import library (libSDL2.dll.a)."
                            else:
                                term_error(f"CASM_MINGW_LIB_DIR contains libSDL2.a which is likely a host static library and not a MinGW import library: {env_dir}")
                                term_info("You still need a MinGW import library (libSDL2.dll.a).")

                if usable:
                    # If usable pkg flags exist, append all pkg flags
                    cmd.extend(pkg_link_flags)
                    if quiet or getattr(self, '_quiet_mode', False):
                        self._last_info = f"Appending pkg-config linker flags: {' '.join(pkg_link_flags)}"
                    else:
                        term_info(f"Appending pkg-config linker flags: {' '.join(pkg_link_flags)}")
                else:
                    # We have pkg flags but they point to host libs which are not
                    # MinGW-compatible. Provide a clear, actionable error.
                    if has_libSDL2:
                        # Emit one concise message (minimal) and one short suggestion
                        if found_static_only:
                            err = "SDL2 pkg-config points to a host static library (libSDL2.a); MinGW needs libSDL2.dll.a"
                        else:
                            err = "SDL2 pkg-config did not provide MinGW import libraries (libSDL2.dll.a)"

                        suggestion = "Provide MinGW SDL2 import libs and set CASM_MINGW_LIB_DIR to that lib directory"
                        if quiet or getattr(self, '_quiet_mode', False):
                            self._last_error = err
                            self._last_info = suggestion
                        else:
                            term_error(err)
                            term_info(suggestion)
                        return False
                    else:
                        # No SDL2 requested or no -lSDL2 present; just append whatever pkg gave
                        cmd.extend(pkg_link_flags)
                        term_info(f"Appending pkg-config linker flags: {' '.join(pkg_link_flags)}")
            # If no pkg flags but extra_libs contains SDL2, try CASM_MINGW_LIB_DIR
            else:
                if extra_libs and 'SDL2' in extra_libs:
                    env_dir = os.environ.get('CASM_MINGW_LIB_DIR')
                    if env_dir and _libs_present_in_dir(env_dir):
                        cmd.extend([f'-L{env_dir}', '-lSDL2'])
                        if quiet or getattr(self, '_quiet_mode', False):
                            self._last_info = f"Using CASM_MINGW_LIB_DIR for SDL2: {env_dir}"
                        else:
                            term_info(f"Using CASM_MINGW_LIB_DIR for SDL2: {env_dir}")
                    else:
                        if quiet or getattr(self, '_quiet_mode', False):
                            self._last_error = "Linking requires SDL2 import libraries for MinGW but none were found."
                            self._last_info = "Set CASM_MINGW_LIB_DIR to a directory containing libSDL2.dll.a and try again."
                        else:
                            term_error("Linking requires SDL2 import libraries for MinGW but none were found.")
                            term_info("Set the CASM_MINGW_LIB_DIR environment variable to a directory containing libSDL2.dll.a and try again.")
                        return False
        except Exception as e:
            term_info(f"Could not gather pkg-config linker flags: {e}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if quiet or getattr(self, '_quiet_mode', False):
                self._last_error = result.stderr.strip() or 'Linker failed'
            else:
                term_error("Linking failed:")
                term_info(result.stderr)
            return False

        return True
    
    def _run_executable(self, exe_file: str) -> bool:
        """Run the compiled executable"""
        wine_cmd = shutil.which('wine') or shutil.which('wine64')
        
        if wine_cmd:
            term_info("-" * 40)
            try:
                result = subprocess.run([wine_cmd, exe_file], capture_output=False)
                term_info("-" * 40)
                term_info(f"Program exited with code: {result.returncode}")
                return result.returncode == 0
            except KeyboardInterrupt:
                term_info("\nProgram interrupted by user")
                return True
            except Exception as e:
                term_error(f"Failed to run executable: {e}")
                return False
        else:
            term_info("Wine not found, cannot run Windows executable")
            term_info(f"Executable saved as: {exe_file}")
            term_info("Install Wine to run: brew install --cask wine-stable")
            return True

# Global instances
processor = CASMProcessor()
compiler = CASMCompiler()