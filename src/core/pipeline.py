from typing import Optional
from .diagnostics import DiagnosticEngine
from .lexer import CASMLexer
from .parser import CASMParser
from .codegen import AssemblyCodeGenerator
import os
import re
import subprocess
from pathlib import Path

class CompilerConfig:
    def __init__(self):
        self.optimization_level = 0
        self.target_arch = "x86_64"
        self.target_os = "linux"
        self.emit_debug_info = False
        self.verbose = False
        self.warnings_as_errors = False
        self.unused_variable_warning = True
        self.output_format = "nasm"

class CASMCompiler:
    def __init__(self, config: Optional[CompilerConfig] = None):
        self.config = config or CompilerConfig()
        self.diagnostics = DiagnosticEngine()
        
    def compile(self, source: str, filename: str = "<stdin>", extract_c: bool = False, compile_embedded_c: bool = False) -> Optional[str]:
        if self.config.verbose:
            print("Phase 1: Lexical Analysis")
        lexer = CASMLexer(source, filename)
        tokens = lexer.tokenize()
        if lexer.diagnostics.has_errors():
            lexer.diagnostics.print_all()
            return None
        if self.config.verbose:
            print(f"  Generated {len(tokens)} tokens")

        if self.config.verbose:
            print("Phase 2: Parsing")
        # Pass the original source text to the parser for simplified parsing
        parser = CASMParser(source)
        ast = parser.parse()
        if parser.diagnostics.has_errors():
            parser.diagnostics.print_all()
            return None
        if not ast:
            print("Parse failed: no AST generated")
            return None
        if self.config.verbose:
            print(f"  Generated AST with {len(ast.statements)} statements")

        if self.config.verbose:
            print("Phase 5: Code Generation")
        codegen = AssemblyCodeGenerator(self.config.target_os)
        assembly = codegen.generate(ast)
        if hasattr(codegen, 'diagnostics') and codegen.diagnostics.has_errors():
            codegen.diagnostics.print_all()
            return None

        if parser.diagnostics.warning_count > 0:
            parser.diagnostics.print_all()
        if hasattr(codegen, 'diagnostics') and getattr(codegen.diagnostics, 'warning_count', 0) > 0:
            codegen.diagnostics.print_all()

        if self.config.verbose:
            print(f"Compilation successful!")
            print(f"  {parser.diagnostics.warning_count} warnings")

        # Handle extraction/compilation of embedded C if requested
        try:
            self._handle_embedded_c(assembly, filename, extract_c, compile_embedded_c)
        except Exception as e:
            # Don't fail the whole compilation for non-critical extraction errors
            print(f"Warning: embedded C handling failed: {e}")

        return assembly

        # NOTE: extraction/compilation of embedded C is intentionally done
        # after generation so that the pipeline can produce companion files
        # when requested. We perform file writes into the `output/` folder
        # using the source filename stem (if provided) so callers can find
        # the generated artifacts.
        # However we still return the assembly string so callers may write it
        # where they prefer.
        
    def _handle_embedded_c(self, assembly: str, source_filename: str, extract_c: bool, compile_embedded_c: bool):
        """Extract embedded C passthrough from assembly and optionally compile it.

        Writes files to output/<stem>.c, .o and .c.s when requested.
        """
        if not (extract_c or compile_embedded_c):
            return

        start_marker = '; --- begin embedded C code passthrough ---'
        end_marker = '; --- end embedded C code passthrough ---'
        if start_marker not in assembly or end_marker not in assembly:
            return

        start = assembly.index(start_marker) + len(start_marker)
        end = assembly.index(end_marker)
        c_block = assembly[start:end]
        # strip leading ';' and whitespace from each line
        c_lines = []
        for line in c_block.splitlines():
            l = line.lstrip()
            if l.startswith(';'):
                l = l[1:]
            c_lines.append(l.lstrip())

        c_text = '\n'.join(c_lines).strip() + '\n'
        # convert CASM extern <...> into real C include
        c_text = re.sub(r"^\s*extern\s*<([^>]+)>", r"#include <\1>", c_text, flags=re.M)

        # Determine output paths using source filename stem
        src_path = Path(source_filename)
        out_dir = Path('output')
        out_dir.mkdir(parents=True, exist_ok=True)
        asm_out = out_dir / (src_path.stem + '.asm')
        c_path = out_dir / (src_path.stem + '.c')
        write_c = False
        try:
            c_path.write_text(c_text, encoding='utf-8')
            write_c = True
        except Exception:
            write_c = False

        if write_c:
            print(f"Extracted embedded C to: {c_path}")

        if compile_embedded_c and write_c:
            cc = os.environ.get('CC', 'gcc')
            obj_path = out_dir / (src_path.stem + '.o')
            s_path = out_dir / (src_path.stem + '.c.s')

            compile_cmd = [cc]
            if self.config.target_os == 'windows':
                compile_cmd += ['-DUNICODE', '-D_UNICODE', '-DWIN32_LEAN_AND_MEAN']
            compile_cmd += ['-c', str(c_path), '-o', str(obj_path)]
            print(f"Compiling embedded C to object with: {' '.join(compile_cmd)}")
            proc = subprocess.run(compile_cmd)
            if proc.returncode == 0:
                print(f"Compiled embedded C to object: {obj_path}")
            else:
                print(f"Failed to compile embedded C (exit {proc.returncode}).")

            asm_cmd = [cc]
            if self.config.target_os == 'windows':
                asm_cmd += ['-DUNICODE', '-D_UNICODE', '-DWIN32_LEAN_AND_MEAN']
            asm_cmd += ['-S', str(c_path), '-o', str(s_path)]
            print(f"Generating assembly from embedded C with: {' '.join(asm_cmd)}")
            proc2 = subprocess.run(asm_cmd)
            if proc2.returncode == 0 and s_path.exists():
                print(f"Generated assembly for embedded C: {s_path}")
                try:
                    asm_extra = s_path.read_text(encoding='utf-8')
                    with open(asm_out, 'a', encoding='utf-8') as f:
                        f.write('\n; --- begin embedded C compiled assembly ---\n')
                        f.write(asm_extra)
                        f.write('\n; --- end embedded C compiled assembly ---\n')
                    print(f"Appended compiled assembly into: {asm_out}")
                except Exception as e:
                    print(f"Failed to append compiled assembly into {asm_out}: {e}")
            else:
                print(f"Failed to generate assembly from embedded C (exit {proc2.returncode}). Ensure your CC supports the target and headers.")

def create_default_compiler() -> CASMCompiler:
    config = CompilerConfig()
    return CASMCompiler(config)

def compile_string(source: str, verbose: bool = False, extract_c: bool = False, compile_embedded_c: bool = False) -> Optional[str]:
    config = CompilerConfig()
    config.verbose = verbose
    compiler = CASMCompiler(config)
    return compiler.compile(source, '<string>', extract_c=extract_c, compile_embedded_c=compile_embedded_c)

def compile_file(filepath: str, verbose: bool = False, target_os: str = 'linux', extract_c: bool = False, compile_embedded_c: bool = False) -> Optional[str]:
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        config = CompilerConfig()
        config.verbose = verbose
        config.target_os = target_os
        compiler = CASMCompiler(config)
        return compiler.compile(source, filepath, extract_c=extract_c, compile_embedded_c=compile_embedded_c)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
