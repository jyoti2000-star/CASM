#!/usr/bin/env python3

"""
Core CASM compiler and processor
Combines lexing, parsing, and code generation in clean architecture
"""

import os
import sys
import tempfile
import subprocess
import shutil
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

            # Write assembly file
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
            
            # Write assembly to temp file
            asm_file = os.path.join(self.temp_dir, 'output.asm')
            with open(asm_file, 'w', encoding='utf-8') as f:
                f.write(assembly_code)
            
            print("[2/4] Assembling with NASM...")
            # Assemble with NASM
            obj_file = os.path.join(self.temp_dir, 'output.o')
            if not self._run_nasm(asm_file, obj_file):
                return False
            
            print("[3/4] Linking...")
            # Link executable
            exe_file = f"{output_executable}.exe"
            if not self._run_linker(obj_file, exe_file):
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
    
    def _run_nasm(self, asm_file: str, obj_file: str) -> bool:
        """Run NASM assembler"""
        cmd = ['nasm', '-f', 'win64', asm_file, '-o', obj_file]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] NASM failed:")
            print(result.stderr)
            return False
        
        return True
    
    def _run_linker(self, obj_file: str, exe_file: str) -> bool:
        """Run linker"""
        cmd = ['x86_64-w64-mingw32-gcc', obj_file, '-o', exe_file, 
               '-mconsole', '-lkernel32', '-lmsvcrt']
        
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