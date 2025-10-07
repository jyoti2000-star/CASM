#!/usr/bin/env python3

import sys
import os
import platform
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from lexer import Lexer
from parser import Parser
from codegen import CodeGenerator
from symbol_table import SymbolTable, global_symbol_table
from optimizer import OptimizationEngine, OptimizationLevel
from error_handler import ErrorHandler, global_error_handler
from preprocessor import Preprocessor
from cross_platform import AssemblyConverter, global_assembly_converter
from code_analysis import CodeQualityEngine
from stdlib_extended import StandardLibrary, global_stdlib
from c_codegen import CCodeGenerator
from assembly_processor import ASMFormatter, AdvancedAssemblyProcessor, process_assembly_with_comparison

# Color constants
class Colors:
    RED = '\033[91m'      # Error messages
    ORANGE = '\033[38;5;208m'  # System commands (proper orange)
    GREEN = '\033[92m'    # Success messages
    BLUE = '\033[94m'     # Info messages
    YELLOW = '\033[93m'   # Warning messages
    RESET = '\033[0m'     # Reset to default

def print_error(message):
    """Print error message with red [x]"""
    print(f"{Colors.RED}[x]{Colors.RESET} {message}", file=sys.stderr)

def print_system(message):
    """Print system command message with orange [*]"""
    print(f"{Colors.ORANGE}[*]{Colors.RESET} {message}")

def print_success(message):
    """Print success message with green [+]"""
    print(f"{Colors.GREEN}[+]{Colors.RESET} {message}")

def print_info(message):
    """Print info message with blue [-]"""
    print(f"{Colors.BLUE}[-]{Colors.RESET} {message}")

def print_warning(message):
    """Print warning message with yellow [!]"""
    print(f"{Colors.YELLOW}[!]{Colors.RESET} {message}")


class HLASMPreprocessor:
    def __init__(self, target_os: str = None, target_arch: str = None, optimization_level: str = "O2"):
        self.lexer = Lexer()
        self.parser = Parser()
        self.codegen = CodeGenerator(target_os, target_arch)
        self.c_codegen = CCodeGenerator()  # For processing C commands
        self.preprocessor = Preprocessor()
        self.symbol_table = SymbolTable()
        self.error_handler = ErrorHandler()
        self.quality_engine = CodeQualityEngine()
        self.formatter = ASMFormatter()  # Add formatter
        
        # Set optimization level
        opt_levels = {
            "O0": OptimizationLevel.O0,
            "O1": OptimizationLevel.O1, 
            "O2": OptimizationLevel.O2,
            "O3": OptimizationLevel.O3,
            "Os": OptimizationLevel.Os
        }
        self.optimizer = OptimizationEngine(opt_levels.get(optimization_level, OptimizationLevel.O2))
        
        # Set target platform
        platform_name = f"{self.codegen.platform.os_name}-{self.codegen.platform.architecture}"
        global_assembly_converter.set_target_platform(platform_name)
    
    def process_file(self, filename: str, enable_optimization: bool = True, 
                    enable_analysis: bool = True) -> str:
        """Process high-level assembly file with 3-stage pipeline: C codegen → HLASM codegen → Pure assembly"""
        try:
            # Set current file for error reporting
            self.error_handler.set_current_file(filename)
            
            print_info("Starting 3-stage compilation pipeline...")
            
            # STAGE 1: Process C commands embedded in assembly using c_codegen.py
            print_system("[Stage 1/3] Processing C commands with c_codegen.py...")
            stage1_output = self._process_c_commands(filename)
            
            # STAGE 2: Process high-level assembly commands using codegen.py  
            print_system("[Stage 2/3] Processing high-level assembly with codegen.py...")
            stage2_output = self._process_hlasm_commands(stage1_output)
            
            # STAGE 3: Final assembly processing and optimization
            print_system("[Stage 3/3] Final processing and optimization...")
            
            # Parse remaining assembly
            output_lines = stage2_output.split('\n')
            
            # Optimization phase
            if enable_optimization:
                output_lines = self.optimizer.optimize(output_lines)
                final_output = '\n'.join(output_lines)
            else:
                final_output = stage2_output
            
            # Code quality analysis
            if enable_analysis:
                self.quality_engine.analyze_code(output_lines)
            
            # Check for errors
            if self.error_handler.has_errors():
                print_error("Compilation failed due to errors:")
                print(self.error_handler.generate_report(show_context=True))
                sys.exit(1)
            
            # Apply formatter to final output
            print_system("[Final] Formatting final assembly...")
            final_output = self._format_assembly_content(final_output)
            
            # Apply advanced assembly processing with automatic extern detection
            print_system("[Post-processing] Applying advanced assembly processing with extern detection...")
            try:
                processor = AdvancedAssemblyProcessor(debug=False)
                final_output = processor.process_assembly(final_output)
                print_success("Advanced assembly processing completed successfully")
            except Exception as e:
                print_warning(f"Advanced assembly processing failed: {e}, continuing...")
            
            # Apply comprehensive final formatting
            print_system("[Final Formatting] Applying comprehensive assembly formatting...")
            try:
                from formatter import AssemblyFormatter
                final_formatter = AssemblyFormatter(debug=False)
                
                # Create temporary files for final formatting
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='_pre_final.asm', delete=False) as temp_input:
                    temp_input.write(final_output)
                    temp_input_path = temp_input.name
                
                with tempfile.NamedTemporaryFile(mode='r', suffix='_final.asm', delete=False) as temp_output:
                    temp_output_path = temp_output.name
                
                # Format with the comprehensive formatter
                success = final_formatter.format_file(temp_input_path, temp_output_path)
                
                if success:
                    with open(temp_output_path, 'r') as f:
                        final_output = f.read()
                    print_success("Comprehensive assembly formatting completed successfully")
                else:
                    print_warning("Comprehensive formatting failed, using previous output")
                
                # Clean up temporary files
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                    
            except ImportError:
                print_warning("Comprehensive formatter not found, skipping final formatting...")
            except Exception as e:
                print_warning(f"Comprehensive formatting failed: {e}, continuing with previous output...")
            
            # Apply conservative assembly fixing (only critical syntax errors)
            print_system("[Post-processing] Applying conservative assembly fixes...")
            try:
                from conservative_assembly_fixer import fix_assembly_conservative
                final_output = fix_assembly_conservative(final_output)
                print_success("Conservative assembly fixes applied successfully")
            except ImportError:
                print_warning("Post-processing module not found, applying built-in fixes...")
                # Apply built-in NASM compatibility fixes
                final_output = self._fix_nasm_compatibility_issues(final_output)
                print_success("Built-in NASM fixes applied successfully")
            except Exception as e:
                print_warning(f"Post-processing failed: {e}, applying built-in fixes...")
                # Apply built-in NASM compatibility fixes as fallback
                final_output = self._fix_nasm_compatibility_issues(final_output)
                print_success("Built-in NASM fixes applied successfully")
            
            # Save final formatted assembly with random name
            import random
            import string
            
            # Generate random prefix  
            random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            base_name = os.path.splitext(os.path.basename(filename))[0] 
            final_asm_file = f"output/{random_chars}_{base_name}_final_formatted.asm"
            
            os.makedirs("output", exist_ok=True)
            with open(final_asm_file, 'w') as f:
                f.write(final_output)
            print_success(f"Final formatted assembly saved to: {final_asm_file}")
            
            print_success("3-stage pipeline with comprehensive formatting completed successfully!")
            return final_output
            
        except FileNotFoundError:
            print_error(f"Error: File '{filename}' not found")
            sys.exit(1)
        except Exception as e:
            print_error(f"Error processing file: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            sys.exit(1)

    
    def _process_c_commands(self, filename: str) -> str:
        """Stage 1: Process C commands using c_codegen.py"""
        import tempfile
        import os
        
        # Create a temporary file for intermediate output
        with tempfile.NamedTemporaryFile(mode='w', suffix='_stage1.asm', delete=False) as temp_file:
            temp_output_path = temp_file.name
        
        try:
            # Use c_codegen to process C commands
            success = self.c_codegen.process_asm_file(filename, temp_output_path, "optimized", "windows")
            
            if not success:
                raise Exception("C code generation failed")
            
            # Read the processed content
            with open(temp_output_path, 'r') as f:
                stage1_content = f.read()
            
            return stage1_content
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
    
    def _format_assembly_content(self, content: str) -> str:
        """Format assembly content with proper section headers"""
        import tempfile
        import os
        
        # Create temporary files for formatting
        with tempfile.NamedTemporaryFile(mode='w', suffix='_unformatted.asm', delete=False) as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(mode='r', suffix='_formatted.asm', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Create a new formatter instance to avoid conflicts
            formatter = ASMFormatter()
            
            # Format the content
            success = formatter.format_file(temp_input_path, temp_output_path)
            
            if success:
                # Read the formatted content
                with open(temp_output_path, 'r') as f:
                    formatted_content = f.read()
                return formatted_content
            else:
                print_warning("Formatting failed, using original content")
                return content
                
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
    
    def _fix_nasm_compatibility_issues(self, assembly_content: str) -> str:
        """Fix common NASM compatibility issues in generated assembly"""
        lines = assembly_content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Fix the specific double-dereference pattern for HASM variables
            # Pattern: mov rbx, [rel V01] followed by mov eax, dword [rbx]
            if ('mov rbx, [rel V01]' in line.strip() and 
                i + 1 < len(lines) and 
                'mov eax, dword [rbx]' in lines[i + 1].strip()):
                
                # Replace with direct access
                indent = line[:len(line) - len(line.lstrip())]
                fixed_lines.append(f'{indent}mov eax, dword [rel V01]  ; Fixed: Direct access to HASM variable')
                i += 2  # Skip both lines
                continue
            
            # Fix similar patterns for other variables (V02, V03, etc.)
            if 'mov rbx, [rel V' in line.strip() and i + 1 < len(lines):
                # Extract variable name (V01, V02, etc.)
                import re
                match = re.search(r'mov rbx, \[rel (V\d+)\]', line.strip())
                if match and 'mov eax, dword [rbx]' in lines[i + 1].strip():
                    var_name = match.group(1)
                    indent = line[:len(line) - len(line.lstrip())]
                    fixed_lines.append(f'{indent}mov eax, dword [rel {var_name}]  ; Fixed: Direct access to HASM variable')
                    i += 2  # Skip both lines
                    continue
            
            # Fix register size mismatches: mov 32bit_reg, 64bit_reg
            if 'mov edx, rax' in line:
                import re
                line = re.sub(r'mov edx, rax', 'mov edx, eax  ; Fixed: Register size match', line)
            elif 'mov ecx, rax' in line:
                import re  
                line = re.sub(r'mov ecx, rax', 'mov ecx, eax  ; Fixed: Register size match', line)
            elif 'mov r8d, rax' in line:
                import re
                line = re.sub(r'mov r8d, rax', 'mov r8d, eax  ; Fixed: Register size match', line)
            elif 'mov r9d, rax' in line:
                import re
                line = re.sub(r'mov r9d, rax', 'mov r9d, eax  ; Fixed: Register size match', line)
            
            # Fix register size mismatches: add 32bit_reg, 64bit_reg
            elif 'add edx, rax' in line:
                import re
                line = re.sub(r'add edx, rax', 'add edx, eax  ; Fixed: Register size match', line)
            elif 'add ecx, rax' in line:
                import re
                line = re.sub(r'add ecx, rax', 'add ecx, eax  ; Fixed: Register size match', line)
            elif 'add r8d, rax' in line:
                import re
                line = re.sub(r'add r8d, rax', 'add r8d, eax  ; Fixed: Register size match', line)
            elif 'add r9d, rax' in line:
                import re
                line = re.sub(r'add r9d, rax', 'add r9d, eax  ; Fixed: Register size match', line)
            
            # Fix standalone [rbx] references (likely orphaned from double-dereference)
            elif 'dword [rbx]' in line and 'mov' in line:
                import re
                line = re.sub(r'dword \[rbx\]', 'dword [rel V01]  ; Fixed: HASM variable reference', line)
            
            # Fix standalone [rax] references  
            elif 'dword [rax]' in line and 'mov' in line:
                import re
                line = re.sub(r'dword \[rax\]', 'dword [rel V01]  ; Fixed: HASM variable reference', line)
            
            # Fix add operations with [rbx]
            elif 'add' in line and '[rbx]' in line:
                import re
                line = re.sub(r'\[rbx\]', '[rel V01]  ; Fixed: HASM variable reference', line)
            
            # Keep original line if no pattern matched
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _process_hlasm_commands(self, content: str) -> str:
        """Stage 2: Process high-level assembly commands using codegen.py"""
        # Check if there are any high-level assembly commands to process
        if not any(line.strip().startswith('%') and not line.strip().startswith('%!') 
                  for line in content.split('\n')):
            # No high-level assembly commands, return as-is
            return content
        
        try:
            # Extract existing data sections from c_codegen output
            existing_data_sections = self._extract_existing_data_sections(content)
            
            # Tokenize the content
            tokens = self.lexer.tokenize(content)
            
            # Parse with error handling
            ast = self.parser.parse(tokens)
            
            # Generate assembly code
            hlasm_output = self.codegen.generate(ast)
            
            # Merge existing data sections with new output
            merged_output = self._merge_data_sections(hlasm_output, existing_data_sections)
            
            return merged_output
            
        except Exception as e:
            # If parsing fails, it might be pure assembly already
            print_warning(f"Stage 2 parsing failed, treating as pure assembly: {e}")
            return content
    
    def _extract_existing_data_sections(self, content: str) -> dict:
        """Extract existing data sections from content"""
        lines = content.split('\n')
        data_sections = {'data': [], 'bss': []}
        current_section = None
        in_data_section = False
        
        for line in lines:
            stripped = line.strip()
            if stripped == '; DATA START':
                in_data_section = True
                current_section = 'data'
                continue
            elif stripped == '; DATA END':
                in_data_section = False
                current_section = None
                continue
            elif in_data_section and current_section:
                data_sections[current_section].append(line)
        
        return data_sections
    
    def _merge_data_sections(self, hlasm_output: str, existing_data: dict) -> str:
        """Merge existing data sections with new HLASM output, removing duplicates"""
        lines = hlasm_output.split('\n')
        result = []
        in_data_section = False
        data_section_written = False
        skip_until_data_end = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip duplicate data sections that come after the first one
            if stripped == '; DATA START' and data_section_written:
                skip_until_data_end = True
                continue
            elif stripped == '; DATA END' and skip_until_data_end:
                skip_until_data_end = False
                continue
            elif skip_until_data_end:
                continue
            
            # When we encounter the first data section marker, merge everything
            if stripped == '; DATA START' and not data_section_written:
                result.append(line)
                
                # Add existing data first
                if existing_data['data']:
                    result.extend(existing_data['data'])
                
                # Add new data from HLASM
                in_data_section = True
                data_section_written = True
                continue
            elif stripped == '; DATA END' and in_data_section:
                result.append(line)
                in_data_section = False
                continue
            elif in_data_section:
                # Add new HLASM data (this will be the %println strings)
                result.append(line)
                continue
            else:
                # Regular lines
                result.append(line)
        
        return '\n'.join(result)
    
    def get_stdlib_used(self):
        """Get the set of standard library functions used"""
        stdlib_used = set()
        if hasattr(self.codegen, 'stdlib_used'):
            stdlib_used.update(self.codegen.stdlib_used)
        if hasattr(self.c_codegen, 'stdlib_used'):
            stdlib_used.update(getattr(self.c_codegen, 'stdlib_used', set()))
        return stdlib_used
    
    def get_optimization_report(self) -> str:
        """Get optimization report"""
        return self.optimizer.get_optimization_report()
    
    def get_quality_report(self) -> str:
        """Get code quality analysis report"""
        return self.quality_engine.generate_quality_report()
    
    def get_symbol_report(self) -> str:
        """Get symbol table report"""
        return self.symbol_table.generate_symbol_report()
    
    def get_platform_report(self) -> str:
        """Get platform compatibility report"""
        return global_assembly_converter.generate_platform_report()
    
    def export_reports(self, base_filename: str):
        """Export all compiler reports to files"""
        reports = {
            'optimization': self.get_optimization_report(),
            'quality': self.get_quality_report(),
            'symbols': self.get_symbol_report(),
            'platform': self.get_platform_report()
        }
        
        for report_type, content in reports.items():
            filename = f"{base_filename}_{report_type}.txt"
            with open(filename, 'w') as f:
                f.write(content)
            print_success(f"Exported {report_type} report to {filename}")


class HLASMCompiler:
    def __init__(self, target_os: str = None, target_arch: str = None, 
                 optimization_level: str = "O2", enable_analysis: bool = True):
        self.preprocessor = HLASMPreprocessor(target_os, target_arch, optimization_level)
        self.temp_dir = None
        self.target_os = self.preprocessor.codegen.platform.os_name
        self.target_arch = self.preprocessor.codegen.platform.architecture
        self.output_dir = "output"
        self.optimization_level = optimization_level
        self.enable_analysis = enable_analysis
        self._ensure_output_dir()
        
        # Print target platform info
        print_info(f"Target platform: {self.target_os}-{self.target_arch}")
        print_info(f"Object format: {self.preprocessor.codegen.platform.object_format}")
        print_info(f"Calling convention: {self.preprocessor.codegen.platform.calling_convention}")
    
    def _ensure_output_dir(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print_success(f"Created output directory: {self.output_dir}")
    
    def check_dependencies(self):
        """Check if required tools are available"""
        required_tools = ['nasm', 'ld']
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            print_error(f"Error: Missing required tools: {', '.join(missing_tools)}")
            print_error("Please install:")
            for tool in missing_tools:
                if tool == 'nasm':
                    print_error("  - NASM assembler (brew install nasm)")
                elif tool == 'ld':
                    print_error("  - GNU linker (usually comes with Xcode Command Line Tools)")
            return False
        return True
    
    def compile_file(self, input_file: str, output_executable: str = None, 
                    run_after_compile: bool = True, generate_reports: bool = False,
                    verbose: bool = False):
        """Compile HLASM file to executable with advanced features"""
        
        if not self.check_dependencies():
            return False
        
        input_path = Path(input_file)
        if not input_path.exists():
            print_error(f"Error: Input file '{input_file}' not found")
            return False
        
        # Set default output executable name in output directory
        if output_executable is None:
            output_executable = os.path.join(self.output_dir, input_path.stem)
        elif not os.path.dirname(output_executable):
            # If no directory specified, put in output folder
            output_executable = os.path.join(self.output_dir, output_executable)
        
        try:
            # Create temporary directory for intermediate files
            self.temp_dir = tempfile.mkdtemp(prefix='hlasm_compile_')
            
            # Step 1: Preprocess high-level assembly to standard assembly
            print_system(f"[1/5] Preprocessing {input_file}...")
            asm_output = self.preprocessor.process_file(input_file, 
                                                       enable_optimization=True,
                                                       enable_analysis=self.enable_analysis)
            
            if verbose:
                print_info(f"Generated {len(asm_output.split(chr(10)))} lines of final assembly")
                stdlib_used = self.preprocessor.get_stdlib_used()
                if stdlib_used:
                    print_info(f"Standard library functions: {', '.join(sorted(stdlib_used))}")
                print_info("Pipeline stages completed: C codegen → HLASM codegen → Pure assembly")
            
            # Write preprocessed assembly to temporary file
            asm_file = os.path.join(self.temp_dir, 'output.asm')
            with open(asm_file, 'w') as f:
                f.write(asm_output)
            
            # Step 2: Show analysis results if enabled
            if self.enable_analysis:
                print_system("[2/5] Running code analysis...")
                quality_report = self.preprocessor.get_quality_report()
                
                # Show summary
                lines = quality_report.split('\n')
                for line in lines:
                    if 'Total Optimizations Applied:' in line:
                        print_info(line)
                    elif '[ERROR]' in line or '[WARNING]' in line:
                        if '[ERROR]' in line:
                            print_error(line)
                        else:
                            print_warning(line)
                
                if verbose:
                    print(quality_report)
            
            # Step 3: Assemble with NASM
            print_system("[3/5] Assembling with NASM...")
            obj_file = os.path.join(self.temp_dir, 'output.o')
            
            nasm_cmd = ['nasm', '-f', self._get_object_format(), asm_file, '-o', obj_file]
            
            result = subprocess.run(nasm_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print_error(f"NASM assembly failed:")
                print_error(result.stderr)
                return False
            
            # Step 4: Link
            print_system("[4/5] Linking...")
            
            # Check if we're cross-compiling
            import platform as host_platform
            host_os = host_platform.system().lower()
            
            if self.target_os == 'windows':
                # For Windows cross-compilation using MinGW-w64
                mingw_ld = shutil.which('x86_64-w64-mingw32-ld')
                mingw_gcc = shutil.which('x86_64-w64-mingw32-gcc')
                
                # Prefer GCC over ld for easier library handling
                if mingw_gcc:
                    print_info("Using MinGW-w64 GCC for Windows cross-compilation")
                    # Update output_executable to include .exe extension
                    output_executable_with_ext = f'{output_executable}.exe'
                    # Use GCC with proper linking - console application with C runtime
                    ld_cmd = ['x86_64-w64-mingw32-gcc', 
                             obj_file, '-o', output_executable_with_ext,
                             '-mconsole', '-lkernel32', '-lmsvcrt']
                    # Update output_executable for subsequent steps
                    output_executable = output_executable_with_ext
                elif mingw_ld:
                    print_info("Using MinGW-w64 ld for Windows cross-compilation")
                    # Update output_executable to include .exe extension
                    output_executable_with_ext = f'{output_executable}.exe'
                    # Get the MinGW library path
                    mingw_lib_path = "/opt/homebrew/Cellar/mingw-w64/13.0.0_2/toolchain-x86_64/x86_64-w64-mingw32/lib"
                    # Use ld directly with explicit library path
                    ld_cmd = ['x86_64-w64-mingw32-ld', 
                             '--entry=main',           # Entry point
                             '--subsystem=console',    # Console subsystem
                             f'-L{mingw_lib_path}',    # Library search path
                             obj_file, 
                             '-o', output_executable_with_ext,
                             '-lkernel32', '-luser32']  # Required Windows libraries
                    # Update output_executable for subsequent steps
                    output_executable = output_executable_with_ext
                else:
                    print_warning("MinGW-w64 not found. Install with: brew install mingw-w64")
                    print_error("Windows cross-compilation requires MinGW-w64 toolchain")
                    return False
            else:
                # Default Windows linking
                ld_cmd = ['ld', obj_file, '-o', output_executable]
                # Add platform-specific linker options
                ld_cmd.extend(self._get_linker_options())
            
            if verbose:
                print_info(f"Linker command: {' '.join(ld_cmd)}")
            
            result = subprocess.run(ld_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print_error(f"Linking failed:")
                print_error(result.stderr)
                return False
            
            print_success(f"Executable created: {output_executable}")
            
            # Step 5: Run the executable (if requested)
            if run_after_compile:
                # Check if we're cross-compiling to a different platform
                import platform as host_platform
                host_os = host_platform.system().lower()
                
                if self.target_os == 'windows' and host_os != 'windows':
                    # Check if Wine is available for running Windows executables
                    wine_cmd = shutil.which('wine') or shutil.which('wine64')
                    if wine_cmd:
                        print_system(f"[5/5] Running {output_executable} with Wine...")
                        print_info(f"Using Wine: {wine_cmd}")
                        print("-" * 40)
                        
                        try:
                            result = subprocess.run([wine_cmd, output_executable], 
                                                  capture_output=False, text=True)
                            print("-" * 40)
                            if result.returncode == 0:
                                print_success(f"Program exited with code: {result.returncode}")
                            else:
                                print_warning(f"Program exited with code: {result.returncode}")
                        except KeyboardInterrupt:
                            print_warning("Program interrupted by user")
                        except Exception as e:
                            print_error(f"Error running executable with Wine: {e}")
                            return False
                    else:
                        print_warning(f"[5/5] Cross-compiled for Windows, cannot run on {host_os}")
                        print_info(f"Windows executable created: {output_executable}")
                        print_info("Install Wine to run Windows executables: brew install --cask wine-stable")
                elif self.target_os != host_os:
                    print_warning(f"[5/5] Cross-compiled for {self.target_os}, cannot run on {host_os}")
                    print_info(f"Executable created: {output_executable}")
                    print_info(f"Transfer to {self.target_os} system to run.")
                else:
                    print_system(f"[5/5] Running {output_executable}...")
                    print("-" * 40)
                    
                    try:
                        result = subprocess.run([f'./{output_executable}'], 
                                              capture_output=False, text=True)
                        print("-" * 40)
                        if result.returncode == 0:
                            print_success(f"Program exited with code: {result.returncode}")
                        else:
                            print_warning(f"Program exited with code: {result.returncode}")
                    except KeyboardInterrupt:
                        print_warning("Program interrupted by user")
                    except Exception as e:
                        print_error(f"Error running executable: {e}")
                        return False
            else:
                print_success("Compilation complete")
            
            # Generate reports if requested
            if generate_reports:
                report_base = os.path.join(self.output_dir, input_path.stem)
                self.preprocessor.export_reports(report_base)
                print_info("Compilation reports generated")
            
            # Show compilation summary
            if verbose:
                print_info(f"Optimization level: {self.optimization_level}")
                print_info(f"Target platform: {self.target_os}")
                print_info("Compilation pipeline: C commands → High-level ASM → Pure assembly")
                stdlib_used = self.preprocessor.get_stdlib_used()
                if stdlib_used:
                    print_info(f"Standard library functions used: {', '.join(sorted(stdlib_used))}")
            
            return True
            
        except Exception as e:
            print_error(f"Compilation failed: {e}")
            return False
        
        finally:
            # Cleanup temporary files
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def run_executable(self, executable_name: str):
        """Run an existing executable"""
        if not os.path.exists(executable_name):
            print(f"Error: Executable '{executable_name}' not found", file=sys.stderr)
            return False
        
        print(f"Running {executable_name}...")
        print("-" * 40)
        try:
            result = subprocess.run([f'./{executable_name}'], capture_output=False, text=True)
            print("-" * 40)
            print(f"Program exited with code: {result.returncode}")
            return True
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
            return True
        except Exception as e:
            print(f"Error running executable: {e}", file=sys.stderr)
            return False
    
    def _get_object_format(self):
        """Get the appropriate object format for the current platform"""
        return self.preprocessor.codegen.platform.object_format
    
    def _get_linker_options(self):
        """Get platform-specific linker options"""
        platform = self.preprocessor.codegen.platform
        
        # Only Windows is supported
        if platform.os_name == 'windows':
            if platform.architecture == 'x86_64':
                return ['--entry=main']
            else:
                return ['--entry=main']
        return []


def main():
    if len(sys.argv) < 2:
        print(f"{Colors.BLUE}Usage:{Colors.RESET} python3 main.py <command> <input.asm> [options]")
        print()
        print("Advanced High-Level Assembly Language Preprocessor and Compiler")
        print("3-Stage Pipeline: C Commands → High-Level Assembly → Pure Assembly")
        print()
        print(f"{Colors.ORANGE}Commands:{Colors.RESET}")
        print("  p <input.asm> [output.asm]  - Preprocess only (default behavior)")
        print("  c <input.asm> [executable]  - Compile to executable only (don't run)")
        print("  r <input.asm> [executable]  - Compile and run .asm file")
        print("  convert <input.asm> <target> [output.asm] - Convert assembly to target architecture")
        print("  analyze <input.asm>         - Run code quality analysis only")
        print("  info                        - Show compiler information")
        print()
        print(f"{Colors.ORANGE}Options:{Colors.RESET}")
        print("  -t <target>     Target platform (os-arch or just os)")
        print("                  Examples: windows-x86_64, windows")
        print("  -a <arch>       Target architecture (x86_64, arm64, x86_32)")
        print("  -O <level>      Optimization level (O0, O1, O2, O3, Os)")
        print("  -v, --verbose   Verbose output")
        print()
        print(f"{Colors.ORANGE}Examples:{Colors.RESET}")
        print("  python3 main.py p program.asm              # Preprocess only (auto-detect)")
        print("  python3 main.py c program.asm -O3 -v       # Compile with max optimization")
        print("  python3 main.py r program.asm -t windows-x86_64  # Cross-compile for Windows x64")
        print("  python3 main.py r program.asm -t windows      # Cross-compile for Windows")
        print("  python3 main.py r program.asm -a arm64         # Target ARM64 (current OS)")
        print("  python3 main.py convert program.asm windows-x86_64 program_win.asm")
        print("  python3 main.py analyze program.asm        # Quality analysis only")
        print("  python3 main.py info --platform-info       # Show platform info")
        print()
        print("All output files are saved in the 'output' folder")
        sys.exit(1)
    
    # Parse command-line arguments
    args = sys.argv[1:]
    
    # Parse options
    target_os = None
    target_arch = None
    optimization_level = "O2"
    verbose = False
    enable_optimization = True
    enable_analysis = True
    generate_reports = False
    
    i = 0
    while i < len(args):
        if args[i] == '-t' and i + 1 < len(args):
            target_spec = args[i + 1]
            if '-' in target_spec:
                target_os, target_arch = target_spec.split('-', 1)
            else:
                target_os = target_spec
            args = args[:i] + args[i + 2:]
        elif args[i] == '-a' and i + 1 < len(args):
            target_arch = args[i + 1]
            args = args[:i] + args[i + 2:]
        elif args[i] in ['-O', '--optimize'] and i + 1 < len(args):
            optimization_level = args[i + 1]
            args = args[:i] + args[i + 2:]
        elif args[i] in ['-v', '--verbose']:
            verbose = True
            args = args[:i] + args[i + 1:]
        else:
            i += 1
    
    if len(args) < 1:
        print_error("Error: Command required")
        sys.exit(1)
    
    command = args[0].lower()
    
    # Handle special commands
    if command == 'info':
        print(f"{Colors.BLUE}HLASM Advanced Compiler Information{Colors.RESET}")
        print("=" * 50)
        print(f"Version: 2.0")
        print(f"Features: Optimization, Analysis, Cross-platform")
        print(f"Supported platforms: {', '.join(global_assembly_converter.get_available_platforms())}")
        print(f"Standard library functions: {len(global_stdlib.functions)}")
        print(f"Categories: {', '.join(global_stdlib.get_all_categories())}")
        return
    
    # Handle legacy usage (no command specified)
    if command.endswith('.asm'):
        # Legacy mode: python3 main.py input.asm [output.asm]
        input_file = args[0]
        if len(args) > 1:
            output_file = args[1]
            if not os.path.dirname(output_file):
                output_file = os.path.join("output", output_file)
        else:
            basename = os.path.basename(input_file).replace('.asm', '_generated.asm')
            output_file = os.path.join("output", basename)
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        preprocessor = HLASMPreprocessor(target_os, target_arch, optimization_level)
        output = preprocessor.process_file(input_file, enable_optimization, enable_analysis)
        
        try:
            with open(output_file, 'w') as f:
                f.write(output)
            print_success(f"Successfully preprocessed: {input_file} -> {output_file}")
            
            if verbose:
                print_info(f"Optimization level: {optimization_level}")
                stdlib_used = preprocessor.get_stdlib_used()
                if stdlib_used:
                    print_info(f"Standard library functions included: {', '.join(sorted(stdlib_used))}")
                print_info("Pipeline: C commands → High-level ASM → Pure assembly")
                
                if enable_analysis:
                    print("\n" + preprocessor.get_quality_report())
            
            if generate_reports:
                report_base = os.path.join("output", Path(input_file).stem)
                preprocessor.export_reports(report_base)
                
        except IOError as e:
            print_error(f"Error writing output file: {e}")
            sys.exit(1)
        return
    
    # New command-based interface
    if command == 'p':  # Preprocess only
        if len(args) < 2:
            print_error("Error: 'p' command requires input file")
            sys.exit(1)
        
        input_file = args[1]
        
        # Create the preprocessor and process the file
        # The process_file method already saves the final formatted file
        preprocessor = HLASMPreprocessor(target_os, target_arch, optimization_level)
        final_output = preprocessor.process_file(input_file, enable_optimization, enable_analysis)
        
        if verbose:
            print_info(f"Optimization level: {optimization_level}")
            stdlib_used = preprocessor.get_stdlib_used()
            if stdlib_used:
                print_info(f"Standard library functions included: {', '.join(sorted(stdlib_used))}")
            print_info("Pipeline: C commands → High-level ASM → Pure assembly")
    
    elif command == 'c':  # Compile only (don't run)
        if len(args) < 2:
            print_error("Error: 'c' command requires input file")
            sys.exit(1)
        
        input_file = args[1]
        output_executable = args[2] if len(args) > 2 else Path(input_file).stem
        
        compiler = HLASMCompiler(target_os, target_arch, optimization_level, enable_analysis)
        success = compiler.compile_file(input_file, output_executable, 
                                       run_after_compile=False, 
                                       generate_reports=generate_reports,
                                       verbose=verbose)
        if not success:
            sys.exit(1)
    
    elif command == 'r':  # Compile and run .asm file
        if len(args) < 2:
            print_error("Error: 'r' command requires input .asm file")
            sys.exit(1)
        
        input_file = args[1]
        if not input_file.endswith('.asm'):
            print_error("Error: 'r' command requires a .asm file")
            sys.exit(1)
        
        output_executable = args[2] if len(args) > 2 else Path(input_file).stem
        
        print_info(f"Compiling and running {input_file}...")
        compiler = HLASMCompiler(target_os, target_arch, optimization_level, enable_analysis)
        success = compiler.compile_file(input_file, output_executable, 
                                       run_after_compile=True,
                                       generate_reports=generate_reports,
                                       verbose=verbose)
        if not success:
            sys.exit(1)
    
    elif command == 'analyze':  # Code quality analysis only
        if len(args) < 2:
            print_error("Error: 'analyze' command requires input file")
            sys.exit(1)
        
        input_file = args[1]
        
        print_info(f"Analyzing {input_file}...")
        
        try:
            with open(input_file, 'r') as f:
                code_lines = f.readlines()
            
            quality_engine = CodeQualityEngine()
            quality_engine.analyze_code(code_lines)
            
            print(quality_engine.generate_quality_report())
            
            if generate_reports:
                report_base = os.path.join("output", Path(input_file).stem)
                os.makedirs("output", exist_ok=True)
                
                with open(f"{report_base}_analysis.txt", 'w') as f:
                    f.write(quality_engine.generate_quality_report())
                
                with open(f"{report_base}_analysis.json", 'w') as f:
                    json.dump(quality_engine.export_results_json(), f, indent=2)
                
                print_success("Analysis reports generated")
        
        except Exception as e:
            print_error(f"Analysis failed: {e}")
            sys.exit(1)
    
    elif command == 'convert':  # Convert assembly to target architecture
        if len(args) < 3:
            print_error("Error: 'convert' command requires input file and target platform")
            print_error("Usage: python3 main.py convert <input.asm> <target-platform> [output.asm]")
            print_error("Example: python3 main.py convert program.asm windows-x86_64 program_win.asm")
            sys.exit(1)
        
        input_file = args[1]
        target_platform = args[2]
        output_file = args[3] if len(args) > 3 else f"{Path(input_file).stem}_{target_platform.replace('-', '_')}.asm"
        
        print_info(f"Converting {input_file} to {target_platform}...")
        
        try:
            # Check if target platform is supported
            if target_platform not in global_assembly_converter.get_available_platforms():
                print_error(f"Error: Unsupported target platform '{target_platform}'")
                print_error(f"Supported platforms: {', '.join(global_assembly_converter.get_available_platforms())}")
                sys.exit(1)
            
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            output_path = os.path.join("output", output_file)
            
            # Perform conversion
            success = global_assembly_converter.convert_file(input_file, output_path, target_platform)
            
            if success:
                print_success(f"Assembly converted successfully to {output_path}")
                
                # Generate conversion report if verbose
                if verbose:
                    with open(input_file, 'r') as f:
                        original_code = f.read()
                    
                    report = global_assembly_converter.get_conversion_report(original_code, target_platform)
                    print("\n" + "=" * 50)
                    print("CONVERSION REPORT")
                    print("=" * 50)
                    print(report)
                    
                    # Save report to file
                    report_path = os.path.join("output", f"{Path(input_file).stem}_conversion_report.txt")
                    with open(report_path, 'w') as f:
                        f.write(report)
                    print_info(f"Conversion report saved to {report_path}")
            else:
                print_error("Conversion failed")
                sys.exit(1)
                
        except Exception as e:
            print_error(f"Conversion failed: {e}")
            sys.exit(1)
    
    else:
        print_error(f"Error: Unknown command '{command}'")
        print_error("Valid commands: p (preprocess), c (compile), r (compile and run), convert, analyze, info")
        sys.exit(1)


if __name__ == "__main__":
    main()