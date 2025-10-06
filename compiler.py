#!/usr/bin/env python3
"""
Complete Compiler for High-Level Assembly Language
Preprocesses, assembles with NASM, links, and runs the executable
"""

import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from main import HLASMPreprocessor


class HLASMCompiler:
    def __init__(self):
        self.preprocessor = HLASMPreprocessor()
        self.temp_dir = None
        self.cleanup_files = []
    
    def check_dependencies(self):
        """Check if required tools are available"""
        import platform
        system = platform.system().lower()
        # Use standard tools for all platforms
        return {
            'assembler': 'nasm',
            'linker': 'ld',
            'objcopy': 'objcopy',
            'objdump': 'objdump'
        }
        
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            print(f"Error: Missing required tools: {', '.join(missing_tools)}", file=sys.stderr)
            print("Please install:", file=sys.stderr)
            for tool in missing_tools:
                if tool == 'nasm':
                    print("  - NASM assembler (brew install nasm)", file=sys.stderr)
                elif tool in ['as', 'ld']:
                    print("  - Xcode Command Line Tools (xcode-select --install)", file=sys.stderr)
            return False
        return True
    
    def compile_and_run(self, input_file: str, output_executable: str = None, 
                       keep_intermediate: bool = False, run_after_compile: bool = True,
                       compile_args: list = None, link_args: list = None):
        """Complete compilation pipeline"""
        
        if not self.check_dependencies():
            return False
        
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
            return False
        
        # Set default output executable name
        if output_executable is None:
            output_executable = input_path.stem
        
        try:
            # Create temporary directory for intermediate files
            self.temp_dir = tempfile.mkdtemp(prefix='hlasm_compile_')
            
            # Step 1: Preprocess high-level assembly to standard assembly
            print(f"[1/4] Preprocessing {input_file}...")
            asm_output = self.preprocessor.process_file(input_file)
            
            # Write preprocessed assembly to temporary file
            asm_file = os.path.join(self.temp_dir, 'output.asm')
            with open(asm_file, 'w') as f:
                f.write(asm_output)
            
            if keep_intermediate:
                # Copy to current directory for inspection
                intermediate_asm = f"{input_path.stem}_preprocessed.asm"
                shutil.copy2(asm_file, intermediate_asm)
                print(f"    Preprocessed assembly saved as: {intermediate_asm}")
            
            # Step 2: Assemble
            print("[2/4] Assembling...")
            obj_file = os.path.join(self.temp_dir, 'output.o')
            
            # Determine which assembler to use
            import platform
            system = platform.system().lower()
            arch = platform.machine().lower()
            
            # Use NASM for all platforms
            asm_cmd = ['nasm', '-f', self._get_object_format(), asm_file, '-o', obj_file]
            if compile_args:
                asm_cmd.extend(compile_args)
            
            result = subprocess.run(asm_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Assembly failed:", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return False
            
            if keep_intermediate:
                # Copy object file to current directory
                intermediate_obj = f"{input_path.stem}.o"
                shutil.copy2(obj_file, intermediate_obj)
                print(f"    Object file saved as: {intermediate_obj}")
                self.cleanup_files.append(intermediate_obj)
            
            # Step 3: Link
            print("[3/4] Linking...")
            
            # Standard linking for all platforms
            ld_cmd = ['ld', obj_file, '-o', output_executable]
            
            if link_args:
                ld_cmd.extend(link_args)
            
            result = subprocess.run(ld_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Linking failed:", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return False
            
            print(f"    Executable created: {output_executable}")
            
            # Step 4: Run the executable (if requested)
            if run_after_compile:
                print(f"[4/4] Running {output_executable}...")
                print("-" * 40)
                
                try:
                    result = subprocess.run([f'./{output_executable}'], 
                                          capture_output=False, text=True)
                    print("-" * 40)
                    print(f"Program exited with code: {result.returncode}")
                except KeyboardInterrupt:
                    print("\nProgram interrupted by user")
                except Exception as e:
                    print(f"Error running executable: {e}", file=sys.stderr)
                    return False
            else:
                print("[4/4] Compilation complete (skipping execution)")
            
            # Show standard library usage
            if self.preprocessor.get_stdlib_used():
                print(f"Standard library functions used: {', '.join(sorted(self.preprocessor.get_stdlib_used()))}")
            
            return True
            
        except Exception as e:
            print(f"Compilation failed: {e}", file=sys.stderr)
            return False
        
        finally:
            # Cleanup temporary files
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _get_object_format(self):
        """Get the appropriate object format for the current platform"""
        import platform
        system = platform.system().lower()
        
        if system == 'windows':
            return 'win64'
        else:
            # Default to win64 (Windows only support)
            return 'win64'
    
    def clean(self, executable_name: str = None):
        """Clean up generated files"""
        if executable_name and os.path.exists(executable_name):
            os.remove(executable_name)
            print(f"Removed: {executable_name}")
        
        for file_path in self.cleanup_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")
        
        self.cleanup_files.clear()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compiler.py <input.asm> [options]", file=sys.stderr)
        print("\nHigh-Level Assembly Language Compiler")
        print("Preprocesses, assembles, links, and runs HLASM programs")
        print("\nOptions:")
        print("  -o <name>        Output executable name (default: input filename without extension)")
        print("  -k, --keep       Keep intermediate files (.asm, .o)")
        print("  -c, --compile    Compile only, don't run")
        print("  -r, --run        Run existing executable without recompiling")
        print("  --clean <name>   Remove generated files")
        print("  --nasm-args      Additional arguments for NASM (comma-separated)")
        print("  --ld-args        Additional arguments for linker (comma-separated)")
        print("\nExamples:")
        print("  python3 compiler.py program.asm")
        print("  python3 compiler.py program.asm -o myprogram -k")
        print("  python3 compiler.py program.asm -c")
        print("  python3 compiler.py --clean myprogram")
        print("  python3 compiler.py program.asm --nasm-args='-g,-F dwarf'")
        sys.exit(1)
    
    compiler = HLASMCompiler()
    
    # Parse command line arguments
    input_file = sys.argv[1]
    output_executable = None
    keep_intermediate = False
    run_after_compile = True
    compile_only = False
    run_only = False
    clean_target = None
    nasm_args = []
    ld_args = []
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ['-o', '--output']:
            if i + 1 < len(sys.argv):
                output_executable = sys.argv[i + 1]
                i += 1
            else:
                print("Error: -o requires an argument", file=sys.stderr)
                sys.exit(1)
        
        elif arg in ['-k', '--keep']:
            keep_intermediate = True
        
        elif arg in ['-c', '--compile']:
            compile_only = True
            run_after_compile = False
        
        elif arg in ['-r', '--run']:
            run_only = True
        
        elif arg == '--clean':
            if i + 1 < len(sys.argv):
                clean_target = sys.argv[i + 1]
                i += 1
            else:
                print("Error: --clean requires an argument", file=sys.stderr)
                sys.exit(1)
        
        elif arg.startswith('--nasm-args='):
            args_str = arg.split('=', 1)[1]
            nasm_args = [arg.strip() for arg in args_str.split(',')]
        
        elif arg.startswith('--ld-args='):
            args_str = arg.split('=', 1)[1]
            ld_args = [arg.strip() for arg in args_str.split(',')]
        
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            sys.exit(1)
        
        i += 1
    
    # Handle clean command
    if clean_target:
        compiler.clean(clean_target)
        return
    
    # Handle run-only command
    if run_only:
        executable = output_executable or Path(input_file).stem
        if not os.path.exists(executable):
            print(f"Error: Executable '{executable}' not found", file=sys.stderr)
            sys.exit(1)
        
        print(f"Running {executable}...")
        print("-" * 40)
        try:
            result = subprocess.run([f'./{executable}'], capture_output=False, text=True)
            print("-" * 40)
            print(f"Program exited with code: {result.returncode}")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"Error running executable: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Compile and optionally run
    success = compiler.compile_and_run(
        input_file=input_file,
        output_executable=output_executable,
        keep_intermediate=keep_intermediate,
        run_after_compile=run_after_compile,
        compile_args=nasm_args,
        link_args=ld_args
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()