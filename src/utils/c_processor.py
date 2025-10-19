#!/usr/bin/env python3
import re
import tempfile
import os
import subprocess
from typing import List, Dict, Tuple, Optional
from ..utils.colors import print_info, print_warning, print_error, print_success

class CCodeProcessor:
    """Process embedded C code in CASM files"""
    
    def __init__(self):
        self.c_functions = []
        self.c_includes = []
        self.c_globals = []
        self.temp_dir = None
        self.casm_variables = {}  # Track CASM variables
        self.c_variables = {}     # Track C variables
        # Track extern headers as tuples: (header_name, use_angle)
        # use_angle=True -> #include <header>
        # use_angle=False -> #include "header"
        self.headers = []         # List[Tuple[str, bool]]
        self.c_code_blocks = []   # Collect all C code blocks with markers
        self._raw_c_blocks = []   # Store raw C code for each block (parallel to c_code_blocks)
        self.assembly_segments = {} # Store extracted assembly segments by marker ID
        self.marker_counter = 0   # Counter for generating unique block markers
        # When True, persist combined C and assembly files to the project's
        # output directory for debugging; otherwise don't persist.
        # Default to False so casual runs are not noisy; top-level CLI can
        # enable this via --debug-save when the user requests full logs.
        self.save_debug = False
        # Optional user-provided flags (set by top-level CLI). If provided,
        # these override or supplement auto-detected pkg-config flags.
        # user_cflags: list of compiler flags (e.g. ['-I/opt/include', '-DDEBUG'])
        # user_ldflags: list of linker flags (e.g. ['-L/mingw/lib', '-lSDL2'])
        self.user_cflags = None
        self.user_ldflags = None
        # Store the last compile command output (stdout+stderr) so callers can
        # present a concise preview while allowing full logs to be saved.
        self._last_compile_output = ''
        # Short status message for the last compile attempt (single-line)
        self._last_status = ''
        # Target platform/arch for compilation (affects compiler selection)
        # platform: 'linux'|'windows'|'macos'
        # arch: 'x86_64'|'x86'|'arm64'
        self.target_platform = 'linux'
        self.target_arch = 'x86_64'
        
    def add_header(self, header_name: str, use_angle: bool = False):
        """Add header file for C compilation.

        header_name: bare header (e.g. math.h or mylib.h or windows.h)
        use_angle: when True emit #include <header_name>, otherwise #include "header_name"
        """
        # Normalize header name: accept inputs like '<SDL2/SDL.h>' or 'SDL2/SDL.h'
        raw = header_name.strip()
        # If the caller provided angle brackets around the header, prefer that
        if raw.startswith('<') and raw.endswith('>'):
            raw = raw[1:-1].strip()
            use_angle = True
        # If caller provided quoted include like "mylib.h", strip quotes and respect use_angle flag
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1].strip()
            # leave use_angle as provided (likely False)

        # Avoid duplicates by header name
        for (h, ua) in self.headers:
            if h == raw:
                # already present; keep existing use_angle value
                return

        self.headers.append((raw, use_angle))
        print_info(f"Added header: {raw} (use_angle={use_angle})")
        
    def set_casm_variables(self, variables: dict):
        """Set CASM variables for C code access"""
        self.casm_variables = variables
        
    def get_c_variables(self) -> dict:
        """Get C variables for CASM access"""
        return self.c_variables
        
    def process_c_code(self, content: str) -> str:
        """Process CASM content with embedded C code - simplified to just collect variables"""
        lines = content.split('\n')
        result_lines = []
        i = 0
        
        # First pass: collect CASM variables
        self._collect_casm_variables(lines)
        
        # Don't process C code here - leave it for code generation
        return content
    
    def _extract_c_block(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract a C code block starting from %!"""
        c_lines = []
        i = start_idx + 1  # Skip the %! line
        
        # Look for end marker or end of file
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for end of C block (empty line or next CASM directive)
            if (not line or 
                line.startswith('%') and not line.startswith('%!') or
                line.startswith(';')):
                break
                
            c_lines.append(lines[i])
            i += 1
        
        return '\n'.join(c_lines), i
    
    def _collect_casm_variables(self, lines: List[str]):
        """Collect CASM variable declarations"""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('%var '):
                parts = stripped.split(None, 2)
                if len(parts) >= 3:
                    var_name = parts[1]
                    var_value = parts[2]
                    # Try to determine type from value
                    var_type = self._infer_variable_type(var_value)
                    # Use placeholder label; real mapping will be provided by codegen
                    self.casm_variables[var_name] = {
                        'type': var_type,
                        'value': var_value,
                        'label': var_name
                    }
                    print_info(f"Found CASM variable: {var_name} ({var_type}) = {var_value}")
    
    def _infer_variable_type(self, value: str) -> str:
        """Infer C type from CASM variable value"""
        value = value.strip()
        
        # String literal
        if value.startswith('"') and value.endswith('"'):
            return 'char*'
        
        # Try to parse as integer
        try:
            int(value)
            return 'int'
        except ValueError:
            pass
        
        # Try to parse as float
        try:
            float(value)
            return 'double'
        except ValueError:
            pass
        
        # Default to int for expressions
        return 'int'
    
    def _process_variable_references(self, c_code: str) -> str:
        """Process variable references between CASM and C"""
        processed_code = c_code
        
        # Add external declarations for CASM variables
        external_decls = []
        for var_name, var_info in self.casm_variables.items():
            if isinstance(var_info, dict):
                var_type = var_info['type']
                label = var_info['label']
            else:
                # Handle case where var_info is a string (label)
                var_type = 'int'  # Default type
                label = var_info
            
            external_decls.append(f"extern {var_type} {label};  // CASM variable: {var_name}")
            
            # Replace $var_name with actual variable reference
            processed_code = processed_code.replace(f'${var_name}', label)
        
        if external_decls:
            processed_code = '\n'.join(external_decls) + '\n\n' + processed_code
        
        # Extract C variable declarations for CASM access
        self._extract_c_variables(processed_code)
        
        return processed_code

    def _get_var_label(self, name: str) -> str:
        """Return the assembler label for a CASM variable name.

        If the variable was discovered in self.casm_variables, return its
        configured 'label'. Otherwise fall back to the predictable
        'var_<name>' form.
        """
        if name in self.casm_variables:
            info = self.casm_variables[name]
            if isinstance(info, dict) and 'label' in info:
                return info['label']
            return info
        return f'var_{name}'
    
    def _extract_c_variables(self, c_code: str):
        """Extract C variable declarations for CASM access"""
        # Simple regex to find global variable declarations
        import re
        
        # Match patterns like: int global_var = 42;
        var_pattern = r'(int|float|double|char)\s+(\w+)\s*=\s*([^;]+);'
        matches = re.finditer(var_pattern, c_code)
        
        for match in matches:
            var_type, var_name, var_value = match.groups()
            self.c_variables[var_name] = {
                'type': var_type,
                'value': var_value.strip(),
                'c_name': var_name
            }
            print_info(f"Found C variable: {var_name} ({var_type}) = {var_value.strip()}")
    
    def _compile_c_to_assembly(self, c_code: str) -> str:
        """Compile C code to Intel format assembly using GCC"""
        print_info(f"Compiling C code: {repr(c_code)}")
        try:
            # Create temporary directory
            if not self.temp_dir:
                self.temp_dir = tempfile.mkdtemp(prefix='casm_c_')
            
            # Write C code to temporary file
            c_file = os.path.join(self.temp_dir, 'temp.c')
            with open(c_file, 'w') as f:
                # Add standard C headers
                f.write('#include <stdio.h>\n')
                f.write('#include <stdlib.h>\n')
                f.write('#include <string.h>\n')
                
                # Add extern headers. self.headers stores entries as tuples
                # (header_name, use_angle) where use_angle=True emits
                # #include <header_name> and use_angle=False emits
                # #include "header_name". Handle legacy string entries
                # defensively as well.
                for header_entry in self.headers:
                    if isinstance(header_entry, tuple) and len(header_entry) >= 2:
                        header_name, use_angle = header_entry[0], bool(header_entry[1])
                    else:
                        # Legacy single-string entry: try to infer behavior and strip brackets
                        header_name = str(header_entry).strip()
                        use_angle = header_name.startswith('<') and header_name.endswith('>')
                        header_name = header_name.replace('<', '').replace('>', '').replace('"', '').replace("'", '').strip()

                    # Emit include using angle brackets when requested, otherwise use quotes
                    if use_angle:
                        f.write(f'#include <{header_name}>\n')
                    else:
                        f.write(f'#include "{header_name}"\n')
                
                f.write('\n')
                
                # Add CASM variable declarations as externals
                for var_name, var_info in self.casm_variables.items():
                    if isinstance(var_info, dict):
                        var_type = var_info['type']
                        label = var_info['label']
                    else:
                        var_type = 'int'  # Default type
                        label = var_info
                    
                    # If this CASM variable is a string/buffer, declare it as an
                    # array of char (extern char NAME[]) so C will treat the
                    # symbol as the data label itself. If we declare it as
                    # 'char *' GCC will assume the symbol is a pointer value
                    # stored in memory and will generate an extra indirection
                    # (causing invalid pointer dereferences at runtime).
                    c_decl = None
                    vt = var_type.lower() if isinstance(var_type, str) else ''
                    # If a size is known, declare as an array with that size so
                    # `sizeof(name)` in C works. Otherwise declare as flexible
                    # array `extern char name[];` which only provides the symbol
                    # address but is an incomplete type.
                    size = None
                    if isinstance(self.casm_variables.get(var_name), dict):
                        size = self.casm_variables[var_name].get('size')

                    if 'char' in vt or vt in ('str', 'string', 'buffer'):
                        if size:
                            c_decl = f'extern char {label}[{size}];'
                        else:
                            c_decl = f'extern char {label}[];'
                    else:
                        c_decl = f'extern {var_type} {label};'

                    f.write(c_decl + "\n")
                    # Allow C code to use CASM variable names (both `name` and `var_name`)
                    f.write(f'#define {var_name} {label}\n')
                    f.write(f'#define var_{var_name} {label}\n')
                
                f.write('\n')
                
                # Create a function wrapper for the C code
                f.write('void casm_c_block() {\n')
                f.write(f'    {c_code}\n')
                f.write('}\n')
            
            # Debug: optionally print the generated C file content
            if self.save_debug:
                with open(c_file, 'r') as f:
                    c_content = f.read()
                print_info(f"Generated C file content:\n{c_content}")
            
            asm_file = os.path.join(self.temp_dir, 'temp.s')

            # Choose compilers based on target platform/arch
            compilers_to_try = []

            # If targeting Windows prefer mingw cross-compiler
            if self.target_platform == 'windows':
                # Try native cross compiler first
                compilers_to_try.append((['x86_64-w64-mingw32-gcc', '-S', '-O0', '-masm=intel'] + extra_flags + [c_file, '-o', asm_file]))
                # Try i686 mingw for 32-bit
                compilers_to_try.append((['i686-w64-mingw32-gcc', '-S', '-O0', '-masm=intel'] + extra_flags + [c_file, '-o', asm_file]))

            # Generic host compilers
            host_flags = ['-S', '-O0', '-masm=intel']
            # If targeting 32-bit x86, add -m32
            if self.target_arch in ('x86', 'i386', 'i686'):
                host_flags = ['-S', '-O0', '-m32', '-masm=intel']

            compilers_to_try.append((['gcc'] + host_flags + extra_flags + [c_file, '-o', asm_file]))
            compilers_to_try.append((['clang'] + host_flags + extra_flags + [c_file, '-o', asm_file]))

            for cmd in compilers_to_try:
                try:
                    # Run the compiler and capture stdout/stderr for callers
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.temp_dir)
                    combined = (result.stdout or '') + '\n' + (result.stderr or '')
                    self._last_compile_output = combined.strip()

                    if result.returncode == 0:
                        # Read generated assembly
                        with open(asm_file, 'r') as f:
                            assembly = f.read()

                        embedded_assembly = f"; === RAW {os.path.basename(cmd[0]).upper()} ASSEMBLY OUTPUT ===\n; {' '.join(cmd)}\n"
                        embedded_assembly += assembly
                        embedded_assembly += f"; === END {os.path.basename(cmd[0]).upper()} ASSEMBLY OUTPUT ==="

                        if self.save_debug:
                            print_success(f"C code compiled with {os.path.basename(cmd[0])} (Intel syntax) - raw output embedded")
                        return embedded_assembly
                    else:
                        # Only emit a concise single-line error unless save_debug
                        if self.save_debug:
                            print_error(f"Compiler {cmd[0]} failed: {result.stderr}")
                        else:
                            print_error(f"Compiler {cmd[0]} failed: see combined_c output; use --debug-save for full logs")
                except FileNotFoundError:
                    print_warning(f"Compiler {cmd[0]} not found, trying next")
                    continue
            
            # If no compiler worked, fall back to simple processing
            print_warning("No suitable C compiler found, using simple processing")
            return self._process_simple_c_statements(c_code)
                
        except Exception as e:
            print_error(f"C compilation failed: {e}")
            return f"; C compilation error: {e}"
    
    def _clean_to_intel_nasm(self, assembly: str) -> str:
        """Clean assembly to Intel NASM format"""
        lines = assembly.split('\n')
        cleaned_lines = []
        in_function = False
        current_function = None
        
        for line in lines:
            original_line = line
            stripped = line.strip()
            
            # Skip GCC/Clang metadata and directives
            skip_patterns = [
                '.file', '.intel_syntax', '.section', '.def', '.scl', '.type', 
                '.size', '.ident', '.globl', '.p2align', '.cfi_', '.build_version', 
                'sdk_version', '.long', '.quad', '@', 'LBB', 'Ltmp', '_GLOBAL_', 
                '.subsections_via_symbols', '.macosx_version_min', '.addrsig',
                '.addrsig_sym', '%bb.'
            ]
            
            if any(stripped.startswith(pattern) for pattern in skip_patterns):
                continue
            
            # Skip platform-specific markers
            if any(marker in stripped for marker in ['#', 'LBB', '_GLOBAL_', 'Ltmp']):
                continue
            
            # Handle function labels
            if stripped.endswith(':') and not any(skip in stripped for skip in skip_patterns):
                func_name = stripped[:-1]
                # Remove leading underscore (common in some platforms)
                if func_name.startswith('_'):
                    func_name = func_name[1:]
                
                # Check if it's a real function (not a local label)
                if not func_name.startswith('.') and len(func_name) > 1:
                    cleaned_lines.append(f"{func_name}:")
                    in_function = True
                    current_function = func_name
                    print_info(f"Found C function: {func_name}")
                continue
            
            # Handle instructions - convert to proper Intel format
            if stripped and not stripped.startswith(';'):
                # Basic instruction processing
                instruction_parts = stripped.split()
                if len(instruction_parts) > 0:
                    instr = instruction_parts[0].lower()
                    
                    # Common x86-64 instructions
                    if instr in ['mov', 'add', 'sub', 'mul', 'div', 'imul', 'idiv',
                                'cmp', 'test', 'and', 'or', 'xor', 'not', 'neg',
                                'push', 'pop', 'call', 'ret', 'jmp', 'je', 'jne',
                                'jz', 'jnz', 'jl', 'jle', 'jg', 'jge', 'ja', 'jae',
                                'jb', 'jbe', 'inc', 'dec', 'lea', 'nop']:
                        
                        # Clean up operands
                        if len(instruction_parts) > 1:
                            operands = ' '.join(instruction_parts[1:])
                            # Remove platform-specific syntax
                            operands = operands.replace('%', '').replace('$', '')

                            # Handle register names (remove % prefix if present)
                            operands = self._clean_intel_operands(operands)

                            # Surround each instruction with NOP markers so the C
                            # block is clearly delimited in the generated asm.
                            cleaned_lines.append(f"    nop  ; start_c_instr {current_function if current_function else ''}")
                            cleaned_lines.append(f"    {instr} {operands}")
                            cleaned_lines.append(f"    nop  ; end_c_instr {current_function if current_function else ''}")
                        else:
                            cleaned_lines.append(f"    nop  ; start_c_instr {current_function if current_function else ''}")
                            cleaned_lines.append(f"    {instr}")
                            cleaned_lines.append(f"    nop  ; end_c_instr {current_function if current_function else ''}")
                        continue
            
            # If it's a simple line that might be useful, keep it
            if stripped and len(stripped) < 100 and not any(bad in stripped for bad in ['#', '@', '%bb']):
                cleaned_lines.append(f"    {stripped}")
        
        # If we got very little output, provide a more informative placeholder
        if len(cleaned_lines) < 3:
            placeholder = [
                "; C code processed but assembly output was minimal",
                "; Functions available as external symbols:",
            ]
            
            # Add function names we found
            for var_name, var_info in self.c_variables.items():
                placeholder.append(f"; extern {var_info['type']} {var_name}")
            
            return '\n'.join(placeholder)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_intel_operands(self, operands: str) -> str:
        """Clean operands for Intel NASM format"""
        # Remove common AT&T syntax elements
        operands = operands.replace('%', '')  # Remove register % prefix
        operands = operands.replace('$', '')  # Remove immediate $ prefix
        
        # Handle memory references
        operands = operands.replace('(', '[').replace(')', ']')
        
        # Common register name fixes
        register_map = {
            'eax': 'eax', 'ebx': 'ebx', 'ecx': 'ecx', 'edx': 'edx',
            'rax': 'rax', 'rbx': 'rbx', 'rcx': 'rcx', 'rdx': 'rdx',
            'rsi': 'rsi', 'rdi': 'rdi', 'rbp': 'rbp', 'rsp': 'rsp',
            'r8': 'r8', 'r9': 'r9', 'r10': 'r10', 'r11': 'r11',
            'r12': 'r12', 'r13': 'r13', 'r14': 'r14', 'r15': 'r15'
        }
        
        for old_reg, new_reg in register_map.items():
            operands = operands.replace(old_reg, new_reg)
        
        return operands
    
    def _is_simple_c_code(self, c_code: str) -> bool:
        """Check if C code can be processed as simple line-by-line statements"""
        lines = [line.strip() for line in c_code.split('\n') if line.strip()]
        
        # Check for complex constructs
        for line in lines:
            # Skip includes and simple declarations
            if (line.startswith('#include') or 
                line.startswith('extern') or
                line.startswith('//') or
                not line):
                continue
            
            # Check for functions, loops, complex statements
            if any(keyword in line for keyword in [
                'int main', 'void ', '(', ')', '{', '}', 'for', 'while', 'if'
            ]):
                return False
        
        return True
    
    def _process_simple_c_statements(self, c_code: str) -> str:
        """Process simple C statements line by line"""
        lines = [line.strip() for line in c_code.split('\n') if line.strip()]
        asm_lines = []
        
        for line in lines:
            # Skip includes and comments
            if (line.startswith('#include') or 
                line.startswith('extern') or
                line.startswith('//') or
                not line):
                continue
            
            # Skip complex C constructs - just add as comments
            if any(keyword in line for keyword in ['for(', 'if(', 'else', 'while(', '{', '}']):
                asm_lines.append(f"    ; C construct (not implemented): {line}")
                continue
            
            # Simple assignment: int x = 5; or x = y;
            if '=' in line and line.endswith(';'):
                asm_lines.extend(self._convert_assignment(line))
            
            # Function call: printf("hello");
            elif line.endswith(';') and '(' in line and ')' in line:
                asm_lines.extend(self._convert_function_call(line))
            
            # Variable declaration without assignment: int x;
            elif any(type_name in line for type_name in ['int ', 'char ', 'float ', 'double ']):
                asm_lines.extend(self._convert_declaration(line))
            
            else:
                # Unknown statement, add as comment
                asm_lines.append(f"    ; C statement: {line}")
        
        return '\n'.join(asm_lines)
    
    def _convert_assignment(self, line: str) -> List[str]:
        """Convert simple C assignment to assembly"""
        line = line.rstrip(';').strip()
        
        if '=' in line:
            left, right = line.split('=', 1)
            left = left.strip()
            right = right.strip()
            
            # Remove type declaration if present
            if any(type_name in left for type_name in ['int ', 'char ', 'float ', 'double ']):
                parts = left.split()
                left = parts[-1]  # Get variable name
            
            # Handle simple assignments only
            try:
                # Try to parse as number
                value = int(right)
                return [
                    f"    ; {line}",
                    f"    mov eax, {value}",
                        f"    mov [" + self._get_var_label(left) + "], eax"
                ]
            except ValueError:
                # Check if it's a simple variable reference
                if right.isidentifier():
                    return [
                        f"    ; {line}",
                        f"    mov eax, [" + self._get_var_label(right) + "]",
                        f"    mov [" + self._get_var_label(left) + "], eax"
                    ]
                else:
                    # Complex expression - generate comment only for now
                    return [
                        f"    ; {line}",
                        f"    ; TODO: Complex expression: {right}",
                        f"    ; mov [var_{left}], eax  ; Result would go here"
                    ]
        
        return [f"    ; Assignment: {line}"]
    
    def _convert_function_call(self, line: str) -> List[str]:
        """Convert simple C function call to assembly"""
        line = line.rstrip(';').strip()
        
        # Extract function name and arguments
        if '(' in line and ')' in line:
            func_name = line[:line.index('(')]
            args_part = line[line.index('(')+1:line.rindex(')')]
            
            asm_lines = [f"    ; {line}"]
            
            # Handle printf specifically
            if func_name.strip() == 'printf':
                # For now, generate a simplified printf call
                # This would need proper argument parsing for full implementation
                if '"' in args_part:
                    # Extract format string
                    format_start = args_part.find('"')
                    format_end = args_part.find('"', format_start + 1)
                    if format_end != -1:
                        format_str = args_part[format_start:format_end+1]
                        # Generate a string label for the format
                        str_label = f"printf_fmt_{hash(format_str) & 0xFFFF:04x}"
                        
                        asm_lines.extend([
                            f"    ; Format string: {format_str}",
                            f"    lea rcx, [rel {str_label}]  ; Load format string",
                            f"    ; TODO: Load printf arguments",
                            f"    call printf"
                        ])
                    else:
                        asm_lines.append("    ; printf call - format string parsing failed")
                else:
                    asm_lines.append("    ; printf call - no format string found")
            else:
                # Generic function call - emit unified call directive so the
                # asm_transpiler can arrange arguments and platform specifics.
                # Try to parse simple comma-separated args
                args = self._parse_printf_args(args_part) if args_part else []
                args_str = ' '.join(a for a in args if a)
                if args_str:
                    asm_lines.extend([
                        f"    ; {line}",
                        f"UCALL {func_name} {args_str}"
                    ])
                else:
                    asm_lines.extend([
                        f"    ; {line}",
                        f"UCALL {func_name}"
                    ])
            
            return asm_lines
        
        return [f"    ; Function call: {line}"]
    
    def _parse_printf_args(self, args_str: str) -> List[str]:
        """Parse printf arguments, handling quoted strings"""
        args = []
        current_arg = ""
        in_quotes = False
        paren_depth = 0
        
        for char in args_str:
            if char == '"' and (not current_arg or current_arg[-1] != '\\'):
                in_quotes = not in_quotes
                current_arg += char
            elif char == '(' and not in_quotes:
                paren_depth += 1
                current_arg += char
            elif char == ')' and not in_quotes:
                paren_depth -= 1
                current_arg += char
            elif char == ',' and not in_quotes and paren_depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    def _convert_declaration(self, line: str) -> List[str]:
        """Convert C variable declaration to assembly"""
        line = line.rstrip(';').strip()
        
        # Extract type and variable name
        parts = line.split()
        if len(parts) >= 2:
            var_type = parts[0]
            var_name = parts[1]
            
            # Check for initialization
            if '=' in line:
                var_name, init_val = line.split('=', 1)
                var_name = var_name.split()[-1]  # Get last part (variable name)
                init_val = init_val.strip()
                
                return [
                    f"    ; Declaration: {line}",
                    f"    mov eax, {init_val}",
                    f"    mov [" + self._get_var_label(var_name) + "], eax"
                ]
            else:
                return [
                    f"    ; Declaration: {line}",
                    f"    ; Variable {var_name} ({var_type}) declared"
                ]
        
        return [f"    ; Declaration: {line}"]
    
    def extract_c_functions(self, content: str) -> List[Dict]:
        """Extract C function definitions for analysis"""
        functions = []
        
        # Simple regex to find function definitions
        func_pattern = r'(\w+\s+)+(\w+)\s*\([^)]*\)\s*\{'
        matches = re.finditer(func_pattern, content)
        
        for match in matches:
            func_info = {
                'name': match.group(2),
                'full_signature': match.group(0),
                'start_pos': match.start(),
                'end_pos': match.end()
            }
            functions.append(func_info)
        
        return functions
    
    def _clean_gcc_assembly_simple(self, assembly: str, block_id: str = "") -> str:
        """Simple cleaning of GCC assembly - remove directives but keep instructions"""
        lines = assembly.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip unwanted directives and comments - don't include string constants here
            skip_patterns = [
                '/APP', '/NO_APP', '# 0 ""', '# ', '.file', '.intel_syntax',
                '.def', '.scl', '.type', '.size', '.ident', 
                '.globl', '.p2align', '.cfi_', '.build_version',
                '.section', '.align', '.ascii', '.LC'  # Skip these, handle separately
            ]
            
            if any(stripped.startswith(pattern) for pattern in skip_patterns):
                continue
                
            # Skip empty lines
            if not stripped:
                continue
            
            # Keep actual assembly instructions and labels (.L labels are needed for jumps)
            if stripped and not stripped.startswith('#'):
                # Keep .L labels (needed for jumps) but skip other directives starting with .
                if stripped.startswith('.') and not stripped.startswith('.L'):
                    continue
                    
                # Make .L labels unique by adding block ID prefix
                cleaned_line = line.replace('\t', '    ')  # Convert tabs to spaces
                if block_id and '.L' in cleaned_line:
                    # Replace .L labels with unique ones scoped to this block.
                    # Use a prefixed form to avoid creating dotted composite
                    # symbols when a local label is referenced as
                    # "FUNCTION.L3" (which could become
                    # "FUNCTION.CASM_BLOCK_3_L3" and collide). We convert
                    # ".L123" -> "_CASM_BLOCK_3_L123" so references like
                    # "FUNC.L123" turn into "FUNC_CASM_BLOCK_3_L123" and
                    # label definitions become "_CASM_BLOCK_3_L123:" which
                    # prevents the dot from joining two names.
                    import re
                    cleaned_line = re.sub(r'(?<!\w)\.L(\d+)', rf'_{block_id}_L\1', cleaned_line)
                
                # If this line is a label (.L...) or an assembler label ending with ':' keep as-is
                s = cleaned_line.strip()
                if s.startswith('.') and not s.startswith(f'.{block_id}_L'):
                    # keep other dot directives (they've been largely filtered) as-is
                    cleaned_lines.append(cleaned_line)
                elif s.endswith(':') or s.startswith(f'.{block_id}_L'):
                    # It's a label
                    cleaned_lines.append(cleaned_line)
                else:
                    # Surround each instruction with NOP markers so it's easy to spot C-generated
                    cleaned_lines.append(f"    nop  ; start_c_instr {block_id}")
                    cleaned_lines.append(cleaned_line)
                    cleaned_lines.append(f"    nop  ; end_c_instr {block_id}")
        
        return '\n'.join(cleaned_lines)
    
    def _clean_gcc_assembly_to_nasm(self, assembly: str) -> str:
        """Clean GCC assembly output to NASM Intel format"""
        lines = assembly.split('\n')
        cleaned_lines = []
        in_function = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip directives and metadata
            skip_patterns = [
                '.file', '.intel_syntax', '.section', '.def', '.scl', '.type', 
                '.size', '.ident', '.globl', '.p2align', '.cfi_', '.build_version',
                'sdk_version', '.long', '.quad', '@', 'LBB', 'Ltmp', '_GLOBAL_',
                '.subsections_via_symbols', '.macosx_version_min', '.addrsig',
                '.addrsig_sym', '%bb.', '.text', '.data', '.bss'
            ]
            
            if any(stripped.startswith(pattern) for pattern in skip_patterns):
                continue
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Handle function labels
            if stripped.endswith(':') and not stripped.startswith('.'):
                func_name = stripped[:-1]
                if func_name == 'casm_c_block':
                    in_function = True
                    cleaned_lines.append('; Generated C code assembly:')
                    continue
                elif in_function:
                    # End of our function
                    break
            
            # Only include lines from our function
            if in_function:
                # Convert GCC assembly syntax to NASM Intel
                cleaned_line = line
                
                # Fix immediate value syntax (remove $ prefix)
                cleaned_line = re.sub(r'\$(\d+)', r'\1', cleaned_line)
                
                # Fix register syntax (remove % prefix)
                cleaned_line = re.sub(r'%([a-z]+\d*)', r'\1', cleaned_line)
                
                # Fix memory references
                cleaned_line = re.sub(r'QWORD PTR \[([^\]]+)\]', r'qword [\1]', cleaned_line)
                cleaned_line = re.sub(r'DWORD PTR \[([^\]]+)\]', r'dword [\1]', cleaned_line)
                cleaned_line = re.sub(r'WORD PTR \[([^\]]+)\]', r'word [\1]', cleaned_line)
                cleaned_line = re.sub(r'BYTE PTR \[([^\]]+)\]', r'byte [\1]', cleaned_line)
                
                # Skip return instruction as we'll handle it in main flow
                if 'ret' in stripped.lower() and len(stripped.split()) == 1:
                    continue
                
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _generate_header(self) -> str:
        """Generate C header with includes and extern declarations"""
        header_lines = []
        
        # Check if any C code blocks already contain includes
        has_includes = False
        for code_block in self.c_code_blocks:
            if '#include' in code_block:
                has_includes = True
                break
        
        # Normalize and deduplicate headers. Build a canonical list of
        # (header_name, use_angle) entries. We also avoid adding standard
        # headers twice if they appear in self.headers.
        normalized = []
        seen = set()

        # Process self.headers entries into normalized form
        for header_entry in self.headers:
            if isinstance(header_entry, tuple) and len(header_entry) >= 2:
                header_name, use_angle = header_entry[0], bool(header_entry[1])
            else:
                raw = str(header_entry).strip()
                use_angle = raw.startswith('<') and raw.endswith('>')
                header_name = raw.replace('<', '').replace('>', '').strip()

            key = header_name.lower()
            if key in seen:
                # If we've already seen this header, prefer angle form if any
                # existing entry was non-angle and this one requests angle.
                for i, (hn, ua) in enumerate(normalized):
                    if hn.lower() == key and (not ua and use_angle):
                        normalized[i] = (header_name, True)
                continue

            seen.add(key)
            normalized.append((header_name, bool(use_angle)))

        # Only add standard headers if no includes are present in the C blocks
        std_headers = []
        if not has_includes:
            std_headers = ['stdio.h', 'stdlib.h', 'string.h']

        # Emit standard headers first, but only if they weren't supplied already
        for std in std_headers:
            if std.lower() not in seen:
                header_lines.append(f'#include <{std}>')
                seen.add(std.lower())

        # Emit normalized extern headers in the order they were added.
        # Heuristic: prefer angle-bracket includes for typical system headers
        # (simple names that end with .h and don't contain path separators).
        for header_name, use_angle in normalized:
            # Normalize header name for decision
            hn = header_name.strip()
            is_simple_h = bool(re.match(r'^[A-Za-z0-9_\-]+\.h$', hn))

            # Prefer angle includes when explicitly requested, when the
            # header is a simple system header (e.g. stdio.h), or when it
            # contains a path component (e.g. SDL2/SDL.h) which conventionally
            # uses angle brackets.
            effective_angle = bool(use_angle) or is_simple_h or ('/' in hn)

            if effective_angle:
                header_lines.append(f'#include <{hn}>')
            else:
                header_lines.append(f'#include "{hn}"')

        header_lines.append('')
        
        # Add CASM variable declarations as externals, but skip assembler-level
        # symbols that were declared with assembler directives (db, dd, dq, dw,
        # res*, equ). Those are meant to be emitted verbatim in the assembly and
        # should not be re-declared as C externs which can cause redefinition
        # conflicts with system headers (e.g. HKEY_LOCAL_MACHINE).
        asm_directives = {'db', 'dd', 'dq', 'dw', 'resb', 'resw', 'resd', 'resq', 'equ'}
        for var_name, var_info in self.casm_variables.items():
            if isinstance(var_info, dict):
                var_type = var_info.get('type')
                label = var_info.get('label')
                size = var_info.get('size', None)
            else:
                var_type = 'int'
                label = var_info
                size = None

            if var_type in asm_directives:
                # Skip creating C externs/defines for assembler-level symbols
                print_info(f"Skipping C extern for assembler-level symbol: {var_name} ({var_type})")
                continue

            # Convert CASM types to C types
            if var_type == 'str' or var_type == 'string':
                c_type = 'char*'
            elif var_type == 'buffer':
                c_type = 'char*'  # Buffers are char arrays
            elif var_type == 'bool':
                c_type = 'int'  # bool maps to int in C
            elif var_type == 'float':
                c_type = 'double'  # Use double for float compatibility
            elif var_type == 'int':
                if size is not None:
                    c_type = 'int*'  # Arrays are pointers in C
                else:
                    c_type = 'int'
            else:
                c_type = 'int'  # Default fallback

            if c_type == 'char*' or c_type == 'char *' or var_type in ('str', 'string', 'buffer'):
                if size:
                    header_lines.append(f'extern char {label}[{size}];')
                else:
                    header_lines.append(f'extern char {label}[];')
            else:
                header_lines.append(f'extern {c_type} {label};')
            header_lines.append(f'#define {var_name} {label}')
            header_lines.append(f'#define var_{var_name} {label}')
        
        if self.casm_variables:
            header_lines.append('')
        
        return '\n'.join(header_lines)
    
    def add_c_code_block(self, code: str) -> str:
        """Add a C code block and return a marker for assembly extraction"""
        block_id = f"CASM_BLOCK_{self.marker_counter}"
        self.marker_counter += 1
        
        # Use assembly labels that will be preserved in the output
        marked_code = f"""    asm volatile("{block_id}_START:");
    {code}
    asm volatile("{block_id}_END:");
"""
        self.c_code_blocks.append(marked_code)
        # Keep raw code so we can decide later whether this block is a function
        # definition (global) or a set of statements to put inside casm_main.
        self._raw_c_blocks.append(code)
        
        return block_id

    def set_target(self, platform: str, arch: str = 'x86_64'):
        """Set target platform and architecture for C compilation.

        platform: 'linux' or 'windows' or 'macos'
        arch: 'x86_64', 'x86' or 'arm64'
        """
        self.target_platform = (platform or 'linux').lower()
        self.target_arch = (arch or 'x86_64').lower()
        print_info(f"C processor target set to: {self.target_platform} / {self.target_arch}")
    
    def compile_all_c_code(self) -> Dict[str, str]:
        """Compile all collected C code blocks and extract assembly segments"""
        print_info(f"compile_all_c_code called with {len(self.c_code_blocks)} blocks")
        
        if not self.c_code_blocks:
            print_warning("c_code_blocks is empty!")
            return {}
        
        # Ensure temp directory exists
        if not self.temp_dir:
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix='casm_c_')
        
        # Combine all C code into one file
        combined_c = self._generate_header()

        # Decide which raw C blocks are full function definitions.
        # Some function definitions may be split across multiple collected
        # raw blocks (header in one block, body lines in following blocks).
        # Merge consecutive raw blocks into a single function text when the
        # initial block looks like a function header (contains '{' or matches
        # function-def regex), tracking brace depth until the function closes.
        # For non-function statements we also need to merge consecutive
        # raw blocks that together form a single C statement (e.g. multi-line
        # calls split across lines). This prevents producing stray fragments
        # like a lone ");" inside casm_main.
        global_functions = []
        main_statements = []

        func_def_re = re.compile(r'^[a-zA-Z_][\w\s\*]+\s+[a-zA-Z_]\w*\s*\([^\)]*\)\s*\{', re.M)

        raw_blocks = getattr(self, '_raw_c_blocks', [])
        marked_blocks = getattr(self, 'c_code_blocks', [])
        # marked_blocks contains the original marked blocks (with CASM_BLOCK_N
        # labels). When we merge multiple raw blocks into one statement we
        # must preserve the marker ID from the first block so the assembler
        # extraction step can match the expected marker names (otherwise
        # codegen will not find the assembly for the original markers).

        i = 0
        n = len(raw_blocks)
        while i < n:
            raw = raw_blocks[i]

            # If this block starts a function definition (regex or a '{' present)
            if func_def_re.search(raw) or '{' in raw:
                # Merge subsequent blocks until braces are balanced
                combined_raw = raw
                brace_depth = raw.count('{') - raw.count('}')
                j = i + 1
                while brace_depth > 0 and j < n:
                    next_raw = raw_blocks[j]
                    combined_raw += '\n' + next_raw
                    brace_depth += next_raw.count('{') - next_raw.count('}')
                    j += 1

                # Append the full function text as a global function
                global_functions.append(combined_raw)
                # Advance i to the next unprocessed block
                i = j
            else:
                # Merge consecutive non-function raw blocks into a single
                # statement until we reach one that looks like it ends with
                # a semicolon. This handles multi-line expressions such as
                # function calls split across lines (e.g. SDL_CreateWindow(...);
                combined_parts = [raw]
                j = i + 1
                # Continue merging until we see a terminating semicolon at
                # the end of the accumulated text or we run out of blocks or
                # the next block starts a function definition.
                while j < n and not combined_parts[-1].strip().endswith(';') and not func_def_re.search(raw_blocks[j]) and '{' not in raw_blocks[j]:
                    combined_parts.append(raw_blocks[j])
                    j += 1

                combined_raw = '\n'.join(combined_parts)

                # Collect original markers in this merged range so we can
                # alias them back to a primary marker after compilation.
                original_markers = []
                for k in range(i, j):
                    if k < len(marked_blocks):
                        import re as _re
                        mm = _re.search(r'(CASM_BLOCK_\d+)_START', marked_blocks[k])
                        if mm:
                            original_markers.append(mm.group(1))

                # Choose primary marker: prefer first original marker when available
                if original_markers:
                    primary = original_markers[0]
                else:
                    primary = f"CASM_BLOCK_{self.marker_counter}"
                    self.marker_counter += 1

                # Record aliases so we can map extracted assembly back to
                # original markers later on.
                if not hasattr(self, '_marker_aliases'):
                    self._marker_aliases = {}
                for om in original_markers:
                    self._marker_aliases[om] = primary

                marked = '    asm volatile("%s_START:");\n%s\n    asm volatile("%s_END:");' % (primary, combined_raw, primary)
                main_statements.append(marked)
                i = j

        # Some entries in global_functions might not actually be proper
        # function definitions (heuristic misclassification). Ensure we only
        # emit real functions as globals; other blocks should be treated as
        # main statements so runtime calls end up inside casm_main().
        verified_globals = []
        for gf in global_functions:
            # Check if gf begins with a function definition (after optional whitespace)
            if func_def_re.search(gf):
                verified_globals.append(gf)
            else:
                # Treat as main statement (wrap as marked if necessary)
                main_statements.append(gf)

        # Emit verified global function definitions first
        for gf in verified_globals:
            combined_c += gf + "\n\n"

        # Now emit casm_main containing non-function statements (marked)
        combined_c += "\nvoid casm_main() {\n"
        for stmt in main_statements:
            combined_c += stmt + "\n"
        combined_c += "}\n"
        
        print_info(f"Generated combined C code:\n{combined_c}")
        
        # Post-process combined C: move any top-level runtime statements that
        # call functions (invalid as initializers) into casm_main. This lets
        # users write natural C-style initialization lines in the CASM file
        # while ensuring the combined C remains valid C.
        combined_lines = combined_c.split('\n')
        # Find the index of 'void casm_main() {' so we can separate header/global
        try:
            main_idx = next(i for i, l in enumerate(combined_lines) if l.strip().startswith('void casm_main('))
        except StopIteration:
            main_idx = None

        moved_to_main = []
        if main_idx is not None:
            header_lines = combined_lines[:main_idx]
            main_lines = combined_lines[main_idx:]

            # First detect function ranges in header_lines so we won't touch
            # lines that are inside function bodies.
            func_ranges = []  # list of (start_idx, end_idx) inclusive in header_lines
            idx = 0
            while idx < len(header_lines):
                line = header_lines[idx]
                if func_def_re.search(line):
                    # scan forward until matching braces balanced
                    depth = line.count('{') - line.count('}')
                    j = idx + 1
                    while depth > 0 and j < len(header_lines):
                        depth += header_lines[j].count('{') - header_lines[j].count('}')
                        j += 1
                    func_ranges.append((idx, j - 1))
                    idx = j
                else:
                    idx += 1

            def inside_func(i):
                for a, b in func_ranges:
                    if a <= i <= b:
                        return True
                return False

            new_header = []
            # For each line in header_lines (but only those outside functions),
            # detect problematic initializers or top-level statements and move
            # them to casm_main.
            decl_re = re.compile(r'^\s*([A-Za-z_][\w\s\*]+)\s+(\w+)\s*=\s*(.+);\s*$')
            stmt_re = re.compile(r'^\s*[^/\s].*;\s*$')

            for i, line in enumerate(header_lines):
                if inside_func(i):
                    # Preserve function lines intact
                    new_header.append(line)
                    continue

                m = decl_re.match(line)
                if m:
                    decl_type, name, init = m.groups()
                    # If initializer likely contains a function call or a runtime
                    # symbol, move the initializer into casm_main.
                    if '(' in init or any(tok in init for tok in ['GetModuleHandle', 'CreateWindowEx', 'RegisterClass', 'ShowWindow', 'GetMessage', 'TranslateMessage', 'DispatchMessage']):
                        new_header.append(f"{decl_type} {name};")
                        moved_to_main.append(f"    {name} = {init};")
                        continue

                # Top-level statement (not a preprocessor/extern/type) -> move
                if stmt_re.match(line):
                    stripped = line.strip()
                    if not stripped.startswith('#') and not any(stripped.startswith(k) for k in ('extern', 'typedef', 'struct', 'enum')):
                        # Never move bare 'return' lines - they belong in functions
                        if stripped.startswith('return'):
                            new_header.append(line)
                        else:
                            moved_to_main.append(f"    {stripped}")
                        continue

                new_header.append(line)

            # Insert moved statements at the start of casm_main body (after the '{')
            if moved_to_main:
                # Find the opening brace line index in main_lines
                for mi, ml in enumerate(main_lines):
                    if '{' in ml:
                        insert_at = mi + 1
                        break
                else:
                    insert_at = 1

                main_lines = main_lines[:insert_at] + moved_to_main + main_lines[insert_at:]

            # Rebuild combined_c
            combined_c = '\n'.join(new_header + main_lines)

        # Write to temporary file
        c_file = os.path.join(self.temp_dir, "combined.c")
        with open(c_file, 'w') as f:
            f.write(combined_c)

        if self.save_debug:
            print_info(f"Combined C file written to: {c_file}")
        # Also persist the combined C to the project's output/ for debugging
        try:
            import shutil
            out_dir = os.path.join(os.getcwd(), 'output')
            os.makedirs(out_dir, exist_ok=True)
            # Only save when explicitly requested
            if getattr(self, 'save_debug', False):
                # Use a randomized filename to avoid collisions and leaking
                # predictable filenames. Use uuid4 hex.
                import uuid
                rand = uuid.uuid4().hex[:8]
                dst_c = os.path.join(out_dir, f'combined_c_from_cprocessor_{rand}.c')
                shutil.copy(c_file, dst_c)
                print_info(f"Saved combined C to: {dst_c}")
        except Exception as e:
            print_warning(f"Could not save combined C file for debugging: {e}")
        
        # Compile to assembly
        asm_file = os.path.join(self.temp_dir, "combined.s")

        # Gather extra flags (e.g., SDL include flags) when headers request them
        extra_flags = []
        try:
            # If the user provided explicit CFLAGS, use them first
            if getattr(self, 'user_cflags', None):
                if isinstance(self.user_cflags, str):
                    import shlex
                    extra_flags = shlex.split(self.user_cflags)
                else:
                    extra_flags = list(self.user_cflags)
                if self.save_debug:
                    print_info(f"Using user-provided CFLAGS: {' '.join(extra_flags)}")
            else:
                pkg_flags = self._gather_pkg_flags_for_headers()
                if pkg_flags:
                    extra_flags.extend(pkg_flags)
                    if self.save_debug:
                        print_info(f"Using extra compiler flags from pkg-config/sdl2-config: {' '.join(pkg_flags)}")
        except Exception as e:
            print_warning(f"Could not gather pkg-config flags: {e}")

        # Try a sequence of compilers that are likely to use system include paths
        compilers_to_try = []

        # Windows target: prefer mingw cross compilers
        if self.target_platform == 'windows':
            compilers_to_try.append((['x86_64-w64-mingw32-gcc', '-S', '-O0', '-masm=intel'] + extra_flags + [c_file, '-o', asm_file]))
            compilers_to_try.append((['i686-w64-mingw32-gcc', '-S', '-O0', '-masm=intel'] + extra_flags + [c_file, '-o', asm_file]))

        host_flags = ['-S', '-O0', '-masm=intel']
        if self.target_arch in ('x86', 'i386', 'i686'):
            host_flags = ['-S', '-O0', '-m32', '-masm=intel']

        compilers_to_try.append((['gcc'] + host_flags + extra_flags + [c_file, '-o', asm_file]))
        compilers_to_try.append((['clang'] + host_flags + extra_flags + [c_file, '-o', asm_file]))

        compiled = False
        # If we're running on a non-x86_64 host (e.g. Apple Silicon), ensure
        # the x86_64-w64-mingw32 cross-compiler is available; otherwise the
        # host compiler will produce ARM/x86 mismatched assembly which is
        # not usable for a Windows x86-64 target. Provide an actionable
        # error message to the user instead of silently embedding wrong asm.
        try:
            import platform, shutil
            host_arch = platform.machine().lower()
            cross_present = bool(shutil.which('x86_64-w64-mingw32-gcc'))
            if host_arch not in ('x86_64', 'amd64') and not cross_present:
                raise RuntimeError(
                    "Cross-compiler x86_64-w64-mingw32-gcc not found on non-x86_64 host. "
                    "Install it (e.g. `brew install mingw-w64`) and ensure it is in PATH, "
                    "also install pkg-config and SDL2 (e.g. `brew install pkg-config sdl2`) so headers/libs are found.")
        except Exception as e:
            # If we raised above, re-raise so the caller sees a clear message
            if isinstance(e, RuntimeError):
                raise
            # Otherwise continue; platform detection may not be available
            pass
        for cmd in compilers_to_try:
            try:
                # Run compiler and capture output; store for later preview/logging
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.temp_dir)
                combined_out = (result.stdout or '') + '\n' + (result.stderr or '')
                self._last_compile_output = combined_out.strip()

                if result.returncode == 0:
                    if self.save_debug:
                        print_success(f"C compilation successful with {cmd[0]}")
                    self._last_status = f"Compiled with {cmd[0]}"
                    compiled = True
                    break
                else:
                    # Save concise status and full output for later inspection
                    self._last_status = f"Compiler {cmd[0]} failed"
                    # Only emit detailed error messages when debugging
                    if self.save_debug:
                        print_error(f"Compiler {cmd[0]} failed: {result.stderr}")
                    continue
            except FileNotFoundError:
                print_warning(f"Compiler {cmd[0]} not found, trying next")
                continue

        if not compiled:
            if self.save_debug:
                print_warning("No suitable C compiler succeeded")
            raise RuntimeError("GCC failed to compile combined C file with available compilers")
        
        # Read and parse assembly
        try:
            with open(asm_file, 'r') as f:
                assembly = f.read()
            if self.save_debug:
                print_info(f"Read {len(assembly)} characters from assembly file")
                print_info(f"Full assembly output:\n{assembly}")
            # Persist the raw assembly output for debugging
            try:
                out_dir = os.path.join(os.getcwd(), 'output')
                os.makedirs(out_dir, exist_ok=True)
                if getattr(self, 'save_debug', False):
                    import uuid
                    rand = uuid.uuid4().hex[:8]
                    asm_out = os.path.join(out_dir, f'combined_asm_from_cprocessor_{rand}.s')
                    with open(asm_out, 'w') as af:
                        af.write(assembly)
                    print_info(f"Saved combined assembly to: {asm_out}")
            except Exception as e:
                print_warning(f"Could not save combined assembly for debugging: {e}")
        except Exception as e:
            print_error(f"Error reading assembly file: {e}")
            return {}
        
        # Extract assembly segments by markers
        segments = self._extract_assembly_segments(assembly)
        print_info(f"Extracted {len(segments)} assembly segments")
        
        # Also extract string constants
        string_constants = self._extract_string_constants(assembly)
        if string_constants:
            print_info(f"Extracted {len(string_constants)} string constants")
            segments['_STRING_CONSTANTS'] = string_constants

        # If we created aliases during merging, map original markers to the
        # primary extracted segment so the consumer (codegen) can look up
        # the assembly by the original CASM block names.
        if hasattr(self, '_marker_aliases') and self._marker_aliases:
            for orig, primary in list(self._marker_aliases.items()):
                if primary in segments and orig not in segments:
                    segments[orig] = segments[primary]
                    print_info(f"Aliased segment: {orig} -> {primary}")
        
        # Store for pretty printer access
        self._last_assembly_segments = segments.copy()
        
        return segments
    
    def _extract_assembly_segments(self, assembly: str) -> Dict[str, str]:
        """Extract assembly segments between assembly labels"""
        print_info("Starting assembly segment extraction (looking for assembly labels)")
        segments = {}
        lines = assembly.split('\n')
        current_block = None
        current_assembly = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Look for our assembly labels
            if "CASM_BLOCK_" in stripped and "_START:" in stripped:
                # Extract block ID from the label
                match = re.search(r'(CASM_BLOCK_\d+)_START:', stripped)
                if match:
                    current_block = match.group(1)
                    current_assembly = []
                    print_info(f"Found start label for {current_block} at line {i}")
            elif "CASM_BLOCK_" in stripped and "_END:" in stripped and current_block:
                # End of block - save the collected assembly
                raw_assembly = '\n'.join(current_assembly)
                # Clean the assembly but keep the core instructions, making labels unique
                cleaned = self._clean_gcc_assembly_simple(raw_assembly, current_block)
                segments[current_block] = cleaned
                print_info(f"Extracted {len(current_assembly)} lines for {current_block}")
                print_info(f"Cleaned assembly for {current_block}:\n{cleaned}")
                current_block = None
                current_assembly = []
            elif current_block:
                # Collect assembly lines between the labels
                current_assembly.append(line)
        
        print_info(f"Final segments extracted: {list(segments.keys())}")
        
        # Post-process to fix cross-block label references
        segments = self._fix_cross_block_labels(segments)
        
        return segments
    
    def _fix_cross_block_labels(self, segments: Dict[str, str]) -> Dict[str, str]:
        """Fix label references that span across different C code blocks"""
        print_info("Fixing cross-block label references...")
        
        # Collect all label definitions and their block IDs
        label_definitions = {}  # original_label -> (block_id, new_label)
        label_references = {}   # block_id -> [(original_label, line_number)]
        
        # First pass: find all label definitions
        for block_id, assembly in segments.items():
            lines = assembly.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Look for label definitions like ".BLOCK_15_L2:"
                import re
                label_match = re.match(r'\.(' + re.escape(block_id) + r'_L\d+):', stripped)
                if label_match:
                    new_label = label_match.group(1)
                    # Extract original label (L2 from BLOCK_15_L2)
                    original_match = re.search(r'_L(\d+)', new_label)
                    if original_match:
                        original_label = f"L{original_match.group(1)}"
                        # If this original_label already has a definition, we
                        # will still record it; duplicates will be resolved
                        # below so labels become unique across blocks.
                        if original_label in label_definitions:
                            # Convert existing entry into a list to preserve
                            # multiple definitions (we'll make them unique)
                            prev = label_definitions[original_label]
                            if isinstance(prev[0], list):
                                prev[0].append((block_id, new_label))
                            else:
                                label_definitions[original_label] = ([prev, (block_id, new_label)], None)
                        else:
                            label_definitions[original_label] = (block_id, new_label)
                        print_info(f"Found label definition: .{original_label} -> .{new_label} in {block_id}")
        
        # label_definitions currently maps original_label -> (block_id, new_label)
        # but may contain lists for duplicates. Normalize into a mapping of
        # original_label -> list of (block_id, new_label) so we can make
        # duplicate new_label names unique.
        normalized_defs = {}
        for orig, val in list(label_definitions.items()):
            if isinstance(val[0], list):
                # val is ([ (block,new_label), (block,new_label) ], None)
                pairs = val[0]
            else:
                pairs = [val]
            normalized_defs[orig] = pairs

        # Simpler approach: rewrite local GCC '.L<number>' references so they
        # become underscore-scoped names that include the originating block id.
        # This avoids generating dotted composites like 'FUNC.CASM_BLOCK_3_L2'
        # which NASM treats as separate symbols and can cause redefinition
        # errors when multiple definitions collide. The scoped name we use is
        # '_<BLOCKID>_L<number>' and qualified references become
        # 'PREFIX_<BLOCKID>_L<number>'.
        fixed_segments = {}
        import re
        for block_id, assembly in segments.items():
            lines = assembly.split('\n')
            out_lines = []
            for line in lines:
                sline = line
                # Normalize label definitions that may start with a dot or underscore
                sline = re.sub(r"^\s*\.?" + re.escape(block_id) + r"_L(\d+):", lambda m: f"_{block_id}_L{m.group(1)}:", sline)

                # Replace standalone .L123 references with _<block>_L123
                sline = re.sub(r'\.L(\d+)(?!\w)', lambda m: f"_{block_id}_L{m.group(1)}", sline)

                # Replace qualified NAME.L123 -> NAME_<block>_L123
                sline = re.sub(r'(?P<prefix>\b\w+)\.L(\d+)(?!\w)', lambda m: f"{m.group('prefix')}_{block_id}_L{m.group(2)}", sline)

                out_lines.append(sline)
            fixed_segments[block_id] = '\n'.join(out_lines)

        return fixed_segments
    
    def _extract_string_constants(self, assembly: str) -> str:
        """Extract and convert string constants from GCC to NASM format"""
        lines = assembly.split('\n')
        string_lines = []
        current_label = None
        
        for line in lines:
            stripped = line.strip()
            
            # Find .LC labels (including .LC0)
            if stripped.startswith('.LC') and stripped.endswith(':'):
                current_label = stripped[:-1]  # Remove the colon
                continue
            
            # Find .ascii strings and convert to NASM format
            if stripped.startswith('.ascii') and current_label:
                # Extract the string content more robustly
                import re
                match = re.search(r'\.ascii\s+"(.*)"', stripped)
                if match:
                    string_content = match.group(1)
                    
                    # Handle escape sequences properly by building a clean sequence
                    # Convert escape sequences to NASM format
                    result_parts = []
                    i = 0
                    current_str = ""
                    
                    while i < len(string_content):
                        if i + 1 < len(string_content) and string_content[i:i+2] in ['\\n', '\\r', '\\t']:
                            # Handle standard escape sequences
                            if current_str:
                                result_parts.append(f'"{current_str}"')
                                current_str = ""
                            if string_content[i:i+2] == '\\n':
                                result_parts.append('10')
                            elif string_content[i:i+2] == '\\r':
                                result_parts.append('13')
                            elif string_content[i:i+2] == '\\t':
                                result_parts.append('9')
                            i += 2
                        elif i + 2 < len(string_content) and string_content[i:i+3] in ['\\12', '\\15']:
                            # Handle octal escape sequences
                            if current_str:
                                result_parts.append(f'"{current_str}"')
                                current_str = ""
                            if string_content[i:i+3] == '\\12':
                                result_parts.append('10')
                            elif string_content[i:i+3] == '\\15':
                                result_parts.append('13')
                            i += 3
                        elif i + 1 < len(string_content) and string_content[i:i+2] == '\\0':
                            # Handle null terminator
                            if current_str:
                                result_parts.append(f'"{current_str}"')
                                current_str = ""
                            result_parts.append('0')
                            i += 2
                        else:
                            current_str += string_content[i]
                            i += 1
                    
                    # Add any remaining string content
                    if current_str:
                        result_parts.append(f'"{current_str}"')
                    
                    # Ensure we have a null terminator
                    if not result_parts or result_parts[-1] != '0':
                        result_parts.append('0')
                    
                    # Join the parts
                    if result_parts:
                        string_content = ', '.join(result_parts)
                    
                    # Create NASM-style string using CSTR prefix instead of LC
                    # Convert a label like .LC0 to CSTR0
                    lbl = current_label.lstrip('.')
                    if lbl.startswith('LC'):
                        num = lbl[2:]
                        lbl = f"CSTR{num}"
                    string_lines.append(f"    {lbl} db {string_content}")
                    current_label = None
            
            # Handle floating point constants like .LC0
            elif (stripped.startswith('.long') or stripped.startswith('.quad')) and current_label:
                # This is likely a floating point constant - collect all parts
                if not hasattr(self, '_current_fp_constant'):
                    self._current_fp_constant = []

                # Convert GCC syntax to NASM syntax
                if stripped.startswith('.long'):
                    # Extract the value and convert to dd
                    parts = stripped.split()
                    if len(parts) >= 2:
                        value = parts[1]
                        self._current_fp_constant.append(f"        dd {value}")
                elif stripped.startswith('.quad'):
                    # Extract the value and convert to dq
                    parts = stripped.split()
                    if len(parts) >= 2:
                        value = parts[1]
                        self._current_fp_constant.append(f"        dq {value}")

                # Check if we have both parts of a double (two .long entries)
                if len(self._current_fp_constant) >= 2 or stripped.startswith('.quad'):
                    # Convert .LC labels to CSTR labels
                    lbl = current_label.lstrip('.')
                    if lbl.startswith('LC'):
                        num = lbl[2:]
                        lbl = f"CSTR{num}"
                    string_lines.append(f"    {lbl}:")
                    for part in self._current_fp_constant:
                        string_lines.append(part)
                    self._current_fp_constant = []
                    current_label = None
            
            # Handle .align directives followed by .LC labels
            elif stripped.startswith('.align') and current_label:
                # Keep the label for the next data
                continue
        
        return '\n'.join(string_lines)

    def _gather_pkg_flags_for_headers(self) -> List[str]:
        """Try to gather compiler flags for headers (SDL2) via pkg-config or sdl2-config.

        Returns a list of flags (e.g. ['-I/opt/homebrew/include']) to prepend to the compiler.
        """
        flags: List[str] = []

        # If the user explicitly provided CFLAGS via the CLI, prefer those
        # and return them directly. Use this to allow explicit override of
        # auto-detection (pkg-config / sdl2-config).
        if getattr(self, 'user_cflags', None):
            try:
                # If it's a string, split on whitespace; if it's already a list
                # assume it's ready to use.
                if isinstance(self.user_cflags, str):
                    import shlex
                    return shlex.split(self.user_cflags)
                return list(self.user_cflags)
            except Exception:
                # Fall back to auto-detection if parsing fails
                pass

        # Collect header names we asked for
        header_names = [h for (h, _) in self.headers]

        # Determine if SDL2 is requested
        want_sdl = any('sdl2' in h.lower() or h.lower().endswith('sdl.h') for h in header_names)
        if not want_sdl:
            return flags

        import shutil
        # Try pkg-config first
        try:
            if shutil.which('pkg-config'):
                result = subprocess.run(['pkg-config', '--cflags', 'sdl2'], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    cflags = result.stdout.strip().split()
                    flags.extend(cflags)
                    # If headers use 'SDL2/...' but pkg-config returned -I.../SDL2,
                    # also add the parent include path so '#include <SDL2/SDL.h>'
                    # resolves correctly when the include is written with the 'SDL2/' prefix.
                    if any('/SDL2' in f for f in cflags) and any('SDL2/' in h for h in header_names):
                        for f in cflags:
                            if f.startswith('-I'):
                                inc_path = f[2:]
                                if inc_path.endswith('/SDL2'):
                                    parent_dir = inc_path[:-len('/SDL2')]
                                    if parent_dir:
                                        parent_flag = '-I' + parent_dir
                                        if parent_flag not in flags:
                                            flags.append(parent_flag)
                    return flags
        except Exception:
            pass

        # Fall back to sdl2-config
        try:
            if shutil.which('sdl2-config'):
                res = subprocess.run(['sdl2-config', '--cflags'], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    cflags = res.stdout.strip().split()
                    flags.extend(cflags)
                    if any('/SDL2' in f for f in cflags) and any('SDL2/' in h for h in header_names):
                        for f in cflags:
                            if f.startswith('-I'):
                                inc_path = f[2:]
                                if inc_path.endswith('/SDL2'):
                                    parent_dir = inc_path[:-len('/SDL2')]
                                    if parent_dir:
                                        parent_flag = '-I' + parent_dir
                                        if parent_flag not in flags:
                                            flags.append(parent_flag)
                    return flags
        except Exception:
            pass

        return flags

    def _gather_pkg_link_flags_for_headers(self) -> List[str]:
        """Gather linker flags for headers (SDL2) via pkg-config or sdl2-config.

        Returns a list of linker flags (e.g. ['-L/path', '-lSDL2']) to append to the linker.
        """
        flags: List[str] = []

        # If user provided explicit ldflags, use those first
        if getattr(self, 'user_ldflags', None):
            try:
                if isinstance(self.user_ldflags, str):
                    import shlex
                    return shlex.split(self.user_ldflags)
                return list(self.user_ldflags)
            except Exception:
                pass

        # Collect header names we asked for
        header_names = [h for (h, _) in self.headers]

        want_sdl = any('sdl2' in h.lower() or h.lower().endswith('sdl.h') for h in header_names)
        if not want_sdl:
            return flags

        import shutil, subprocess
        # Try pkg-config first for linker flags
        try:
            if shutil.which('pkg-config'):
                result = subprocess.run(['pkg-config', '--libs', 'sdl2'], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    libs = result.stdout.strip().split()
                    flags.extend(libs)
                    return flags
        except Exception:
            pass

        # Fall back to sdl2-config --libs
        try:
            if shutil.which('sdl2-config'):
                res = subprocess.run(['sdl2-config', '--libs'], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    libs = res.stdout.strip().split()
                    flags.extend(libs)
                    return flags
        except Exception:
            pass

        return flags
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

# Global instance
c_processor = CCodeProcessor()