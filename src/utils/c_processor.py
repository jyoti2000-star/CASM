#!/usr/bin/env python3

"""
C Code processor for CASM
Processes embedded C code blocks marked with %!
"""

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
        self.headers = []         # Track extern headers
        self.c_code_blocks = []   # Collect all C code blocks with markers
        self.assembly_segments = {} # Store extracted assembly segments by marker ID
        self.marker_counter = 0   # Counter for generating unique block markers
        
    def add_header(self, header_name: str):
        """Add header file for C compilation"""
        if header_name not in self.headers:
            self.headers.append(header_name)
            print_info(f"Added header: {header_name}")
        
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
                    self.casm_variables[var_name] = {
                        'type': var_type,
                        'value': var_value,
                        'label': f'var_{var_name}'
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
                
                # Add extern headers
                for header in self.headers:
                    if header.endswith('.h'):
                        f.write(f'#include "{header}"\n')
                    else:
                        f.write(f'#include <{header}>\n')
                
                f.write('\n')
                
                # Add CASM variable declarations as externals
                for var_name, var_info in self.casm_variables.items():
                    if isinstance(var_info, dict):
                        var_type = var_info['type']
                        label = var_info['label']
                    else:
                        var_type = 'int'  # Default type
                        label = var_info
                    
                    # Declare with both the CASM name and the label for compatibility
                    f.write(f'extern {var_type} {label};\n')
                    f.write(f'#define {var_name} {label}\n')  # Allow C code to use CASM variable names
                
                f.write('\n')
                
                # Create a function wrapper for the C code
                f.write('void casm_c_block() {\n')
                f.write(f'    {c_code}\n')
                f.write('}\n')
            
            # Debug: print the generated C file content
            with open(c_file, 'r') as f:
                c_content = f.read()
            print_info(f"Generated C file content:\n{c_content}")
            
            # Compile with x86_64-w64-mingw32-gcc using Intel syntax
            asm_file = os.path.join(self.temp_dir, 'temp.s')
            
            # Try the specific compiler you mentioned
            cmd = ['x86_64-w64-mingw32-gcc', '-S', '-O0', '-masm=intel', c_file, '-o', asm_file]
            
            try:
                print_info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.temp_dir)
                
                if result.returncode == 0:
                    # Read generated assembly
                    with open(asm_file, 'r') as f:
                        assembly = f.read()
                    
                    print_info(f"Generated {len(assembly)} characters of assembly")
                    # Embed the raw GCC assembly output directly
                    embedded_assembly = f"; === RAW GCC ASSEMBLY OUTPUT ===\n; {' '.join(cmd)}\n"
                    embedded_assembly += assembly
                    embedded_assembly += "; === END GCC ASSEMBLY OUTPUT ==="
                    
                    print_success(f"C code compiled with x86_64-w64-mingw32-gcc (Intel syntax) - raw output embedded")
                    return embedded_assembly
                else:
                    print_error(f"GCC compilation failed: {result.stderr}")
                    
            except FileNotFoundError:
                print_warning("x86_64-w64-mingw32-gcc not found, trying fallback compilers")
            
            # Fallback to other compilers
            fallback_compilers = [
                ['gcc', '-S', '-O0', '-masm=intel'],
                ['clang', '-S', '-O0', '-x86-asm-syntax=intel']
            ]
            
            for compiler_cmd in fallback_compilers:
                try:
                    full_cmd = compiler_cmd + [c_file, '-o', asm_file]
                    result = subprocess.run(full_cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        with open(asm_file, 'r') as f:
                            assembly = f.read()
                        
                        # Embed raw assembly output directly
                        embedded_assembly = f"; === RAW {compiler_cmd[0].upper()} ASSEMBLY OUTPUT ===\n; {' '.join(full_cmd)}\n"
                        embedded_assembly += assembly
                        embedded_assembly += f"; === END {compiler_cmd[0].upper()} ASSEMBLY OUTPUT ==="
                        
                        print_success(f"C code compiled with {compiler_cmd[0]} (Intel syntax) - raw output embedded")
                        return embedded_assembly
                        
                except FileNotFoundError:
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
                            
                            cleaned_lines.append(f"    {instr} {operands}")
                        else:
                            cleaned_lines.append(f"    {instr}")
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
                    f"    mov [var_{left}], eax"
                ]
            except ValueError:
                # Check if it's a simple variable reference
                if right.isidentifier():
                    return [
                        f"    ; {line}",
                        f"    mov eax, [var_{right}]",
                        f"    mov [var_{left}], eax"
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
                # Generic function call
                asm_lines.extend([
                    f"    ; TODO: Set up arguments for {func_name}",
                    f"    call {func_name}"
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
                    f"    mov [var_{var_name}], eax"
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
    
    def _clean_gcc_assembly_simple(self, assembly: str) -> str:
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
            
            # Keep actual assembly instructions
            if stripped and not stripped.startswith('.') and not stripped.startswith('#'):
                # Simple register name fixes for NASM
                cleaned_line = line.replace('\t', '    ')  # Convert tabs to spaces
                cleaned_lines.append(cleaned_line)
        
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
        
        # Add standard C headers
        header_lines.extend([
            '#include <stdio.h>',
            '#include <stdlib.h>',
            '#include <string.h>',
        ])
        
        # Add extern headers
        for header in self.headers:
            if header.endswith('.h'):
                header_lines.append(f'#include "{header}"')
            else:
                header_lines.append(f'#include <{header}>')
        
        header_lines.append('')
        
        # Add CASM variable declarations as externals
        for var_name, var_info in self.casm_variables.items():
            if isinstance(var_info, dict):
                var_type = var_info['type']
                label = var_info['label']
            else:
                var_type = 'int'  # Default type
                label = var_info
            
            # Declare with both the CASM name and the label for compatibility
            header_lines.append(f'extern {var_type} {label};')
        
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
        
        return block_id
    
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
        combined_c += "\nvoid casm_main() {\n"
        combined_c += "".join(self.c_code_blocks)
        combined_c += "}\n"
        
        print_info(f"Generated combined C code:\n{combined_c}")
        
        # Write to temporary file
        c_file = os.path.join(self.temp_dir, "combined.c")
        with open(c_file, 'w') as f:
            f.write(combined_c)
        
        print_info(f"Combined C file written to: {c_file}")
        
        # Compile to assembly
        asm_file = os.path.join(self.temp_dir, "combined.s")
        cmd = [
            "x86_64-w64-mingw32-gcc", "-S", "-O0", "-masm=intel",
            c_file, "-o", asm_file
        ]
        
        print_info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print_success(f"GCC compilation successful")
            if result.stdout:
                print_info(f"GCC stdout: {result.stdout}")
            if result.stderr:
                print_info(f"GCC stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print_error(f"Error compiling combined C code: {e}")
            print_error(f"GCC stdout: {e.stdout}")
            print_error(f"GCC stderr: {e.stderr}")
            return {}
        
        # Read and parse assembly
        try:
            with open(asm_file, 'r') as f:
                assembly = f.read()
            print_info(f"Read {len(assembly)} characters from assembly file")
            print_info(f"Full assembly output:\n{assembly}")
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
                # Clean the assembly but keep the core instructions
                cleaned = self._clean_gcc_assembly_simple(raw_assembly)
                segments[current_block] = cleaned
                print_info(f"Extracted {len(current_assembly)} lines for {current_block}")
                print_info(f"Cleaned assembly for {current_block}:\n{cleaned}")
                current_block = None
                current_assembly = []
            elif current_block:
                # Collect assembly lines between the labels
                current_assembly.append(line)
        
        print_info(f"Final segments extracted: {list(segments.keys())}")
        return segments
    
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
                # Extract the string content
                match = re.search(r'\.ascii\s+"([^"]*)"', stripped)
                if match:
                    string_content = match.group(1)
                    # Convert \12 to proper newline and \0 to null terminator
                    if '\\12\\0' in string_content:
                        # Handle the case where both \12 and \0 are present
                        string_content = string_content.replace('\\12\\0', '", 10, 0')
                    elif '\\12' in string_content:
                        string_content = string_content.replace('\\12', '", 10, "')
                        if string_content.endswith('"'):
                            string_content = string_content[:-1] + '0'
                    elif '\\0' in string_content:
                        string_content = string_content.replace('\\0', '", 0')
                    else:
                        string_content += '", 0'
                    
                    # Create NASM-style string
                    string_lines.append(f"    {current_label} db \"{string_content}")
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
                    string_lines.append(f"    {current_label}:")
                    for part in self._current_fp_constant:
                        string_lines.append(part)
                    self._current_fp_constant = []
                    current_label = None
            
            # Handle .align directives followed by .LC labels
            elif stripped.startswith('.align') and current_label:
                # Keep the label for the next data
                continue
        
        return '\n'.join(string_lines)
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

# Global instance
c_processor = CCodeProcessor()