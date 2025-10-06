#!/usr/bin/env python3

import re
import os
import tempfile
import subprocess
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum
from dataclasses import dataclass

class CTokenType(Enum):
    # C command types
    C_COMMAND = "C_COMMAND"
    C_IDENTIFIER = "C_IDENTIFIER" 
    C_NUMBER = "C_NUMBER"
    C_STRING = "C_STRING"
    C_OPERATOR = "C_OPERATOR"
    C_PUNCTUATION = "C_PUNCTUATION"
    C_KEYWORD = "C_KEYWORD"
    C_NEWLINE = "C_NEWLINE"
    C_COMMENT = "C_COMMENT"

@dataclass
class CToken:
    type: CTokenType
    value: str
    line: int
    column: int
    command_name: str = ""  # The C command this token belongs to

@dataclass
class HASMVariable:
    name: str
    value: Any
    var_type: str  # 'int', 'string', 'array', etc.
    line_number: int
    is_array: bool = False
    array_size: int = 0

class CCodeGenerator:
    def __init__(self):
        self.hasm_variables = {}  # name -> HASMVariable
        self.c_variables = {}     # name -> C variable info
        self.string_constants = []
        self.data_section = []
        self.constant_counter = 0  # Global counter for string constants
        self.variable_counter = 0
        self.stack_offset = 0
        
        # Global counters for variable naming
        self.hasm_var_counter = 0  # For V01, V02, V03...
        self.c_var_counter = 0     # For L01, L02, L03...
        self.hasm_var_mapping = {}  # original_name -> V01
        self.c_var_mapping = {}     # original_name -> L01
        
        # C language keywords
        self.c_keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
            'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
            'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
            'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
            'unsigned', 'void', 'volatile', 'while', '_Bool', '_Complex', '_Imaginary',
            'include', 'define', 'ifdef', 'ifndef', 'endif', 'pragma', 'undef',
            'malloc', 'free', 'calloc', 'realloc', 'strlen', 'strcpy', 'strcmp',
            'printf', 'fprintf', 'sprintf', 'scanf', 'fscanf', 'sscanf', 'memcpy',
            'memset', 'memmove', 'memcmp', 'strcat', 'strchr', 'strstr', 'strtok'
        }
        
        # C operators
        self.c_operators = [
            ('++', 'INCREMENT'), ('--', 'DECREMENT'), ('==', 'EQUALS'),
            ('!=', 'NOT_EQUALS'), ('<=', 'LESS_EQUAL'), ('>=', 'GREATER_EQUAL'),
            ('&&', 'LOGICAL_AND'), ('||', 'LOGICAL_OR'), ('<<', 'LEFT_SHIFT'),
            ('>>', 'RIGHT_SHIFT'), ('->', 'ARROW'), ('+=', 'PLUS_ASSIGN'),
            ('-=', 'MINUS_ASSIGN'), ('*=', 'MULT_ASSIGN'), ('/=', 'DIV_ASSIGN'),
            ('%=', 'MOD_ASSIGN'), ('&=', 'AND_ASSIGN'), ('|=', 'OR_ASSIGN'),
            ('^=', 'XOR_ASSIGN'), ('<', 'LESS_THAN'), ('>', 'GREATER_THAN'),
            ('=', 'ASSIGN'), ('+', 'PLUS'), ('-', 'MINUS'), ('*', 'MULTIPLY'),
            ('/', 'DIVIDE'), ('%', 'MODULO'), ('&', 'BITWISE_AND'),
            ('|', 'BITWISE_OR'), ('^', 'BITWISE_XOR'), ('~', 'BITWISE_NOT'),
            ('!', 'LOGICAL_NOT'), ('?', 'TERNARY'), (':', 'COLON'),
            (';', 'SEMICOLON'), (',', 'COMMA'), ('.', 'DOT'),
        ]
        
        # C punctuation
        self.c_punctuation = [
            ('(', 'LPAREN'), (')', 'RPAREN'), ('[', 'LBRACKET'),
            (']', 'RBRACKET'), ('{', 'LBRACE'), ('}', 'RBRACE'),
        ]
    
    def is_c_command(self, line: str) -> bool:
        """Check if a line contains a C command (starts with %!)"""
        return line.strip().startswith('%!')
    
    # ========== HASM VARIABLE PARSING ==========
    
    def parse_hasm_variables(self, content: str) -> Dict[str, HASMVariable]:
        """Parse HASM variables from the content"""
        variables = {}
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Parse %var declarations
            if stripped.startswith('%var '):
                var_info = self._parse_var_declaration(stripped, line_num)
                if var_info:
                    variables[var_info.name] = var_info
        
        return variables
    
    def _parse_var_declaration(self, line: str, line_num: int) -> Optional[HASMVariable]:
        """Parse a single %var declaration"""
        # Remove %var prefix
        var_part = line[5:].strip()
        
        # Check for array declaration: name[size] value
        array_match = re.match(r'(\w+)\[(\d+)\]\s+(.+)', var_part)
        if array_match:
            name = array_match.group(1)
            size = int(array_match.group(2))
            value = array_match.group(3)
            return HASMVariable(
                name=name,
                value=value,
                var_type='array',
                line_number=line_num,
                is_array=True,
                array_size=size
            )
        
        # Regular variable: name value
        parts = var_part.split(None, 1)
        if len(parts) >= 2:
            name = parts[0]
            value = parts[1]
            
            # Determine type based on value
            if value.startswith('"') and value.endswith('"'):
                var_type = 'string'
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                var_type = 'int'
            else:
                var_type = 'unknown'
            
            return HASMVariable(
                name=name,
                value=value,
                var_type=var_type,
                line_number=line_num
            )
        
        return None
    
    # ========== C CODE TOKENIZATION ==========
    
    def extract_c_commands(self, text: str) -> List[Dict[str, Any]]:
        """Extract all C commands from the text"""
        c_commands = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if self.is_c_command(line):
                command_info = self._parse_c_command_line(line, line_num)
                if command_info:
                    c_commands.append(command_info)
        
        return c_commands
    
    def _parse_c_command_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse a single C command line"""
        stripped = line.strip()
        if not stripped.startswith('%!'):
            return None
        
        # Remove the %! prefix to get the C code
        c_code = stripped[2:].strip()
        
        # Extract command name (first word of C code or function name)
        command_name = self._extract_command_name(c_code)
        
        # Tokenize the C code part
        tokens = self._tokenize_c_code(c_code, line_num, command_name)
        
        # Analyze the C construct
        construct_info = self._analyze_c_construct(c_code, tokens)
        
        return {
            'command_name': command_name,
            'line_number': line_num,
            'original_line': line,
            'c_code': c_code,
            'tokens': tokens,
            'construct_type': construct_info['type'],
            'construct_details': construct_info['details']
        }
    
    def _extract_command_name(self, c_code: str) -> str:
        """Extract the main command/function name from C code"""
        # Look for function calls: function_name(
        func_match = re.search(r'(\w+)\s*\(', c_code)
        if func_match:
            return func_match.group(1)
        
        # Look for variable declarations: type var_name
        var_match = re.search(r'^\s*\w+\s+(\w+)', c_code)
        if var_match:
            return var_match.group(1)
        
        # Look for simple assignments: var_name =
        assign_match = re.search(r'^(\w+)\s*=', c_code)
        if assign_match:
            return assign_match.group(1)
        
        # Default: first word
        parts = c_code.split()
        return parts[0] if parts else "unknown"
    
    def _analyze_c_construct(self, c_code: str, tokens: List[CToken]) -> Dict[str, Any]:
        """Analyze what type of C construct this is"""
        construct_info = {
            'type': 'unknown',
            'details': {}
        }
        
        # Function call: function_name(args)
        if re.search(r'\w+\s*\([^)]*\)', c_code):
            construct_info['type'] = 'function_call'
            func_match = re.search(r'(\w+)\s*\(([^)]*)\)', c_code)
            if func_match:
                construct_info['details'] = {
                    'function_name': func_match.group(1),
                    'arguments': [arg.strip() for arg in func_match.group(2).split(',') if arg.strip()]
                }
        
        # Variable declaration: type var = value;
        elif re.search(r'^\s*\w+\s+\w+\s*=', c_code):
            construct_info['type'] = 'variable_declaration'
            var_match = re.search(r'^\s*(\w+)\s+(\w+)\s*=\s*(.+?);?$', c_code)
            if var_match:
                construct_info['details'] = {
                    'type': var_match.group(1),
                    'variable_name': var_match.group(2),
                    'initial_value': var_match.group(3)
                }
        
        # Simple assignment: var = value;
        elif re.search(r'^\s*\w+\s*=', c_code):
            construct_info['type'] = 'assignment'
            assign_match = re.search(r'^\s*(\w+)\s*=\s*(.+?);?$', c_code)
            if assign_match:
                construct_info['details'] = {
                    'variable_name': assign_match.group(1),
                    'value': assign_match.group(2)
                }
        
        # Preprocessor directive: #include, #define, etc.
        elif c_code.strip().startswith('#'):
            construct_info['type'] = 'preprocessor'
            construct_info['details'] = {
                'directive': c_code.strip()
            }
        
        # Control flow: if, for, while, etc.
        elif re.search(r'^\s*(if|for|while|switch|return)\s*[\(;]', c_code):
            construct_info['type'] = 'control_flow'
            control_match = re.search(r'^\s*(\w+)', c_code)
            if control_match:
                construct_info['details'] = {
                    'statement_type': control_match.group(1),
                    'condition': c_code[len(control_match.group(1)):].strip()
                }
        
        return construct_info
    
    def _tokenize_c_code(self, c_code: str, line_num: int, command_name: str) -> List[CToken]:
        """Tokenize C code part of a command"""
        tokens = []
        i = 0
        col = 1
        
        while i < len(c_code):
            # Skip whitespace
            if c_code[i].isspace():
                i += 1
                col += 1
                continue
            
            # Check for C operators (multi-character first)
            found_op = False
            for op_text, op_type in self.c_operators:
                if c_code[i:i+len(op_text)] == op_text:
                    token = CToken(CTokenType.C_OPERATOR, op_text, line_num, col, command_name)
                    tokens.append(token)
                    i += len(op_text)
                    col += len(op_text)
                    found_op = True
                    break
            
            if found_op:
                continue
            
            # Check for C punctuation
            found_punct = False
            for punct_text, punct_type in self.c_punctuation:
                if c_code[i:i+len(punct_text)] == punct_text:
                    token = CToken(CTokenType.C_PUNCTUATION, punct_text, line_num, col, command_name)
                    tokens.append(token)
                    i += len(punct_text)
                    col += len(punct_text)
                    found_punct = True
                    break
            
            if found_punct:
                continue
            
            # Check for strings
            if c_code[i] in ['"', "'"]:
                string_val, new_i = self._extract_string(c_code, i)
                token = CToken(CTokenType.C_STRING, string_val, line_num, col, command_name)
                tokens.append(token)
                col += new_i - i
                i = new_i
                continue
            
            # Check for numbers
            if c_code[i].isdigit() or (c_code[i] == '.' and i + 1 < len(c_code) and c_code[i + 1].isdigit()):
                num_val, new_i = self._extract_number(c_code, i)
                token = CToken(CTokenType.C_NUMBER, num_val, line_num, col, command_name)
                tokens.append(token)
                col += new_i - i
                i = new_i
                continue
            
            # Check for identifiers/keywords
            if c_code[i].isalpha() or c_code[i] == '_':
                identifier, new_i = self._extract_identifier(c_code, i)
                
                # Check if it's a C keyword
                if identifier.lower() in self.c_keywords:
                    token_type = CTokenType.C_KEYWORD
                else:
                    token_type = CTokenType.C_IDENTIFIER
                
                token = CToken(token_type, identifier, line_num, col, command_name)
                tokens.append(token)
                col += new_i - i
                i = new_i
                continue
            
            # Skip other characters
            i += 1
            col += 1
        
        return tokens
    
    def _extract_string(self, code: str, start: int) -> Tuple[str, int]:
        """Extract a string literal"""
        quote_char = code[start]
        i = start + 1
        result = quote_char
        
        while i < len(code):
            if code[i] == '\\' and i + 1 < len(code):
                # Handle escape sequences
                result += code[i:i+2]
                i += 2
            elif code[i] == quote_char:
                result += code[i]
                return result, i + 1
            else:
                result += code[i]
                i += 1
        
        # Unclosed string
        return result, i
    
    def _extract_number(self, code: str, start: int) -> Tuple[str, int]:
        """Extract a number (integer or float)"""
        i = start
        result = ""
        has_dot = False
        
        while i < len(code):
            if code[i].isdigit():
                result += code[i]
                i += 1
            elif code[i] == '.' and not has_dot:
                has_dot = True
                result += code[i]
                i += 1
            elif code[i] in 'fFlLuU':  # Number suffixes
                result += code[i]
                i += 1
                break
            else:
                break
        
        return result, i
    
    def _extract_identifier(self, code: str, start: int) -> Tuple[str, int]:
        """Extract an identifier or keyword"""
        i = start
        result = ""
        
        while i < len(code) and (code[i].isalnum() or code[i] == '_'):
            result += code[i]
            i += 1
        
        return result, i
    
    # ========== VARIABLE MANAGEMENT ==========
    
    def get_hasm_var_name(self, original_name: str) -> str:
        """Get HASM variable name in V01, V02 format"""
        if original_name not in self.hasm_var_mapping:
            self.hasm_var_counter += 1
            self.hasm_var_mapping[original_name] = f"V{self.hasm_var_counter:02d}"
        return self.hasm_var_mapping[original_name]
    
    def get_c_var_name(self, original_name: str) -> str:
        """Get C variable name in L01, L02 format"""
        if original_name not in self.c_var_mapping:
            self.c_var_counter += 1
            self.c_var_mapping[original_name] = f"L{self.c_var_counter:02d}"
        return self.c_var_mapping[original_name]
    
    def generate_hasm_variable_data_section(self, hasm_vars: Dict[str, HASMVariable]) -> List[str]:
        """Generate data section for HASM variables"""
        data_lines = []
        
        for name, var in hasm_vars.items():
            var_name = self.get_hasm_var_name(name)
            if var.var_type == 'string':
                # Remove quotes from string value
                clean_value = var.value.strip('"')
                data_lines.append(f'{var_name}:')
                data_lines.append(f'\tdb "{clean_value}", 0')
            elif var.var_type == 'int':
                data_lines.append(f'{var_name}:')
                data_lines.append(f'\tdd {var.value}')
            elif var.is_array:
                data_lines.append(f'{var_name}:')
                # Parse array initialization
                if var.value.startswith('{') and var.value.endswith('}'):
                    elements = var.value[1:-1].split(',')
                    elements = [e.strip() for e in elements]
                    data_lines.append(f'\tdd {", ".join(elements)}')
                else:
                    data_lines.append(f'\ttimes {var.array_size} dd 0')
        
        return data_lines
    
    def track_c_variable(self, var_name: str, var_type: str, size: int = 4) -> int:
        """Track a C variable and return its stack offset"""
        mapped_name = self.get_c_var_name(var_name)
        if mapped_name not in self.c_variables:
            self.stack_offset += size
            self.c_variables[mapped_name] = {
                'type': var_type,
                'size': size,
                'offset': self.stack_offset,
                'original_name': var_name
            }
        return self.c_variables[mapped_name]['offset']
    
    def get_variable_reference(self, var_name: str, hasm_vars: Dict[str, HASMVariable]) -> str:
        """Get the proper assembly reference for a variable"""
        # Check if it's a HASM variable
        if var_name in hasm_vars:
            hasm_var = hasm_vars[var_name]
            mapped_name = self.get_hasm_var_name(var_name)
            if hasm_var.var_type == 'int':
                return f'dword ptr [{mapped_name}]'
            elif hasm_var.var_type == 'string':
                return mapped_name
            elif hasm_var.is_array:
                return mapped_name
        
        # Check if it's a tracked C variable (check by mapped name)
        mapped_c_name = self.get_c_var_name(var_name)
        if mapped_c_name in self.c_variables:
            offset = self.c_variables[mapped_c_name]['offset']
            return f'dword ptr [rbp-{offset}]'
        
        # Default to immediate value if it's a number
        if var_name.isdigit():
            return var_name
        
        # Unknown variable - might be a local that needs tracking
        return f'[rbp-{self.track_c_variable(var_name, "int")}]'
    
    def generate_optimized_assembly(self, c_code: str, hasm_vars: Dict[str, HASMVariable], mode: str = "optimized") -> str:
        """Generate assembly using GCC compilation with external declarations"""
        # Always use GCC compilation for proper external linkage
        return self._compile_with_gcc(c_code, hasm_vars)
    
    def _generate_direct_assembly(self, c_code: str, hasm_vars: Dict[str, HASMVariable]) -> str:
        """Generate optimized assembly directly without full compilation"""
        lines = c_code.split('\n')
        asm_lines = []
        
        # Reset variable tracking for this C block
        self.c_variables = {}
        self.stack_offset = 0
        self.string_constants = []
        self.constant_counter = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle printf statements
            if line.startswith('printf('):
                result = self._generate_printf_asm(line, hasm_vars)
                if any("TODO" in r for r in result):
                    return f"TODO: Complex printf: {line}"
                asm_lines.extend(result)
            # Handle puts statements  
            elif line.startswith('puts('):
                result = self._generate_puts_asm(line, hasm_vars)
                if any("TODO" in r for r in result):
                    return f"TODO: Complex puts: {line}"
                asm_lines.extend(result)
            # Handle variable declarations
            elif re.match(r'^\w+\s+\w+\s*=', line):
                result = self._generate_var_decl_asm(line, hasm_vars)
                if any("TODO" in r for r in result):
                    return f"TODO: Complex declaration: {line}"
                asm_lines.extend(result)
            # Handle assignments
            elif re.match(r'^\w+\s*=', line):
                result = self._generate_assignment_asm(line, hasm_vars)
                if any("TODO" in r for r in result):
                    return f"TODO: Complex assignment: {line}"
                asm_lines.extend(result)
            else:
                # For complex statements, return error to trigger GCC fallback
                return f"TODO: Complex statement: {line}"
        
        # Build complete assembly with proper structure
        complete_asm = []
        
        # Data section for HASM variables
        hasm_data = self.generate_hasm_variable_data_section(hasm_vars)
        if hasm_data:
            complete_asm.extend(hasm_data)
            complete_asm.append('')
        
        # String constants
        if self.string_constants:
            complete_asm.extend(self.string_constants)
            complete_asm.append('')
        
        # Function prologue if we have C variables
        if self.c_variables:
            complete_asm.append('; Function prologue for C variables')
            complete_asm.append(f'\tsub rsp, {max(32, self.stack_offset + 16)}  ; Allocate stack space')
            complete_asm.append('')
        
        # Main assembly code
        complete_asm.extend(asm_lines)
        
        # Function epilogue if we have C variables
        if self.c_variables:
            complete_asm.append('')
            complete_asm.append('; Function epilogue')
            complete_asm.append(f'\tadd rsp, {max(32, self.stack_offset + 16)}  ; Restore stack')
        
        return '\n'.join(complete_asm) if complete_asm else '; (no assembly generated)'
    
    def _generate_printf_asm(self, line: str, hasm_vars: Dict[str, HASMVariable]) -> List[str]:
        """Generate assembly for printf statements"""
        # Extract format string and arguments
        match = re.search(r'printf\s*\(\s*"([^"]*)"(?:\s*,\s*(.+))?\s*\)', line)
        if not match:
            return [f'; Could not parse printf: {line}']
        
        format_str = match.group(1)
        args_str = match.group(2) if match.group(2) else ""
        
        # Create string constant with global counter
        const_label = f"str_const_{self.constant_counter}"
        self.constant_counter += 1  # Increment global counter
        self.string_constants.append(f'{const_label}:')
        self.string_constants.append(f'\tdb "{format_str}", 0')
        
        asm_lines = []
        
        # Handle arguments
        if args_str:
            args = [arg.strip() for arg in args_str.split(',')]
            
            # For Windows x64 calling convention: RCX, RDX, R8, R9
            registers = ['rcx', 'rdx', 'r8', 'r9']
            
            # First argument is format string
            asm_lines.append(f'\tlea rcx, [{const_label}]')
            
            # Handle additional arguments
            for i, arg in enumerate(args):
                if i + 1 < len(registers):
                    reg = registers[i + 1]
                    
                    # Get proper variable reference
                    if arg in hasm_vars:
                        var = hasm_vars[arg]
                        var_name = self.get_hasm_var_name(arg)
                        if var.var_type == 'int':
                            asm_lines.append(f'\tmov {reg}, [{var_name}]')
                        elif var.var_type == 'string':
                            asm_lines.append(f'\tlea {reg}, [{var_name}]')
                    elif arg in self.c_var_mapping:
                        # C variable
                        mapped_name = self.get_c_var_name(arg)
                        if mapped_name in self.c_variables:
                            offset = self.c_variables[mapped_name]['offset']
                            asm_lines.append(f'\tmov {reg}, dword ptr [rbp-{offset}]')
                    elif arg.isdigit():
                        # Numeric literal
                        asm_lines.append(f'\tmov {reg}, {arg}')
                    else:
                        # Expression or unknown - try to evaluate
                        evaluated = self._evaluate_simple_expression(arg, hasm_vars)
                        if evaluated.isdigit():
                            asm_lines.append(f'\tmov {reg}, {evaluated}')
                        else:
                            asm_lines.append(f'\tmov {reg}, {arg}  ; TODO: evaluate')
        else:
            # No arguments, just format string
            asm_lines.append(f'\tlea rcx, [{const_label}]')
        
        asm_lines.append('\tcall printf')
        return asm_lines
    
    def _generate_puts_asm(self, line: str, hasm_vars: Dict[str, HASMVariable]) -> List[str]:
        """Generate assembly for puts statements"""
        match = re.search(r'puts\s*\(\s*"([^"]*)"\s*\)', line)
        if not match:
            return [f'; Could not parse puts: {line}']
        
        text = match.group(1)
        const_label = f"str_const_{self.constant_counter}"
        self.constant_counter += 1  # Increment global counter
        self.string_constants.append(f'{const_label}:')
        self.string_constants.append(f'\tdb "{text}", 0')
        
        return [
            f'\tlea rcx, [{const_label}]',
            '\tcall puts'
        ]
    
    def _generate_var_decl_asm(self, line: str, hasm_vars: Dict[str, HASMVariable]) -> List[str]:
        """Generate assembly for variable declarations"""
        match = re.match(r'(\w+)\s+(\w+)\s*=\s*(.+);?', line)
        if not match:
            return [f'; Could not parse variable declaration: {line}']
        
        var_type = match.group(1)
        var_name = match.group(2)
        var_value = match.group(3)
        
        # Track the C variable
        offset = self.track_c_variable(var_name, var_type)
        mapped_name = self.get_c_var_name(var_name)
        
        asm_lines = [f'; {var_type} {mapped_name} = {var_value} (original: {var_name})']
        
        # Handle different assignment types
        if var_value in hasm_vars:
            # Assigning from HASM variable
            hasm_var = hasm_vars[var_value]
            hasm_var_name = self.get_hasm_var_name(var_value)
            if hasm_var.var_type == 'int':
                asm_lines.append(f'\tmov dword ptr [rbp-{offset}], [{hasm_var_name}]')
        elif var_value.replace('-', '').isdigit():
            # Numeric literal
            asm_lines.append(f'\tmov dword ptr [rbp-{offset}], {var_value}')
        else:
            # Expression - try to evaluate or return error to trigger GCC fallback
            evaluated = self._evaluate_simple_expression(var_value, hasm_vars)
            if evaluated.isdigit() or (evaluated.startswith('-') and evaluated[1:].isdigit()):
                asm_lines.append(f'\tmov dword ptr [rbp-{offset}], {evaluated}')
            else:
                return [f'TODO: Handle expression: {var_value}']
        
        return asm_lines
    
    def _generate_assignment_asm(self, line: str, hasm_vars: Dict[str, HASMVariable]) -> List[str]:
        """Generate assembly for assignments"""
        match = re.match(r'(\w+)\s*=\s*(.+);?', line)
        if not match:
            return [f'; Could not parse assignment: {line}']
        
        var_name = match.group(1)
        expression = match.group(2)
        
        # Ensure the variable is tracked (might be from a previous declaration)
        mapped_name = self.get_c_var_name(var_name)
        if mapped_name not in self.c_variables:
            self.track_c_variable(var_name)
        
        offset = self.c_variables[mapped_name]['offset']
        
        # Handle different assignment types
        if expression in hasm_vars:
            # Assigning from HASM variable
            hasm_var = hasm_vars[expression]
            hasm_var_name = self.get_hasm_var_name(expression)
            if hasm_var.var_type == 'int':
                return [f'\tmov dword ptr [rbp-{offset}], [{hasm_var_name}]']
        elif expression.replace('-', '').isdigit():
            # Numeric literal
            return [f'\tmov dword ptr [rbp-{offset}], {expression}']
        else:
            # Expression - try to evaluate
            evaluated = self._evaluate_simple_expression(expression, hasm_vars)
            if evaluated.isdigit() or (evaluated.startswith('-') and evaluated[1:].isdigit()):
                return [f'\tmov dword ptr [rbp-{offset}], {evaluated}']
            else:
                return [f'TODO: Handle expression: {mapped_name} = {expression}']
    
    def _evaluate_simple_expression(self, expression: str, hasm_vars: Dict[str, HASMVariable]) -> str:
        """Evaluate simple expressions for variable assignments"""
        expression = expression.strip()
        
        # Handle simple arithmetic with variables
        # Pattern: variable * number or number * variable
        mult_match = re.match(r'(\w+)\s*\*\s*(\d+)', expression)
        if mult_match:
            var_name = mult_match.group(1)
            multiplier = int(mult_match.group(2))
            if var_name in hasm_vars:
                var_value = int(hasm_vars[var_name].value)
                return str(var_value * multiplier)
        
        mult_match = re.match(r'(\d+)\s*\*\s*(\w+)', expression)
        if mult_match:
            multiplier = int(mult_match.group(1))
            var_name = mult_match.group(2)
            if var_name in hasm_vars:
                var_value = int(hasm_vars[var_name].value)
                return str(multiplier * var_value)
        
        # Handle addition
        add_match = re.match(r'(\w+)\s*\+\s*(\d+)', expression)
        if add_match:
            var_name = add_match.group(1)
            addend = int(add_match.group(2))
            if var_name in hasm_vars:
                var_value = int(hasm_vars[var_name].value)
                return str(var_value + addend)
        
        # Single variable reference
        if expression in hasm_vars:
            return hasm_vars[expression].value
        
        # If it's already a number, return as is
        if expression.replace('-', '').isdigit():
            return expression
        
        # Can't evaluate
        return expression

    def _get_local_offset(self, var_name: str) -> int:
        """Get stack offset for local variable (simplified)"""
        # This is a simplified approach - in practice you'd track variable locations
        return hash(var_name) % 100 + 4
    
    def _compile_with_gcc(self, c_code: str, hasm_vars: Dict[str, HASMVariable]) -> str:
        """Compile C code with HASM variable context using GCC"""
        try:
            # Create temporary C file with proper context
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as c_file:
                full_c_code = self._generate_complete_c_program(c_code, hasm_vars)
                c_file.write(full_c_code)
                c_file_path = c_file.name
            
            # Create temporary assembly file path
            asm_file_path = c_file_path.replace('.c', '.s')
            
            # Compile C to assembly
            compile_cmd = [
                "x86_64-w64-mingw32-gcc", "-S", "-masm=intel", "-O2", 
                "-fno-asynchronous-unwind-tables", "-fno-stack-protector",
                "-fomit-frame-pointer", "-fno-pic", "-fno-exceptions",
                c_file_path, "-o", asm_file_path
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return f"; Error compiling C code: {result.stderr}"
            
            # Read and clean the generated assembly
            with open(asm_file_path, 'r') as asm_file:
                asm_content = asm_file.read()
            
            # Clean up temporary files
            os.unlink(c_file_path)
            os.unlink(asm_file_path)
            
            # Extract clean assembly and convert to NASM syntax
            clean_asm = self._extract_clean_asm(asm_content)
            nasm_asm = self._convert_to_nasm_syntax(clean_asm, hasm_vars)
            
            return nasm_asm
            
        except Exception as e:
            return f"; Error: {str(e)}"
    
    def _convert_to_nasm_syntax(self, asm_content: str, hasm_vars: Dict[str, HASMVariable]) -> str:
        """Convert GCC Intel assembly to NASM-compatible syntax"""
        lines = asm_content.split('\n')
        converted_lines = []
        string_counter = 0
        
        for line in lines:
            converted_line = line
            
            # Convert .ascii to db
            if '.ascii' in converted_line:
                # Extract the string content
                match = re.search(r'\.ascii\s+"([^"]*)"', converted_line)
                if match:
                    string_content = match.group(1)
                    # Convert escape sequences properly
                    if '\\12' in string_content:
                        # Split at \12 and create proper db directive
                        parts = string_content.split('\\12')
                        if len(parts) == 2 and parts[1] == '\\0':
                            # Common pattern: "text\12\0" -> db "text", 10, 0
                            converted_line = re.sub(r'\.ascii\s+"[^"]*"', f'db "{parts[0]}", 10, 0', converted_line)
                        else:
                            # More complex pattern, just convert \12 to 10
                            string_content = string_content.replace('\\12', '", 10, "')
                            string_content = string_content.replace('\\0', '', 0)
                            converted_line = re.sub(r'\.ascii\s+"[^"]*"', f'db "{string_content}", 0', converted_line)
                    else:
                        # Simple string without escape sequences
                        converted_line = re.sub(r'\.ascii\s+"[^"]*"', f'db "{string_content}", 0', converted_line)
            
            # Replace .LC labels with unique identifiers
            if line.strip().endswith(':') and '.LC' in line:
                converted_line = f'str_const_{string_counter}:'
                string_counter += 1
            elif '.LC' in line and not line.strip().startswith(';') and not line.strip().endswith(':'):
                # This is a reference to a .LC label, we need to map it to the most recent str_const
                # For now, let's use a simple approach and map all .LC references to str_const_0, 1, 2, etc.
                if '.LC0' in line:
                    converted_line = line.replace('.LC0', f'str_const_{string_counter}')
                elif '.LC1' in line:
                    converted_line = line.replace('.LC1', f'str_const_1')
                elif '.LC2' in line:
                    converted_line = line.replace('.LC2', f'str_const_2')
                elif '.LC3' in line:
                    converted_line = line.replace('.LC3', f'str_const_3')
            
            # Convert QWORD PTR references to direct variable access
            # For integers, we want the value, for strings we want the address
            for var_name, var in hasm_vars.items():
                mapped_name = self.get_hasm_var_name(var_name)
                if var.var_type == 'int':
                    # For integers: mov rax, QWORD PTR .refptr.V01[rip] -> mov eax, [V01]
                    pattern = f'mov\\s+(\\w+),\\s+QWORD PTR \\.refptr\\.{mapped_name}\\[rip\\]'
                    replacement = f'mov \\1, [{mapped_name}]'
                    converted_line = re.sub(pattern, replacement, converted_line)
                elif var.var_type == 'string':
                    # For strings: mov r8, QWORD PTR .refptr.V02[rip] -> mov r8, V02
                    pattern = f'mov\\s+(\\w+),\\s+QWORD PTR \\.refptr\\.{mapped_name}\\[rip\\]'
                    replacement = f'mov \\1, {mapped_name}'
                    converted_line = re.sub(pattern, replacement, converted_line)
            
            # Convert remaining DWORD PTR
            converted_line = re.sub(r'DWORD PTR \[([^]]+)\]', r'dword [\1]', converted_line)
            
            # Handle RIP-relative addressing for labels
            converted_line = re.sub(r'(\w+)\[rip\]', r'\1', converted_line)
            
            # Remove .refptr references since we're using direct references
            if '.refptr.' in converted_line:
                continue
            
            # Remove __main calls as they're not needed
            if 'call\t__main' in converted_line or 'call __main' in converted_line:
                continue
            
            converted_lines.append(converted_line)
        
        # Add external declarations at the top
        external_decls = []
        if any('call\tprintf' in line or 'call printf' in line for line in converted_lines):
            external_decls.append('extern printf')
        
        if external_decls:
            # Find the section .text line and insert externals before it
            for i, line in enumerate(converted_lines):
                if line.strip() == 'section .text':
                    converted_lines = converted_lines[:i] + external_decls + [''] + converted_lines[i:]
                    break
        
        return '\n'.join(converted_lines)
    
    def _map_hasm_variables_in_asm(self, asm_content: str, hasm_vars: Dict[str, HASMVariable]) -> str:
        """Clean up the assembly and ensure proper external linkage"""
        lines = asm_content.split('\n')
        mapped_lines = []
        
        for line in lines:
            # Remove any duplicate external variable declarations from GCC
            if line.strip().startswith('.comm') or line.strip().startswith('.globl hasm_'):
                continue  # Skip these since we already have our data section
            
            mapped_lines.append(line)
        
        return '\n'.join(mapped_lines)
    
    def _generate_complete_c_program(self, c_code: str, hasm_vars: Dict[str, HASMVariable]) -> str:
        """Generate a complete C program with external HASM variable declarations"""
        includes = """#include <stdio.h>
#include <stdlib.h>
#include <string.h>"""
        
        # Generate external variable declarations instead of local ones
        var_declarations = []
        for name, var in hasm_vars.items():
            var_name = self.get_hasm_var_name(name)
            if var.is_array:
                if 'int' in var.var_type or var.value.strip().startswith('{'):
                    var_declarations.append(f"extern int {var_name}[{var.array_size}];")
                else:
                    var_declarations.append(f"extern char {var_name}[{var.array_size}];")
            elif var.var_type == 'string':
                var_declarations.append(f"extern char {var_name}[];")
            elif var.var_type == 'int':
                var_declarations.append(f"extern int {var_name};")
            else:
                var_declarations.append(f"extern int {var_name};  // assumed int")
        
        # Create aliases for the original variable names
        alias_declarations = []
        for name, var in hasm_vars.items():
            var_name = self.get_hasm_var_name(name)
            if var.var_type == 'string':
                alias_declarations.append(f"#define {name} {var_name}")
            elif var.var_type == 'int':
                alias_declarations.append(f"#define {name} {var_name}")
        
        var_decl_str = '\n'.join(var_declarations)
        alias_decl_str = '\n'.join(alias_declarations)
        
        # Handle preprocessor directives
        if c_code.strip().startswith('#'):
            return f"{includes}\n{c_code}"
        else:
            return f"""{includes}

// External HASM variable declarations
{var_decl_str}

// Variable aliases for C code
{alias_decl_str}

int main() {{
    // User C code
    {c_code}
    
    return 0;
}}"""
    
    def _extract_clean_asm(self, asm_content: str) -> str:
        """Extract clean assembly from GCC output"""
        lines = asm_content.split('\n')
        result_lines = []
        string_constants = []
        in_main = False
        
        # First pass: collect string constants
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('.LC') and ':' in stripped:
                label = stripped
                continue
            elif stripped.startswith('.ascii') or stripped.startswith('.string'):
                string_constants.append(f"{label}\n\t{stripped}")
        
        # Second pass: extract main function code
        for line in lines:
            stripped = line.strip()
            
            if 'main:' in stripped:
                in_main = True
                continue
            
            if in_main:
                # Skip prologue, epilogue, and unnecessary instructions
                if any(x in stripped for x in [
                    'push rbp', 'mov rbp, rsp', 'sub rsp', 'call __main',
                    'xor eax, eax', 'mov eax, 0', 'add rsp', 'pop rbp', 'ret', 'leave'
                ]):
                    continue
                
                # Skip directives
                if stripped.startswith('.') or stripped.startswith('#'):
                    continue
                
                if stripped and stripped != 'nop':
                    result_lines.append('\t' + stripped)
        
        # Combine string constants and code
        result = []
        if string_constants:
            result.append('; String constants')
            result.extend(string_constants)
            result.append('')
        
        if result_lines:
            result.extend(result_lines)
        else:
            result.append('; (optimized away)')
        
        return '\n'.join(result)
    
    # ========== C BLOCK PROCESSING ==========
    
    def _identify_c_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Identify consecutive C commands that should be combined into blocks"""
        blocks = []
        current_block = None
        
        for i, line in enumerate(lines):
            if self.is_c_command(line):
                c_code = line.strip()[2:].strip()
                
                # Check if this should start a new block or continue current block
                if current_block is None:
                    # Start new block
                    current_block = {
                        'start_line': i,
                        'end_line': i,
                        'original_lines': [c_code],
                        'combined_code': c_code,
                        'line_count': 1
                    }
                else:
                    # Check if this line is consecutive (allowing for whitespace/comments)
                    is_consecutive = True
                    for j in range(current_block['end_line'] + 1, i):
                        check_line = lines[j].strip()
                        if check_line and not check_line.startswith(';') and not self.is_c_command(check_line):
                            is_consecutive = False
                            break
                    
                    if is_consecutive:
                        # Continue current block
                        current_block['end_line'] = i
                        current_block['original_lines'].append(c_code)
                        current_block['combined_code'] += '\n' + c_code
                        current_block['line_count'] += 1
                    else:
                        # Finish current block and start new one
                        if current_block['line_count'] > 1:
                            blocks.append(current_block)
                        current_block = {
                            'start_line': i,
                            'end_line': i,
                            'original_lines': [c_code],
                            'combined_code': c_code,
                            'line_count': 1
                        }
            else:
                # Non-C line, finish current block if it exists
                if current_block and current_block['line_count'] > 1:
                    blocks.append(current_block)
                    current_block = None
        
        # Don't forget the last block
        if current_block and current_block['line_count'] > 1:
            blocks.append(current_block)
        
        return blocks
    
    # ========== MAIN PROCESSING METHODS ==========
    
    def process_asm_file(self, input_file: str, output_file: str, mode: str = "optimized", target_platform: str = "windows") -> bool:
        """Process an ASM file with C code generation"""
        try:
            # Reset counters for each new file but preserve constant counter logic
            self.hasm_var_counter = 0
            self.c_var_counter = 0
            self.hasm_var_mapping = {}
            self.c_var_mapping = {}
            self.string_constants = []
            self.data_section = []
            self.constant_counter = 0  # Reset string constant counter for new file
            self.variable_counter = 0
            self.stack_offset = 0
            
            with open(input_file, 'r') as f:
                content = f.read()
            
            # Parse HASM variables first
            hasm_vars = self.parse_hasm_variables(content)
            
            # Process the content
            processed_content = self._process_content(content, hasm_vars, mode, target_platform)
            
            # Write to output file
            with open(output_file, 'w') as f:
                f.write(processed_content)
            
            print(f"Successfully processed {input_file} -> {output_file}")
            print(f"Found {len(hasm_vars)} HASM variables: {list(hasm_vars.keys())}")
            print(f"  Mapped to: {list(self.hasm_var_mapping.values())}")
            if self.c_var_mapping:
                print(f"C variables: {list(self.c_var_mapping.keys())}")
                print(f"  Mapped to: {list(self.c_var_mapping.values())}")
            print(f"Generated {self.constant_counter} string constants")
            print(f"Mode: {mode}")
            return True
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return False
    
    def _process_content(self, content: str, hasm_vars: Dict[str, HASMVariable], mode: str, target_platform: str) -> str:
        """Process content with C code generation"""
        lines = content.split('\n')
        processed_lines = []
        data_section_lines = []
        bss_section_lines = []
        text_section_lines = []
        
        # Add HASM variables to data section first
        hasm_data = self.generate_hasm_variable_data_section(hasm_vars)
        data_section_lines.extend(hasm_data)
        
        # First pass: identify C code blocks that should be combined
        c_blocks = self._identify_c_blocks(lines)
        
        # Reset C variables tracking for this file but maintain counter across blocks
        self.c_variables = {}
        self.stack_offset = 0
        
        i = 0
        block_number = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this line starts a C block
            block_info = None
            for block in c_blocks:
                if block['start_line'] == i:
                    block_info = block
                    break
            
            if block_info:
                block_number += 1
                # Process the entire C block
                combined_c_code = block_info['combined_code']
                
                # Add comment showing original C code to text section
                if block_info['line_count'] > 1:
                    text_section_lines.append(f"; C code block {block_number} ({block_info['line_count']} lines):")
                    for c_line in block_info['original_lines']:
                        text_section_lines.append(f"; {c_line}")
                else:
                    text_section_lines.append(f"; C code block {block_number}: {combined_c_code}")
                
                # Generate assembly
                asm_code = self.generate_optimized_assembly(combined_c_code, hasm_vars, mode)
                
                # Parse the assembly code to separate data and text sections
                self._parse_assembly_sections(asm_code, data_section_lines, bss_section_lines, text_section_lines)
                
                # Skip the lines that were part of this block
                i = block_info['end_line'] + 1
            elif self.is_c_command(line):
                # Single C command (not part of a block)
                block_number += 1
                stripped = line.strip()
                c_code = stripped[2:].strip()  # Remove %!
                
                # Add comment showing original C code to text section
                text_section_lines.append(f"; C code block {block_number}: {c_code}")
                
                # Generate assembly
                asm_code = self.generate_optimized_assembly(c_code, hasm_vars, mode)
                
                # Parse the assembly code to separate data and text sections
                self._parse_assembly_sections(asm_code, data_section_lines, bss_section_lines, text_section_lines)
                
                i += 1
            else:
                # Keep original line (HASM commands, assembly, comments)
                # But skip %var declarations since they'll be converted to data sections
                if not line.strip().startswith('%var '):
                    text_section_lines.append(line)
                i += 1
        
        # Build the final output with proper sections
        final_output = []
        
        # Data section
        if data_section_lines:
            final_output.append('section .data')
            final_output.extend(data_section_lines)
            final_output.append('')
        
        # BSS section
        if bss_section_lines:
            final_output.append('section .bss')
            final_output.extend(bss_section_lines)
            final_output.append('')
        
        # Text section
        if text_section_lines:
            final_output.append('section .text')
            final_output.extend(text_section_lines)
        
        return '\n'.join(final_output)
    
    def _parse_assembly_sections(self, asm_code: str, data_section: List[str], bss_section: List[str], text_section: List[str]):
        """Parse assembly code and separate into data, bss, and text sections"""
        if not asm_code or asm_code.startswith("; Error") or asm_code.startswith("TODO"):
            if asm_code:
                text_section.append(asm_code)
            return
        
        lines = asm_code.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines and comments for section detection
            if not line_stripped or line_stripped.startswith(';'):
                text_section.append(line)
                continue
            
            # Data section items: variable definitions, string constants, refptr sections
            if (line_stripped.endswith(':') and 
                (line_stripped.startswith('hasm_') or 
                 line_stripped.startswith('str_') or 
                 line_stripped.startswith('.LC') or
                 line_stripped.startswith('.refptr.'))):
                data_section.append(line)
            elif (line_stripped.startswith('dd ') or 
                  line_stripped.startswith('db ') or 
                  line_stripped.startswith('dw ') or 
                  line_stripped.startswith('dq ') or
                  line_stripped.startswith('.ascii ') or
                  line_stripped.startswith('.quad ') or
                  line_stripped.startswith('\tdd ') or 
                  line_stripped.startswith('\tdb ') or 
                  line_stripped.startswith('\tdw ') or 
                  line_stripped.startswith('\tdq ') or
                  line_stripped.startswith('\t.ascii ') or
                  line_stripped.startswith('\t.quad ')):
                data_section.append(line)
            # BSS section items: uninitialized data (for future use)
            elif line_stripped.startswith('resb ') or line_stripped.startswith('resd '):
                bss_section.append(line)
            # Everything else goes to text section
            else:
                text_section.append(line)

    # ========== ANALYSIS AND UTILITY METHODS ==========
    
    def analyze_file(self, filename: str) -> Dict[str, Any]:
        """Analyze HASM variables and C code usage in a file"""
        with open(filename, 'r') as f:
            content = f.read()
        
        hasm_vars = self.parse_hasm_variables(content)
        c_commands = self.extract_c_commands(content)
        variable_usage = {var_name: [] for var_name in hasm_vars.keys()}
        
        # Check which HASM variables are used in C code
        for cmd in c_commands:
            for var_name in hasm_vars.keys():
                if var_name in cmd['c_code']:
                    variable_usage[var_name].append(cmd['line_number'])
        
        return {
            'hasm_variables': hasm_vars,
            'c_commands': c_commands,
            'variable_usage': variable_usage
        }
    
    def print_analysis(self, analysis: Dict[str, Any]):
        """Print analysis results"""
        print("=== HASM VARIABLE ANALYSIS ===")
        print(f"Found {len(analysis['hasm_variables'])} HASM variables:")
        for name, var in analysis['hasm_variables'].items():
            print(f"  {name}: {var.var_type} = {var.value} (line {var.line_number})")
        
        print(f"\nFound {len(analysis['c_commands'])} C commands:")
        for cmd in analysis['c_commands']:
            print(f"  Line {cmd['line_number']}: {cmd['c_code']}")
        
        print("\nVariable usage in C code:")
        for var_name, usage_lines in analysis['variable_usage'].items():
            if usage_lines:
                print(f"  {var_name}: used in lines {usage_lines}")
            else:
                print(f"  {var_name}: not used in C code")
    
    def get_c_commands_by_type(self, commands: List[Dict[str, Any]], construct_type: str) -> List[Dict[str, Any]]:
        """Get all C commands of a specific construct type"""
        return [cmd for cmd in commands if cmd['construct_type'] == construct_type]
    
    def get_all_construct_types(self, commands: List[Dict[str, Any]]) -> List[str]:
        """Get all unique construct types"""
        return list(set(cmd['construct_type'] for cmd in commands))

# Command line interface
if __name__ == "__main__":
    import sys
    
    codegen = CCodeGenerator()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "compile":
            if len(sys.argv) < 4:
                print("Usage: python3 c_codegen.py compile <input_file> <output_file> [mode] [target_platform]")
                print("Modes: optimized (default), direct, gcc")
                print("Example: python3 c_codegen.py compile test.asm test_compiled.asm optimized windows")
                sys.exit(1)
            
            input_file = sys.argv[2]
            output_file = sys.argv[3]
            mode = sys.argv[4] if len(sys.argv) > 4 else "optimized"
            target_platform = sys.argv[5] if len(sys.argv) > 5 else "windows"
            
            success = codegen.process_asm_file(input_file, output_file, mode, target_platform)
            sys.exit(0 if success else 1)
        
        elif sys.argv[1] == "analyze":
            if len(sys.argv) < 3:
                print("Usage: python3 c_codegen.py analyze <file>")
                sys.exit(1)
            
            filename = sys.argv[2]
            analysis = codegen.analyze_file(filename)
            codegen.print_analysis(analysis)
            sys.exit(0)