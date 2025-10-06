#!/usr/bin/env python3

import re
import os
import tempfile
import subprocess
from typing import List, Tuple, Optional, Dict, Any
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

class CLexer:
    def __init__(self):
        self.tokens = []
        self.current_line = 1
        self.current_column = 1
        
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
            ('++', 'INCREMENT'),
            ('--', 'DECREMENT'),
            ('==', 'EQUALS'),
            ('!=', 'NOT_EQUALS'),
            ('<=', 'LESS_EQUAL'),
            ('>=', 'GREATER_EQUAL'),
            ('&&', 'LOGICAL_AND'),
            ('||', 'LOGICAL_OR'),
            ('<<', 'LEFT_SHIFT'),
            ('>>', 'RIGHT_SHIFT'),
            ('->', 'ARROW'),
            ('+=', 'PLUS_ASSIGN'),
            ('-=', 'MINUS_ASSIGN'),
            ('*=', 'MULT_ASSIGN'),
            ('/=', 'DIV_ASSIGN'),
            ('%=', 'MOD_ASSIGN'),
            ('&=', 'AND_ASSIGN'),
            ('|=', 'OR_ASSIGN'),
            ('^=', 'XOR_ASSIGN'),
            ('<', 'LESS_THAN'),
            ('>', 'GREATER_THAN'),
            ('=', 'ASSIGN'),
            ('+', 'PLUS'),
            ('-', 'MINUS'),
            ('*', 'MULTIPLY'),
            ('/', 'DIVIDE'),
            ('%', 'MODULO'),
            ('&', 'BITWISE_AND'),
            ('|', 'BITWISE_OR'),
            ('^', 'BITWISE_XOR'),
            ('~', 'BITWISE_NOT'),
            ('!', 'LOGICAL_NOT'),
            ('?', 'TERNARY'),
            (':', 'COLON'),
            (';', 'SEMICOLON'),
            (',', 'COMMA'),
            ('.', 'DOT'),
        ]
        
        # C punctuation
        self.c_punctuation = [
            ('(', 'LPAREN'),
            (')', 'RPAREN'),
            ('[', 'LBRACKET'),
            (']', 'RBRACKET'),
            ('{', 'LBRACE'),
            ('}', 'RBRACE'),
        ]
    
    def is_c_command(self, line: str) -> bool:
        """Check if a line contains a C command (starts with %!)"""
        stripped = line.strip()
        return stripped.startswith('%!')
    
    def compile_c_code_to_asm(self, c_code: str, target_platform: str = "windows") -> str:
        """Compile C code to assembly using GCC"""
        try:
            # Create temporary C file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as c_file:
                # Add necessary includes and wrapper for simple statements
                if not c_code.strip().startswith('#'):
                    # For simple C statements, wrap in a function
                    full_c_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void generated_code() {{
    {c_code}
}}
"""
                else:
                    # For preprocessor directives, just add them
                    full_c_code = c_code
                
                c_file.write(full_c_code)
                c_file_path = c_file.name
            
            # Create temporary assembly file path
            asm_file_path = c_file_path.replace('.c', '.s')
            
            # Choose compiler based on target platform
            if target_platform.lower() == "windows":
                compiler = "x86_64-w64-mingw32-gcc"
            else:
                compiler = "gcc"
            
            # Compile C to assembly
            compile_cmd = [
                compiler, "-S", "-masm=intel", "-O0",
                c_file_path, "-o", asm_file_path
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return f"; Error compiling C code: {result.stderr}"
            
            # Read the generated assembly
            with open(asm_file_path, 'r') as asm_file:
                asm_content = asm_file.read()
            
            # Clean up temporary files
            os.unlink(c_file_path)
            os.unlink(asm_file_path)
            
            # Extract the relevant assembly code (remove boilerplate)
            cleaned_asm = self._extract_relevant_asm(asm_content)
            return cleaned_asm
            
        except Exception as e:
            return f"; Error: {str(e)}"
    
    def _extract_relevant_asm(self, asm_content: str) -> str:
        """Extract relevant assembly code from GCC output"""
        lines = asm_content.split('\n')
        relevant_lines = []
        in_function = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Skip most directives except important ones
            if stripped.startswith('.') and not any(x in stripped for x in ['.text', '.data', '.bss']):
                if 'generated_code:' in stripped:
                    in_function = True
                continue
            
            # Look for function start
            if 'generated_code:' in stripped:
                in_function = True
                continue
            
            # Skip function prologue/epilogue
            if in_function:
                if any(x in stripped for x in ['push rbp', 'mov rbp, rsp', 'pop rbp', 'ret']):
                    if 'ret' in stripped:
                        break
                    continue
                
                # Add relevant instructions
                if stripped and not stripped.startswith('.'):
                    # Indent the assembly instruction
                    relevant_lines.append('\t' + stripped)
        
        return '\n'.join(relevant_lines) if relevant_lines else '; (no assembly generated)'
    
    def process_asm_file_with_c_compilation(self, input_file: str, output_file: str, target_platform: str = "windows") -> bool:
        """Process an ASM file, compile C commands, and generate new ASM file"""
        try:
            with open(input_file, 'r') as f:
                content = f.read()
            
            # Process the content
            processed_content = self._process_content_with_c_compilation(content, target_platform)
            
            # Write to output file
            with open(output_file, 'w') as f:
                f.write(processed_content)
            
            print(f"Successfully processed {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return False
    
    def _process_content_with_c_compilation(self, content: str, target_platform: str) -> str:
        """Process content, replacing %! commands with compiled assembly"""
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            if self.is_c_command(line):
                # Extract C code
                stripped = line.strip()
                c_code = stripped[2:].strip()  # Remove %!
                
                # Add comment showing original C code
                processed_lines.append(f"; Original C code: {c_code}")
                
                # Compile C code to assembly
                asm_code = self.compile_c_code_to_asm(c_code, target_platform)
                
                # Add the generated assembly
                if asm_code.strip():
                    processed_lines.append(asm_code)
                else:
                    processed_lines.append("; (empty assembly generated)")
            else:
                # Keep original line (HASM commands, assembly, comments)
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
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
    
    def extract_c_commands_from_file(self, filename: str) -> List[Dict[str, Any]]:
        """Extract C commands from a file"""
        try:
            with open(filename, 'r') as f:
                text = f.read()
            return self.extract_c_commands(text)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' not found")
    
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
    
    def get_c_commands_by_type(self, commands: List[Dict[str, Any]], construct_type: str) -> List[Dict[str, Any]]:
        """Get all C commands of a specific construct type"""
        return [cmd for cmd in commands if cmd['construct_type'] == construct_type]
    
    def get_all_construct_types(self, commands: List[Dict[str, Any]]) -> List[str]:
        """Get all unique construct types"""
        return list(set(cmd['construct_type'] for cmd in commands))
    
    def print_c_commands_summary(self, commands: List[Dict[str, Any]]):
        """Print a summary of all C commands found"""
        if not commands:
            print("No C commands found.")
            return
        
        print(f"Found {len(commands)} C command(s):")
        print("-" * 50)
        
        for i, cmd in enumerate(commands, 1):
            print(f"{i}. Line {cmd['line_number']}: %!{cmd['command_name']}")
            print(f"   Original: {cmd['original_line'].strip()}")
            print(f"   C Code: {cmd['c_code']}")
            print(f"   Type: {cmd['construct_type']}")
            
            if cmd['construct_details']:
                print("   Details:")
                for key, value in cmd['construct_details'].items():
                    print(f"     {key}: {value}")
            
            print(f"   Tokens: {len(cmd['tokens'])}")
            print()

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    clexer = CLexer()
    
    # Command line interface
    if len(sys.argv) > 1:
        if sys.argv[1] == "compile":
            if len(sys.argv) < 4:
                print("Usage: python3 clexer.py compile <input_file> <output_file> [target_platform]")
                print("Example: python3 clexer.py compile test.asm test_compiled.asm windows")
                sys.exit(1)
            
            input_file = sys.argv[2]
            output_file = sys.argv[3]
            target_platform = sys.argv[4] if len(sys.argv) > 4 else "windows"
            
            success = clexer.process_asm_file_with_c_compilation(input_file, output_file, target_platform)
            sys.exit(0 if success else 1)
        
        elif sys.argv[1] == "analyze":
            if len(sys.argv) < 3:
                print("Usage: python3 clexer.py analyze <file>")
                sys.exit(1)
            
            filename = sys.argv[2]
            c_commands = clexer.extract_c_commands_from_file(filename)
            clexer.print_c_commands_summary(c_commands)
            sys.exit(0)
    
    # Default: run test
    else:
        print("CLexer - C Code to Assembly Compiler")
        print("Usage:")
        print("  python3 clexer.py compile <input.asm> <output.asm> [platform]")
        print("  python3 clexer.py analyze <file.asm>")
        print()
        print("Example:")
        print("  python3 clexer.py compile test/c_commands_test.asm test/compiled.asm windows")