#!/usr/bin/env python3

"""
Pretty printer for CASM - Context-aware assembly cleanup
Intelligently organizes combined C and assembly code while preserving structure
"""

import re
from typing import List, Dict, Tuple, Optional
from .ast_nodes import *
from ..utils.colors import print_info, print_success, print_warning

class CASMPrettyPrinter:
    """Context-aware pretty printer for CASM assembly output"""
    
    def __init__(self):
        self.variable_map = {}  # CASM var -> assembly label mapping
        self.c_variables = {}   # C variables discovered
        self.string_constants = {}  # String constants found
        self.current_context = None  # Current control flow context
        self.indent_level = 0
        self.base_indent = "    "
        self.string_counter = 0  # Counter for string labels
        
    def prettify(self, assembly_code: str, ast: ProgramNode, variable_map: Dict, c_assembly_segments: Dict = None) -> str:
        """Main entry point - prettify the assembly code with context awareness"""
        self.variable_map = variable_map
        self.c_assembly_segments = c_assembly_segments or {}
        
        # Reset counters for consistent labeling
        self.string_counter = 0
        self.string_label_map = {}  # Map original labels to sequential ones
        self.collected_strings = {}  # Map string_label -> content
        
        # Pre-process AST to collect all strings
        self._collect_all_strings(ast)
        
        self._extract_metadata(assembly_code)
        
        print_info("Prettifying assembly with context awareness...")
        
        # Parse the assembly into structured sections
        sections = self._parse_assembly_sections(assembly_code)
        
        # Rebuild with proper structure and context
        prettified = self._rebuild_with_context(sections, ast)
        
        # Final cleanup and formatting
        final_output = self._final_cleanup(prettified)
        
        print_success("Assembly prettified with context preservation")
        return final_output
    
    def _collect_all_strings(self, ast: ProgramNode):
        """Pre-process AST to collect all string literals for consistent labeling"""
        def visit_statements(statements):
            for stmt in statements:
                if isinstance(stmt, PrintlnNode):
                    # Extract message content
                    message = stmt.message.strip()
                    if message.startswith('"') and message.endswith('"'):
                        message = message[1:-1]  # Remove quotes
                    
                    # Generate consistent string label
                    self.string_counter += 1
                    string_label = f"str_{self.string_counter}"
                    self.collected_strings[string_label] = f'"{message}", 10, 0'
                
                elif isinstance(stmt, IfNode):
                    visit_statements(stmt.if_body)
                    if stmt.else_body:
                        visit_statements(stmt.else_body)
                
                elif isinstance(stmt, WhileNode):
                    visit_statements(stmt.body)
                
                elif isinstance(stmt, ForNode):
                    visit_statements(stmt.body)
        
        visit_statements(ast.statements)
        print_info(f"Collected {len(self.collected_strings)} string literals")
    
    def _extract_metadata(self, assembly_code: str):
        """Extract metadata from the assembly code"""
        lines = assembly_code.split('\n')
        
        for line in lines:
            # Extract string constants
            if '.ascii' in line or 'db "' in line:
                self._extract_string_constant(line)
            
            # Extract C variable references
            if 'extern' in line and 'var_' in line:
                self._extract_variable_reference(line)
    
    def _extract_string_constant(self, line: str):
        """Extract string constants and their labels"""
        # Match patterns like: .LC0: .ascii "Hello World\0"
        match = re.search(r'(\.LC\d+|str_\d+).*["\']([^"\']*)["\']', line)
        if match:
            label, text = match.groups()
            self.string_constants[label] = text
    
    def _extract_variable_reference(self, line: str):
        """Extract variable references from extern declarations"""
        # Match: extern int var_number;
        match = re.search(r'extern\s+(\w+)\s+(var_\w+)', line)
        if match:
            var_type, var_label = match.groups()
            # Find original CASM variable name
            for casm_var, label_info in self.variable_map.items():
                if isinstance(label_info, dict) and label_info.get('label') == var_label:
                    self.c_variables[var_label] = {
                        'casm_name': casm_var,
                        'type': var_type,
                        'label': var_label
                    }
                    break
    
    def _parse_assembly_sections(self, assembly_code: str) -> Dict:
        """Parse assembly into logical sections"""
        lines = assembly_code.split('\n')
        sections = {
            'header': [],
            'externals': [],
            'data': [],
            'text': [],
            'gcc_blocks': []
        }
        
        current_section = 'header'
        current_gcc_block = None
        
        for line in lines:
            stripped = line.strip()
            
            # Detect GCC assembly blocks
            if '=== RAW GCC ASSEMBLY OUTPUT ===' in line:
                current_gcc_block = {
                    'start_line': line,
                    'content': [],
                    'command': '',
                    'context': self.current_context
                }
                continue
            elif '=== END GCC ASSEMBLY OUTPUT ===' in line:
                if current_gcc_block:
                    current_gcc_block['end_line'] = line
                    sections['gcc_blocks'].append(current_gcc_block)
                    current_gcc_block = None
                continue
            elif current_gcc_block is not None:
                # Inside GCC block
                if line.startswith('; x86_64-w64-mingw32-gcc'):
                    current_gcc_block['command'] = line.strip()[2:]  # Remove '; '
                else:
                    current_gcc_block['content'].append(line)
                continue
            
            # Regular assembly parsing
            if stripped.startswith('extern'):
                sections['externals'].append(line)
            elif stripped.startswith('section .data') or current_section == 'data':
                if stripped.startswith('section .text'):
                    current_section = 'text'
                    sections['text'].append(line)
                else:
                    sections['data'].append(line)
                    if stripped.startswith('section .data'):
                        current_section = 'data'
            elif stripped.startswith('section .text') or current_section == 'text':
                sections['text'].append(line)
                current_section = 'text'
            else:
                sections[current_section].append(line)
        
        return sections
    
    def _rebuild_with_context(self, sections: Dict, ast: ProgramNode) -> str:
        """Rebuild assembly with proper context and structure"""
        output = []
        
        # Header section
        output.extend(self._format_header(sections['header']))
        output.append("")
        
        # External declarations - cleaned up
        if sections['externals']:
            output.append("; External function declarations")
            for line in sections['externals']:
                if line.strip():
                    output.append(line)
            output.append("")
        
        # Data section - organized
        output.extend(self._format_data_section(sections['data']))
        output.append("")
        
        # Text section with context-aware C code placement
        output.extend(self._format_text_section_with_context(sections, ast))
        
        return '\n'.join(output)
    
    def _format_header(self, header_lines: List[str]) -> List[str]:
        """Format the header section"""
        formatted = []
        for line in header_lines:
            if line.strip():
                formatted.append(line)
        return formatted
    
    def _format_data_section(self, data_lines: List[str]) -> List[str]:
        """Format the data section with proper organization"""
        formatted = []
        in_data_section = False
        additional_data = []
        
        for line in data_lines:
            stripped = line.strip()
            
            if stripped.startswith('section .data'):
                formatted.append("section .data")
                in_data_section = True
                continue
            
            if in_data_section and stripped:
                # Clean up data declarations
                if stripped.startswith('str_') or stripped.startswith('var_'):
                    # Add context comment for variables
                    if stripped.startswith('var_'):
                        var_label = stripped.split()[0]
                        if var_label in self.c_variables:
                            casm_name = self.c_variables[var_label]['casm_name']
                            formatted.append(f"    ; CASM variable: {casm_name}")
                    formatted.append(f"    {stripped}")
                else:
                    formatted.append(line)
        
        # Add additional data items that might be needed (string constants, loop counters)
        self._add_generated_data_items(formatted)
        
        # Add all collected strings at the end of data section
        for string_label, content in self.collected_strings.items():
            formatted.append(f"    {string_label} db {content}")
        
        return formatted
    
    def _add_generated_data_items(self, formatted: List[str]):
        """Add generated data items like string constants and loop counters"""
        # This will be populated when we encounter println, scanf, and for loops
        # For now, we'll add placeholders that will be filled during AST processing
        pass
    
    def _format_text_section_with_context(self, sections: Dict, ast: ProgramNode) -> List[str]:
        """Format text section with context-aware C code placement"""
        formatted = []
        formatted.append("section .text")
        formatted.append("global main")
        formatted.append("")
        formatted.append("main:")
        formatted.append("    push rbp")
        formatted.append("    mov rbp, rsp")
        formatted.append("    sub rsp, 32  ; Shadow space for Windows x64")
        formatted.append("")
        
        # Process AST to maintain structure
        gcc_block_index = 0
        formatted.extend(self._process_ast_with_gcc_blocks(ast, sections['gcc_blocks'], gcc_block_index))
        
        # Add exit code
        formatted.append("")
        formatted.append("    ; Program exit")
        formatted.append("    mov rax, 0")
        formatted.append("    add rsp, 32")
        formatted.append("    pop rbp")
        formatted.append("    ret")
        
        return formatted
    
    def _process_ast_with_gcc_blocks(self, node: ASTNode, gcc_blocks: List[Dict], block_index: int) -> List[str]:
        """Process AST nodes and insert GCC blocks in proper context"""
        output = []
        current_block_index = block_index
        
        if isinstance(node, ProgramNode):
            for stmt in node.statements:
                stmt_output, current_block_index = self._process_statement_with_context(
                    stmt, gcc_blocks, current_block_index
                )
                output.extend(stmt_output)
        
        return output
    
    def _process_statement_with_context(self, stmt: ASTNode, gcc_blocks: List[Dict], block_index: int) -> Tuple[List[str], int]:
        """Process individual statement with proper context"""
        output = []
        current_index = block_index
        
        if isinstance(stmt, VarDeclarationNode):
            output.append(f"    ; Variable declaration: {stmt.name} = {stmt.value}")
            # Generate actual variable initialization assembly
            var_label = f"var_{stmt.name}"
            try:
                # Try to parse as integer
                value = int(stmt.value)
                output.append(f"    mov dword [rel {var_label}], {value}")
            except ValueError:
                # Handle variable references or expressions
                if stmt.value in self.variable_map:
                    src_label = f"var_{stmt.value}"
                    output.append(f"    mov eax, dword [rel {src_label}]")
                    output.append(f"    mov dword [rel {var_label}], eax")
                else:
                    output.append(f"    ; Complex expression: {stmt.value}")
            
        elif isinstance(stmt, IfNode):
            # Generate real if assembly
            end_label = f"if_end_{id(stmt)}"
            else_label = f"else_{id(stmt)}" if stmt.else_body else None
            
            output.append(f"    ; if {stmt.condition}")
            
            # Generate condition evaluation assembly
            condition_asm = self._generate_condition_assembly(stmt.condition)
            output.extend(condition_asm)
            
            if else_label:
                output.append(f"    je {else_label}")
            else:
                output.append(f"    je {end_label}")
            
            self.indent_level += 1
            
            # Process if body
            for if_stmt in stmt.if_body:
                if_output, current_index = self._process_statement_with_context(
                    if_stmt, gcc_blocks, current_index
                )
                output.extend(if_output)
            
            # Process else body if exists
            if stmt.else_body:
                output.append(f"    jmp {end_label}")
                output.append(f"{else_label}:")
                
                for else_stmt in stmt.else_body:
                    else_output, current_index = self._process_statement_with_context(
                        else_stmt, gcc_blocks, current_index
                    )
                    output.extend(else_output)
            
            output.append(f"{end_label}:")
            self.indent_level -= 1
            
        elif isinstance(stmt, WhileNode):
            # Generate real while loop assembly
            start_label = f"while_start_{id(stmt)}"
            end_label = f"while_end_{id(stmt)}"
            
            output.append(f"    ; while {stmt.condition}")
            output.append(f"{start_label}:")
            
            # Generate condition evaluation assembly
            condition_asm = self._generate_condition_assembly(stmt.condition)
            output.extend(condition_asm)
            output.append(f"    je {end_label}")
            
            self.indent_level += 1
            
            for while_stmt in stmt.body:
                while_output, current_index = self._process_statement_with_context(
                    while_stmt, gcc_blocks, current_index
                )
                output.extend(while_output)
            
            output.append(f"    jmp {start_label}")
            output.append(f"{end_label}:")
            self.indent_level -= 1
            
        elif isinstance(stmt, ForNode):
            # Generate real for loop assembly
            start_label = f"for_start_{id(stmt)}"
            end_label = f"for_end_{id(stmt)}"
            counter_var = f"for_counter_{id(stmt)}"
            
            output.append(f"    ; for {stmt.variable} in range({stmt.count})")
            
            # Initialize counter
            output.append(f"    mov dword [rel {counter_var}], 0")
            output.append(f"{start_label}:")
            
            # Check condition
            output.append(f"    mov eax, dword [rel {counter_var}]")
            output.append(f"    cmp eax, {stmt.count}")
            output.append(f"    jge {end_label}")
            
            self.indent_level += 1
            
            for for_stmt in stmt.body:
                for_output, current_index = self._process_statement_with_context(
                    for_stmt, gcc_blocks, current_index
                )
                output.extend(for_output)
            
            # Increment counter
            output.append(f"    inc dword [rel {counter_var}]")
            output.append(f"    jmp {start_label}")
            output.append(f"{end_label}:")
            self.indent_level -= 1
            
        elif isinstance(stmt, PrintlnNode):
            # Generate real printf assembly
            output.append(f"    ; println {stmt.message}")
            
            # Clean the message
            message = stmt.message.strip()
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]  # Remove quotes
            
            # Find the corresponding string label from pre-collected strings
            string_label = None
            for label, content in self.collected_strings.items():
                if f'"{message}", 10, 0' == content:
                    string_label = label
                    break
            
            if not string_label:
                # Fallback: generate new label
                self.string_counter += 1
                string_label = f"str_{self.string_counter}"
            
            # Add printf call assembly (Windows x64 calling convention)
            output.append(f"    lea rcx, [rel {string_label}]")
            output.append(f"    call printf")
            
            # Note: String would need to be added to data section
            
        elif isinstance(stmt, ScanfNode):
            # Generate real scanf assembly
            output.append(f"    ; scanf {stmt.format_string} -> {stmt.variable}")
            
            # Clean format string
            format_str = stmt.format_string.strip()
            if format_str.startswith('"') and format_str.endswith('"'):
                format_str = format_str[1:-1]
            
            # Generate labels
            format_label = f"fmt_scanf_{id(stmt)}"
            var_label = f"var_{stmt.variable}"
            
            # Generate scanf call assembly (Windows x64 calling convention)
            output.append(f"    lea rcx, [rel {format_label}]")
            output.append(f"    lea rdx, [rel {var_label}]")
            output.append(f"    call scanf")
            
        elif isinstance(stmt, CCodeBlockNode):
            # Handle the new marker-based C compilation system
            indent = self.base_indent * (self.indent_level + 1)
            output.append(f"{indent}; === C CODE BLOCK ===")
            output.append(f"{indent}; Original C: {stmt.c_code}")
            output.append(f"{indent}; Compiled with combined GCC compilation")
            
            # Get the actual compiled assembly for this block
            block_id = getattr(stmt, '_block_id', None)
            if block_id is not None:
                marker_key = f"CASM_BLOCK_{block_id}"
                if marker_key in self.c_assembly_segments:
                    assembly_code = self.c_assembly_segments[marker_key]
                    if assembly_code.strip():
                        output.append(f"{indent}; Generated assembly:")
                        for asm_line in assembly_code.split('\n'):
                            if asm_line.strip():
                                output.append(f"{indent}{asm_line}")
                    else:
                        output.append(f"{indent}; (Empty assembly block)")
                else:
                    output.append(f"{indent}; (Assembly not found for {marker_key})")
            else:
                output.append(f"{indent}; (No block ID available)")
            
            output.append(f"{indent}; === END C CODE BLOCK ===")
        
        elif isinstance(stmt, ExternDirectiveNode):
            output.append(f"    ; extern {stmt.header_name}")
            
        elif isinstance(stmt, AssemblyNode):
            indent = self.base_indent * (self.indent_level + 1)
            output.append(f"{indent}{stmt.code}")
            
        elif isinstance(stmt, CommentNode):
            output.append(f"    {stmt.text}")
        
        return output, current_index
    
    def _clean_gcc_block(self, gcc_content: List[str], indent_level: int) -> List[str]:
        """Clean and properly indent GCC assembly block"""
        cleaned = []
        base_indent = self.base_indent * indent_level
        asm_indent = self.base_indent * (indent_level + 1)
        
        in_function = False
        
        for line in gcc_content:
            stripped = line.strip()
            
            # Skip metadata and directives we don't need
            if any(skip in stripped for skip in ['.file', '.ident', '.def', '.scl', '.type', '.seh_']):
                continue
            
            # Handle section declarations
            if stripped.startswith('.text') or stripped.startswith('.section'):
                cleaned.append(f"{base_indent}; {stripped}")
                continue
            
            # Handle labels
            if stripped.endswith(':') and not stripped.startswith('.'):
                if 'casm_c_block' in stripped:
                    in_function = True
                    cleaned.append(f"{base_indent}; C function start")
                else:
                    cleaned.append(f"{base_indent}{stripped}")
                continue
            
            # Handle instructions inside function
            if in_function and stripped:
                # Clean up the instruction
                clean_instr = self._clean_instruction(stripped)
                if clean_instr:
                    cleaned.append(f"{asm_indent}{clean_instr}")
                
                # Check for function end
                if stripped == 'ret':
                    cleaned.append(f"{base_indent}; C function end")
                    in_function = False
            elif stripped.startswith('.'):
                # Handle data declarations
                if '.ascii' in stripped or 'db ' in stripped:
                    cleaned.append(f"{asm_indent}{self._clean_data_declaration(stripped)}")
                else:
                    cleaned.append(f"{base_indent}; {stripped}")
        
        return cleaned
    
    def _clean_instruction(self, instruction: str) -> str:
        """Clean individual assembly instruction"""
        # Remove unnecessary prefixes and clean syntax
        instruction = instruction.strip()
        
        # Handle common patterns
        if instruction.startswith('mov') or instruction.startswith('add') or instruction.startswith('sub'):
            # Clean register references and memory operands
            instruction = re.sub(r'DWORD PTR', 'dword', instruction)
            instruction = re.sub(r'QWORD PTR', 'qword', instruction)
            instruction = re.sub(r'\[rbp\]', '[rbp]', instruction)
            
        elif instruction.startswith('call'):
            # Map C function calls to comments
            if 'printf' in instruction:
                return f"{instruction}  ; C printf call"
        
        # Map variable references in comments
        for var_label, var_info in self.c_variables.items():
            if var_label in instruction:
                instruction += f"  ; CASM variable: {var_info['casm_name']}"
                break
        
        return instruction
    
    def _clean_data_declaration(self, declaration: str) -> str:
        """Clean data declaration"""
        # Extract string content and create clean declaration
        if '.ascii' in declaration:
            match = re.search(r'"([^"]*)"', declaration)
            if match:
                string_content = match.group(1)
                return f'db "{string_content}", 0  ; C string constant'
        
        return declaration
    
    def _final_cleanup(self, assembly_code: str) -> str:
        """Final cleanup and formatting"""
        lines = assembly_code.split('\n')
        cleaned_lines = []
        
        prev_empty = False
        
        for line in lines:
            # Remove excessive empty lines
            if not line.strip():
                if not prev_empty:
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        # Ensure proper spacing around sections
        final_lines = []
        for i, line in enumerate(cleaned_lines):
            stripped = line.strip()
            
            # Add spacing before major sections
            if (stripped.startswith('section') and i > 0 and 
                cleaned_lines[i-1].strip() and 
                not cleaned_lines[i-1].strip().startswith('section')):
                final_lines.append('')
            
            final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def _generate_condition_assembly(self, condition: str) -> List[str]:
        """Generate assembly code for condition evaluation"""
        asm_lines = []
        condition = condition.strip()
        
        # Handle simple comparisons like "number > 0", "x == 5", etc.
        comparison_ops = {
            '>=': 'jl',   # jump if less (opposite of >=)
            '<=': 'jg',   # jump if greater (opposite of <=)
            '>': 'jle',   # jump if less or equal (opposite of >)
            '<': 'jge',   # jump if greater or equal (opposite of <)
            '==': 'jne',  # jump if not equal (opposite of ==)
            '!=': 'je'    # jump if equal (opposite of !=)
        }
        
        # Find the comparison operator
        for op in comparison_ops.keys():
            if op in condition:
                left, right = condition.split(op, 1)
                left = left.strip()
                right = right.strip()
                
                # Generate assembly for left operand
                if left in self.variable_map:
                    var_label = f"var_{left}"
                    asm_lines.append(f"    mov eax, dword [rel {var_label}]")
                elif left.isdigit():
                    asm_lines.append(f"    mov eax, {left}")
                else:
                    asm_lines.append(f"    ; TODO: Complex left operand: {left}")
                    asm_lines.append(f"    mov eax, 0  ; placeholder")
                
                # Generate assembly for right operand and comparison
                if right in self.variable_map:
                    var_label = f"var_{right}"
                    asm_lines.append(f"    cmp eax, dword [rel {var_label}]")
                elif right.isdigit():
                    asm_lines.append(f"    cmp eax, {right}")
                else:
                    asm_lines.append(f"    ; TODO: Complex right operand: {right}")
                    asm_lines.append(f"    cmp eax, 0  ; placeholder")
                
                return asm_lines
        
        # Fallback for simple boolean conditions
        if condition in self.variable_map:
            var_label = f"var_{condition}"
            asm_lines.append(f"    mov eax, dword [rel {var_label}]")
            asm_lines.append(f"    cmp eax, 0")
        elif condition.isdigit():
            asm_lines.append(f"    mov eax, {condition}")
            asm_lines.append(f"    cmp eax, 0")
        else:
            asm_lines.append(f"    ; TODO: Complex condition: {condition}")
            asm_lines.append(f"    mov eax, 1  ; assume true")
            asm_lines.append(f"    cmp eax, 0")
        
        return asm_lines

# Global instance
pretty_printer = CASMPrettyPrinter()