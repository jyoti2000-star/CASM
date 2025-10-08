#!/usr/bin/env python3

"""
Code generator for CASM
Converts AST to x86-64 Windows assembly
"""

from typing import List, Dict, Set
import re
from .ast_nodes import *
from ..utils.formatter import formatter
from ..utils.colors import print_info, print_success, print_error, print_warning

class AssemblyCodeGenerator(ASTVisitor):
    """Generate x86-64 Windows assembly from CASM AST"""
    
    def __init__(self):
        """Initialize code generator with empty sections"""
        self.data_section = []
        self.bss_section = []
        self.text_section = []
        self.string_labels = {}
        self.variable_labels = {}
        self.variable_info = {}  # Store complete variable information for C integration
        self.label_counter = 0
        self.used_functions = set()
    
    def generate(self, ast: ProgramNode) -> str:
        """Generate complete assembly program"""
        # Reset state
        self.data_section = []
        self.bss_section = []
        self.text_section = []
        self.string_labels = {}
        self.variable_labels = {}
        self.variable_info = {}
        self.label_counter = 0
        self.used_functions = set()
        
        # Pass variable information to C processor
        try:
            from ..utils.c_processor import c_processor
            c_processor.set_casm_variables(self.variable_info)
        except ImportError:
            print_warning("C processor not available")
        
        # Visit the AST
        ast.accept(self)
        
        # Finalize C code compilation and replace placeholders
        self.finalize_c_code()
        
        # Get any C variables for potential CASM access
        try:
            from ..utils.c_processor import c_processor
            c_variables = c_processor.get_c_variables()
            if c_variables:
                print_info(f"C variables available for CASM: {list(c_variables.keys())}")
        except ImportError:
            c_variables = {}
        
        # Build final assembly
        lines = []
        
        # Add external function declarations
        external_functions = set()
        external_functions.update(self.used_functions)
        
        # Always add common C library functions that might be used by compiled C code
        common_c_functions = {'printf', 'puts', 'putchar', 'scanf', 'strlen', 'strcpy', 
                             'strcmp', 'strstr', 'malloc', 'free', 'pow', 'sqrt', 'sin', 'cos',
                             'srand', 'rand', 'exit', 'abort'}
        external_functions.update(common_c_functions)
        
        if external_functions:
            lines.append("; External function declarations")
            for func in sorted(external_functions):
                lines.append(f"extern {func}")
            lines.append("")
        
        # Add C function declarations if any
        if c_variables:
            lines.append("; C variables accessible from CASM")
            for var_name, var_info in c_variables.items():
                lines.append(f"extern {var_info['c_name']}")
            lines.append("")
        
        # Data section
        if self.data_section:
            lines.append("section .data")
            lines.extend(self.data_section)
            lines.append("")
            print(f"[DEBUG] Added .data section with {len(self.data_section)} items")
        
        # BSS section (uninitialized data)
        if hasattr(self, 'bss_section') and self.bss_section:
            lines.append("section .bss")
            lines.extend(self.bss_section)
            lines.append("")
            print(f"[DEBUG] Added .bss section with {len(self.bss_section)} items")
        
        # Text section
        lines.append("section .text")
        lines.append("global main")
        lines.append("")
        lines.append("main:")
        lines.append("    push rbp")
        lines.append("    mov rbp, rsp")
        lines.append("    sub rsp, 32  ; Shadow space for Windows x64")
        lines.append("")
        
        lines.extend(self.text_section)
        
        lines.append("")
        lines.append("    ; Exit program")
        lines.append("    mov rax, 0")
        lines.append("    add rsp, 32")
        lines.append("    pop rbp")
        lines.append("    ret")
        
        # Format the assembly
        assembly_code = '\n'.join(lines)
        
        # Use the pretty printer for context-aware formatting
        try:
            from .pretty import pretty_printer
            from ..utils.c_processor import c_processor
            from .assembly_fixer import assembly_fixer
            
            # Get the compiled assembly segments for the pretty printer
            c_assembly_segments = getattr(c_processor, '_last_assembly_segments', {})
            
            prettified_code = pretty_printer.prettify(assembly_code, ast, self.variable_labels, c_assembly_segments)
            
            # Fix assembly for NASM compatibility
            fixed_code = assembly_fixer.fix_assembly(prettified_code)
            
            print_success("Assembly code generated, prettified and fixed for NASM")
            return fixed_code
        except ImportError as e:
            print_warning(f"Pretty printer or fixer not available: {e}")
            # Fallback to regular formatting if pretty printer fails
            formatted_code = formatter.format_assembly(assembly_code)
            print_success("Assembly code generated and formatted")
            return formatted_code
    
    def visit_program(self, node: ProgramNode):
        """Visit program root node"""
        for stmt in node.statements:
            stmt.accept(self)
    
    def visit_var_declaration(self, node: VarDeclarationNode):
        """Visit variable declaration with type support"""
        label = f"var_{node.name}"
        self.variable_labels[node.name] = label
        
        # Store complete variable information for C integration
        self.variable_info[node.name] = {
            'type': node.var_type,
            'size': node.size,
            'label': label,
            'value': node.value
        }
        
        var_type = node.var_type
        var_size = node.size
        value = node.value
        
        print(f"[DEBUG] Processing variable: {node.name}, type: {var_type}, size: {var_size}, value: '{value}'")
        
        if var_type == "buffer" and var_size:
            # Buffers go in .bss section (uninitialized data)
            self.bss_section.append(f"    {label} resb {var_size}  ; buffer[{var_size}]")
            self.text_section.append(f"    ; Buffer: {node.name}[{var_size}] in .bss")
            print(f"[DEBUG] Added buffer to bss_section: {label} resb {var_size}")
            
        elif var_type == "int":
            try:
                int_value = int(value)
                self.data_section.append(f"    {label} dd {int_value}  ; int {node.name}")
            except ValueError:
                self.data_section.append(f"    {label} dd 0  ; int {node.name} = {value}")
            self.text_section.append(f"    ; Variable: int {node.name} = {value}")
            
        elif var_type == "bool":
            bool_value = 1 if value.lower() in ['true', '1', 'yes'] else 0
            self.data_section.append(f"    {label} db {bool_value}  ; bool {node.name}")
            self.text_section.append(f"    ; Variable: bool {node.name} = {value}")
            
        elif var_type == "float":
            try:
                # Convert float to IEEE 754 format or use default
                float_value = float(value)
                # For simplicity, store as a string and let assembler handle it
                self.data_section.append(f"    {label} dq {float_value}  ; float {node.name}")
            except ValueError:
                self.data_section.append(f"    {label} dq 0.0  ; float {node.name} = {value}")
            self.text_section.append(f"    ; Variable: float {node.name} = {value}")
            
        elif var_type == "str":
            # Remove quotes if present
            str_value = value.strip()
            if str_value.startswith('"') and str_value.endswith('"'):
                str_value = str_value[1:-1]
            
            # Add null terminator and escape sequences
            str_value = str_value.replace('\\n', '", 10, "').replace('\\t', '", 9, "')
            self.data_section.append(f'    {label} db "{str_value}", 0  ; str {node.name}')
            self.text_section.append(f"    ; Variable: str {node.name} = {value}")
            
        elif var_type == "int" and var_size:
            # Integer array
            if value and value.strip():
                # Parse comma-separated values
                values = [v.strip() for v in value.split(',')]
                values_str = ', '.join(values)
                self.data_section.append(f"    {label} dd {values_str}  ; int {node.name}[{var_size}]")
            else:
                # Initialize with zeros
                self.data_section.append(f"    {label} times {var_size} dd 0  ; int {node.name}[{var_size}]")
            self.text_section.append(f"    ; Array: int {node.name}[{var_size}]")
            
        else:
            # Fallback to old behavior
            try:
                int_value = int(value)
                self.data_section.append(f"    {label} dd {int_value}")
            except ValueError:
                self.data_section.append(f"    {label} dd 0  ; {value}")
            self.text_section.append(f"    ; Variable: {node.name} = {value}")
    
    def visit_assignment(self, node: AssignmentNode):
        """Visit variable assignment"""
        # Find the variable label
        var_label = self.variable_labels.get(node.name, f"var_{node.name}")
        
        # Parse the expression (for now, handle simple cases)
        expression = node.value.strip()
        
        # Handle simple arithmetic like "i + 1" or "i - 1"
        if '+' in expression:
            parts = expression.split('+')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                
                # Load left operand
                if left == node.name:
                    self.text_section.append(f"    mov eax, dword [rel {var_label}]")
                elif left.isdigit():
                    self.text_section.append(f"    mov eax, {left}")
                else:
                    # Assume it's another variable
                    self.text_section.append(f"    mov eax, dword [rel var_{left}]")
                
                # Add right operand
                if right.isdigit():
                    self.text_section.append(f"    add eax, {right}")
                else:
                    # Assume it's another variable
                    self.text_section.append(f"    add eax, dword [rel var_{right}]")
                
                # Store result
                self.text_section.append(f"    mov dword [rel {var_label}], eax")
                return
        
        # Handle simple assignment like "i = 5"
        if expression.isdigit():
            self.text_section.append(f"    mov dword [rel {var_label}], {expression}")
        else:
            # Variable assignment like "i = j"
            if expression in self.variable_labels:
                source_label = self.variable_labels[expression]
                self.text_section.append(f"    mov eax, dword [rel {source_label}]")
            else:
                self.text_section.append(f"    mov eax, dword [rel var_{expression}]")
            self.text_section.append(f"    mov dword [rel {var_label}], eax")
    
    def visit_if(self, node: IfNode):
        """Visit if statement"""
        end_label = self._generate_label("if_end")
        else_label = self._generate_label("else") if node.else_body else None
        
        self.text_section.append(f"    ; if {node.condition}")
        
        # Evaluate the condition and get the appropriate jump instruction
        jump_instruction = self._evaluate_condition_with_jump(node.condition)
        
        if else_label:
            self.text_section.append(f"    {jump_instruction} {else_label}")
        else:
            self.text_section.append(f"    {jump_instruction} {end_label}")
        
        # If body
        for stmt in node.if_body:
            stmt.accept(self)
        
        if else_label:
            self.text_section.append(f"    jmp {end_label}")
            self.text_section.append(f"{else_label}:")
            
            # Else body
            for stmt in node.else_body:
                stmt.accept(self)
        
        self.text_section.append(f"{end_label}:")
    
    def visit_while(self, node: WhileNode):
        """Visit while loop"""
        start_label = self._generate_label("while_start")
        end_label = self._generate_label("while_end")
        
        self.text_section.append(f"{start_label}:")
        self.text_section.append(f"    ; while {node.condition}")
        
        # Evaluate the condition and get the appropriate jump instruction
        jump_instruction = self._evaluate_condition_with_jump(node.condition)
        self.text_section.append(f"    {jump_instruction} {end_label}")
        
        # Loop body
        for stmt in node.body:
            stmt.accept(self)
        
        self.text_section.append(f"    jmp {start_label}")
        self.text_section.append(f"{end_label}:")
    
    def visit_for(self, node: ForNode):
        """Visit for loop"""
        # Generate unique ID for this for loop consistent across data and text sections
        self.label_counter += 1
        for_loop_id = self.label_counter
        start_label = f"for_start_{for_loop_id}"
        end_label = f"for_end_{for_loop_id}"
        counter_var = f"for_counter_{for_loop_id}"
        
        self.text_section.append(f"    ; for {node.variable} in range({node.count})")
        
        # Add counter variable to data section first
        self.data_section.append(f"    {counter_var} dd 0")
        
        # Initialize counter
        self.text_section.append(f"    mov dword [rel {counter_var}], 0")
        self.text_section.append(f"{start_label}:")
        
        # Check condition
        self.text_section.append(f"    mov eax, dword [rel {counter_var}]")
        self.text_section.append(f"    cmp eax, {node.count}")
        self.text_section.append(f"    jge {end_label}")
        
        # Loop body
        for stmt in node.body:
            stmt.accept(self)
        
        # Increment counter
        self.text_section.append(f"    inc dword [rel {counter_var}]")
        self.text_section.append(f"    jmp {start_label}")
        self.text_section.append(f"{end_label}:")
    
    def visit_println(self, node: PrintlnNode):
        """Visit println statement"""
        self.used_functions.add("printf")
        
        # Clean the message
        message = node.message.strip()
        if message.startswith('"') and message.endswith('"'):
            message = message[1:-1]  # Remove quotes
        
        # Generate string label
        string_label = self._generate_label("str")
        self.string_labels[message] = string_label
        
        # Add string to data section with newline
        escaped_message = message.replace('\\n', '\n').replace('\\t', '\t')
        self.data_section.append(f'    {string_label} db "{escaped_message}", 10, 0')
        
        # Generate printf call (Windows x64 calling convention)
        self.text_section.append(f"    ; println \"{message}\"")
        self.text_section.append(f"    lea rcx, [rel {string_label}]")
        self.text_section.append("    call printf")
    
    def visit_scanf(self, node: ScanfNode):
        """Visit scanf statement"""
        self.used_functions.add("scanf")
        
        # Clean format string
        format_str = node.format_string.strip()
        if format_str.startswith('"') and format_str.endswith('"'):
            format_str = format_str[1:-1]
        
        # Generate labels
        format_label = self._generate_label("fmt")
        var_label = self.variable_labels.get(node.variable, f"var_{node.variable}")
        
        # Add format string to data section
        self.data_section.append(f'    {format_label} db "{format_str}", 0')
        
        # Ensure variable exists
        if node.variable not in self.variable_labels:
            self.data_section.append(f"    {var_label} dd 0")
            self.variable_labels[node.variable] = var_label
        
        # Generate scanf call (Windows x64 calling convention)
        self.text_section.append(f"    ; scanf \"{format_str}\" -> {node.variable}")
        self.text_section.append(f"    lea rcx, [rel {format_label}]")
        self.text_section.append(f"    lea rdx, [rel {var_label}]")
        self.text_section.append("    call scanf")
    
    def visit_assembly(self, node: AssemblyNode):
        """Visit raw assembly line"""
        self.text_section.append(f"    {node.code}")
    
    def visit_comment(self, node: CommentNode):
        """Visit comment"""
        self.text_section.append(f"    {node.text}")
    
    def visit_c_code_block(self, node: CCodeBlockNode):
        """Visit C code block"""
        print_info(f"Processing C code: '{node.c_code}'")
        
        if not node.c_code.strip():
            # Empty C code block
            self.text_section.append("    ; Empty C code block")
            return
        
        # Update C processor with current CASM variables before processing
        try:
            from ..utils.c_processor import c_processor
            c_processor.set_casm_variables(self.variable_info)
            
            # Add C code block to collection and get marker
            block_marker = c_processor.add_c_code_block(node.c_code)
            
            # Store the block ID in the node for the pretty printer
            node._block_id = block_marker.replace('CASM_BLOCK_', '')
            
            # Add NOP-wrapped placeholder for the assembly that will be filled after compilation
            self.text_section.append(f"    ; C code block: {block_marker}")
            self.text_section.append(f"    nop  ; Start of C block {block_marker}")
            self.text_section.append(f"    ; {{{block_marker}}} ; Placeholder for assembly")
            self.text_section.append(f"    nop  ; End of C block {block_marker}")
            
        except ImportError:
            print_warning("C processor not available, using fallback")
            self._fallback_c_processing(node.c_code)
        except Exception as e:
            print_error(f"C compilation error: {e}")
            self._fallback_c_processing(node.c_code)
        
        print_success(f"C code collected: {node.c_code}")
    
    def visit_asm_block(self, node: AsmBlockNode):
        """Visit assembly block"""
        print_info(f"Processing assembly block: '{node.asm_code}'")
        
        if not node.asm_code.strip():
            # Empty assembly block
            self.text_section.append("    ; Empty assembly block")
            return
        
        # Split assembly code into lines and add each line
        asm_lines = node.asm_code.strip().split('\n')
        self.text_section.append("    ; === Raw Assembly Block ===")
        for line in asm_lines:
            line = line.strip()
            if line:
                # Add proper indentation if not already present
                if not line.startswith('    ') and not line.endswith(':'):
                    line = f"    {line}"
                self.text_section.append(line)
        self.text_section.append("    ; === End Assembly Block ===")
    
    def finalize_c_code(self):
        """Compile all C code and replace placeholders with actual assembly"""
        try:
            from ..utils.c_processor import c_processor
            
            print_info(f"Finalizing C code: {len(c_processor.c_code_blocks)} blocks collected")
            
            # Compile all collected C code blocks
            assembly_segments = c_processor.compile_all_c_code()
            
            if assembly_segments:
                print_success(f"Compiled {len(assembly_segments)} C code blocks")
                
                # Add string constants to data section if available
                if '_STRING_CONSTANTS' in assembly_segments:
                    string_constants = assembly_segments['_STRING_CONSTANTS']
                    if string_constants.strip():
                        print_info("Adding C string constants to data section")
                        # Add string constants to the beginning of data section
                        string_lines = string_constants.split('\n')
                        for line in reversed(string_lines):
                            if line.strip():
                                self.data_section.insert(0, f"    {line.strip()}")
                    del assembly_segments['_STRING_CONSTANTS']
                
                # Replace placeholders with actual assembly
                updated_sections = []
                for line in self.text_section:
                    if line.strip().startswith("; {") and line.strip().endswith("} ; Placeholder for assembly"):
                        # Extract marker from placeholder
                        marker_match = re.search(r'; \{(CASM_BLOCK_\d+)\}', line)
                        if marker_match:
                            marker = marker_match.group(1)
                            if marker in assembly_segments:
                                # Replace with actual assembly
                                assembly_code = assembly_segments[marker]
                                if assembly_code.strip():
                                    updated_sections.append("    ; Generated assembly for " + marker)
                                    for asm_line in assembly_code.split('\n'):
                                        if asm_line.strip():
                                            updated_sections.append(f"    {asm_line}")
                                else:
                                    updated_sections.append("    ; Empty assembly block")
                            else:
                                updated_sections.append(f"    ; Assembly not found for {marker}")
                        else:
                            updated_sections.append(line)
                    else:
                        updated_sections.append(line)
                
                # Replace placeholders with actual assembly
                updated_sections = []
                for line in self.text_section:
                    if line.strip().startswith("; {") and line.strip().endswith("} ; Placeholder for assembly"):
                        # Extract marker from placeholder
                        marker_match = re.search(r'; \{(CASM_BLOCK_\d+)\}', line)
                        if marker_match:
                            marker = marker_match.group(1)
                            if marker in assembly_segments:
                                # Replace with actual assembly (wrapped in NOPs)
                                assembly_code = assembly_segments[marker]
                                if assembly_code.strip():
                                    updated_sections.append(f"    ; Generated assembly for {marker}")
                                    for asm_line in assembly_code.split('\n'):
                                        if asm_line.strip():
                                            updated_sections.append(f"    {asm_line}")
                                else:
                                    updated_sections.append("    ; Empty assembly block")
                            else:
                                updated_sections.append(f"    ; Assembly not found for {marker}")
                        else:
                            updated_sections.append(line)
                    else:
                        updated_sections.append(line)
                
                self.text_section = updated_sections
                print_success("C code assembly integration complete")
            else:
                print_warning("No C code blocks to compile")
                
        except Exception as e:
            print_error(f"C code finalization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _fallback_c_processing(self, c_code: str):
        """Fallback processing for C code when GCC compilation fails"""
        try:
            from ..utils.c_processor import c_processor
            # Process variable references in C code
            processed_c_code = c_processor._process_variable_references(c_code)
            
            # Process the single line of C code
            if '=' in processed_c_code and processed_c_code.endswith(';'):
                # Assignment
                asm_lines = c_processor._convert_assignment(processed_c_code)
            elif processed_c_code.endswith(';') and '(' in processed_c_code:
                # Function call
                asm_lines = c_processor._convert_function_call(processed_c_code)
            elif any(type_name in processed_c_code for type_name in ['int ', 'char ', 'float ', 'double ']):
                # Declaration
                asm_lines = c_processor._convert_declaration(processed_c_code)
            else:
                # Unknown C statement
                asm_lines = [f"    ; C: {processed_c_code}"]
            
            # Add the generated assembly
            for line in asm_lines:
                if line.strip():
                    self.text_section.append(line)
        except ImportError:
            # Ultimate fallback
            self.text_section.append(f"    ; C code: {c_code}")
    
    def visit_extern_directive(self, node: 'ExternDirectiveNode'):
        """Visit extern directive for header files"""
        header = node.header_name
        print_info(f"Processing extern directive: {header}")
        
        # Store header for C compilation
        try:
            from ..utils.c_processor import c_processor
            c_processor.add_header(header)
        except ImportError:
            print_warning("C processor not available for header processing")
        
        # Add as comment in assembly
        self.text_section.append(f"    ; extern {header}")
    
    def _generate_label(self, prefix: str) -> str:
        """Generate unique label"""
        self.label_counter += 1
        return f"{prefix}_{self.label_counter}"
    
    def _evaluate_condition_with_jump(self, condition: str):
        """Evaluate a condition and return the appropriate jump instruction to skip the block"""
        condition = condition.strip()
        
        # Handle comparisons like "variable > value", "variable == value", etc.
        comparison_ops = ['>=', '<=', '==', '!=', '>', '<']
        
        for op in comparison_ops:
            if op in condition:
                left, right = condition.split(op, 1)
                left = left.strip()
                right = right.strip()
                
                # Load left operand into eax
                if left.isdigit():
                    self.text_section.append(f"    mov eax, {left}")
                elif left in self.variable_labels:
                    var_label = self.variable_labels[left]
                    self.text_section.append(f"    mov eax, dword [rel {var_label}]")
                else:
                    # Try to treat as variable name
                    self.text_section.append(f"    mov eax, dword [rel var_{left}]")
                
                # Compare with right operand
                if right.isdigit():
                    self.text_section.append(f"    cmp eax, {right}")
                elif right in self.variable_labels:
                    var_label = self.variable_labels[right]
                    self.text_section.append(f"    cmp eax, dword [rel {var_label}]")
                else:
                    # Try to treat as variable name
                    self.text_section.append(f"    cmp eax, dword [rel var_{right}]")
                
                # Return the opposite jump to skip the if block when condition is false
                if op == '<':
                    return 'jge'    # Jump if greater or equal (opposite of <)
                elif op == '<=':
                    return 'jg'     # Jump if greater (opposite of <=)
                elif op == '>':
                    return 'jle'    # Jump if less or equal (opposite of >)
                elif op == '>=':
                    return 'jl'     # Jump if less (opposite of >=)
                elif op == '==':
                    return 'jne'    # Jump if not equal (opposite of ==)
                elif op == '!=':
                    return 'je'     # Jump if equal (opposite of !=)
        
        # Handle simple variable checks (non-zero)
        if condition in self.variable_labels:
            var_label = self.variable_labels[condition]
            self.text_section.append(f"    mov eax, dword [rel {var_label}]")
            self.text_section.append(f"    cmp eax, 0")
        elif condition.isdigit():
            # Direct number comparison
            self.text_section.append(f"    mov eax, {condition}")
            self.text_section.append(f"    cmp eax, 0")
        else:
            # Try to treat as variable name
            self.text_section.append(f"    mov eax, dword [rel var_{condition}]")
            self.text_section.append(f"    cmp eax, 0")
        
        # For simple variable/number checks, jump if zero (false)
        return 'je'

    def _evaluate_condition(self, condition: str):
        """Evaluate a condition and set flags for conditional jumps"""
        # Parse common condition patterns
        condition = condition.strip()
        
        # Handle comparisons like "variable > value", "variable == value", etc.
        comparison_ops = ['>=', '<=', '==', '!=', '>', '<']
        
        for op in comparison_ops:
            if op in condition:
                left, right = condition.split(op, 1)
                left = left.strip()
                right = right.strip()
                
                # Load left operand into eax
                if left.isdigit():
                    self.text_section.append(f"    mov eax, {left}")
                elif left in self.variable_labels:
                    var_label = self.variable_labels[left]
                    self.text_section.append(f"    mov eax, dword [rel {var_label}]")
                else:
                    # Try to treat as variable name
                    self.text_section.append(f"    mov eax, dword [rel var_{left}]")
                
                # Compare with right operand
                if right.isdigit():
                    self.text_section.append(f"    cmp eax, {right}")
                elif right in self.variable_labels:
                    var_label = self.variable_labels[right]
                    self.text_section.append(f"    cmp eax, dword [rel {var_label}]")
                else:
                    # Try to treat as variable name
                    self.text_section.append(f"    cmp eax, dword [rel var_{right}]")
                
                # The comparison sets flags, caller will use je/jne/jg/jl etc.
                return
        
        # Handle simple variable checks (non-zero)
        if condition in self.variable_labels:
            var_label = self.variable_labels[condition]
            self.text_section.append(f"    mov eax, dword [rel {var_label}]")
            self.text_section.append(f"    cmp eax, 0")
        elif condition.isdigit():
            # Direct number comparison
            self.text_section.append(f"    mov eax, {condition}")
            self.text_section.append(f"    cmp eax, 0")
        else:
            # Try to treat as variable name
            self.text_section.append(f"    mov eax, dword [rel var_{condition}]")
            self.text_section.append(f"    cmp eax, 0")