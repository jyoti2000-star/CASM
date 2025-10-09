#!/usr/bin/env python3

from typing import List, Dict, Set
import secrets
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
        # Deterministic counters for human-friendly labels
        self.naming_salt = None
        self.var_counter = 0
        self.str_counter = 0
        self.fmt_counter = 0
        self.lc_counter = 0
    
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
            from ..utils.colors import print_debug
            print_debug(f"Added .data section with {len(self.data_section)} items")
        
        # BSS section (uninitialized data)
        if hasattr(self, 'bss_section') and self.bss_section:
            lines.append("section .bss")
            lines.extend(self.bss_section)
            lines.append("")
            from ..utils.colors import print_debug
            print_debug(f"Added .bss section with {len(self.bss_section)} items")
        
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
            
            # Pass existing string label mapping so the pretty printer doesn't
            # generate duplicate STR labels for the same literals.
            prettified_code = pretty_printer.prettify(assembly_code, ast, self.variable_labels, c_assembly_segments, existing_string_labels=self.string_labels)

            # Dump prettified code to a temp file for debugging inspection
            try:
                with open('/tmp/casm_prettified_debug.asm', 'w', encoding='utf-8') as _df:
                    _df.write(prettified_code)
            except Exception:
                pass

            # Fix assembly for NASM compatibility
            fixed_code = assembly_fixer.fix_assembly(prettified_code)

            # Ensure string definitions are placed in .data (post-pass)
            final_code = self._move_strings_to_data(fixed_code)

            print_success("Assembly code generated, prettified and fixed for NASM")
            return final_code
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

    def _move_strings_to_data(self, assembly: str) -> str:
        """Ensure any `str_... db ...` lines are in section .data and not in .bss.

        This is a small, defensive post-pass to avoid string declarations ending up
        in the BSS after other transformation passes.
        """
        import re

        lines = assembly.split('\n')

        # Collect all string db lines (deduplicated, preserve order)
        string_lines = []
        seen = set()
        str_re = re.compile(r"^\s*(STR[\w\d_]+\s+db\s+.*)$")
        for line in lines:
            m = str_re.match(line)
            if m:
                s = m.group(1).strip()
                if s not in seen:
                    seen.add(s)
                    string_lines.append(s)

        if not string_lines:
            return assembly

        # Remove all occurrences of those string lines from the assembly
        new_lines = []
        for line in lines:
            if str_re.match(line):
                # skip string lines (we will reinsert under .data)
                continue
            new_lines.append(line)

        # Find insertion point: after 'section .data' header if present, else before first section
        out = []
        inserted = False
        for i, line in enumerate(new_lines):
            out.append(line)
            if not inserted and line.strip().startswith('section .data'):
                # Insert strings after this header
                for s in string_lines:
                    out.append(f"    {s}")
                inserted = True

        if not inserted:
            # No data section found; prepend one at top
            header = ["section .data"] + [f"    {s}" for s in string_lines] + [""]
            out = header + out

        return '\n'.join(out)
    
    def visit_var_declaration(self, node: VarDeclarationNode):
        """Visit variable declaration with type support"""
        # Use deterministic variable labels V1, V2, ... for readability and
        # to match requested naming scheme.
        self.var_counter += 1
        label = f"V{self.var_counter}"
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

        from ..utils.colors import print_debug
        print_debug(f"Processing variable: {node.name}, type: {var_type}, size: {var_size}, value: '{value}'")
        
        if var_type == "buffer" and var_size:
            # Buffers go in .bss section (uninitialized data)
            self.bss_section.append(f"    {label} resb {var_size}  ; buffer[{var_size}]")
            self.text_section.append(f"    ; Buffer: {node.name}[{var_size}] in .bss")
            print_debug(f"Added buffer to bss_section: {label} resb {var_size}")
            
        elif var_type == "int":
            # Initialize to 0 in data section, then calculate value if it's an expression
            self.data_section.append(f"    {label} dd 0  ; int {node.name}")
            self.text_section.append(f"    ; Variable: int {node.name} = {value}")
            
            # Handle initialization expression
            if value and value.strip() and value.strip() != "0":
                self._generate_assignment_code(node.name, value.strip())
            
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
            # Strings get STR# labels
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
    
    def _generate_assignment_code(self, var_name: str, expression: str):
        """Generate assembly code for variable assignment with expression evaluation (supports parentheses and precedence)"""
        var_label = self._get_var_label(var_name)

        # Tokenize the expression
        def tokenize(expr):
            tokens = []
            i = 0
            while i < len(expr):
                if expr[i].isspace():
                    i += 1
                elif expr[i] in '+-*/()':
                    tokens.append(expr[i])
                    i += 1
                else:
                    j = i
                    while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                        j += 1
                    tokens.append(expr[i:j])
                    i = j
            return tokens

        tokens = tokenize(expression)
        pos = [0]

        # Recursive descent parser for expressions
        def parse_expr():
            node = parse_term()
            while pos[0] < len(tokens) and tokens[pos[0]] in ('+', '-'):
                op = tokens[pos[0]]
                pos[0] += 1
                right = parse_term()
                node = (op, node, right)
            return node

        def parse_term():
            node = parse_factor()
            while pos[0] < len(tokens) and tokens[pos[0]] in ('*', '/'):
                op = tokens[pos[0]]
                pos[0] += 1
                right = parse_factor()
                node = (op, node, right)
            return node

        def parse_factor():
            if pos[0] < len(tokens) and tokens[pos[0]] == '(':  # Parenthesized
                pos[0] += 1
                node = parse_expr()
                if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                    pos[0] += 1
                return node
            elif pos[0] < len(tokens):
                token = tokens[pos[0]]
                pos[0] += 1
                return token
            return None

        ast = parse_expr()

        # Generate assembly from AST
        temp_reg = ['eax', 'ebx', 'ecx', 'edx']
        used_regs = set()

        def eval_node(node):
            # Returns the register holding the result
            if isinstance(node, tuple):
                op, left, right = node
                reg_left = eval_node(left)
                # For division, right must be in ecx
                if op == '/':
                    reg_right = 'ecx'
                    if reg_left != 'eax':
                        self.text_section.append(f"    mov eax, {reg_left}")
                        reg_left = 'eax'
                    eval_node_to_reg(right, reg_right)
                    self.text_section.append(f"    xor edx, edx")
                    self.text_section.append(f"    div {reg_right}")
                    return 'eax'
                else:
                    # For +, -, *
                    reg_right = None
                    for r in temp_reg:
                        if r not in used_regs and r != reg_left:
                            reg_right = r
                            break
                    if reg_right is None:
                        reg_right = 'ebx'  # fallback
                    eval_node_to_reg(right, reg_right)
                    if op == '+':
                        self.text_section.append(f"    add {reg_left}, {reg_right}")
                    elif op == '-':
                        self.text_section.append(f"    sub {reg_left}, {reg_right}")
                    elif op == '*':
                        self.text_section.append(f"    imul {reg_left}, {reg_right}")
                    return reg_left
            else:
                # node is a variable or number
                reg = None
                for r in temp_reg:
                    if r not in used_regs:
                        reg = r
                        used_regs.add(r)
                        break
                if node.isdigit():
                    self.text_section.append(f"    mov {reg}, {node}")
                elif node in self.variable_labels:
                    # Always load variable into register, never memory-to-memory
                    self.text_section.append(f"    mov {reg}, dword [rel {self.variable_labels[node]}]")
                else:
                    var_label = self._get_var_label(node)
                    self.text_section.append(f"    mov {reg}, dword [rel {var_label}]")
                return reg

        def eval_node_to_reg(node, reg):
            # Evaluate node and move result to reg
            result_reg = eval_node(node)
            if result_reg != reg:
                self.text_section.append(f"    mov {reg}, {result_reg}")

        result_reg = eval_node(ast)
        # If result_reg is not a register, move to eax first
        valid_regs = {"eax", "ebx", "ecx", "edx"}
        if result_reg not in valid_regs:
            self.text_section.append(f"    mov eax, {result_reg}")
            result_reg = "eax"
        self.text_section.append(f"    mov dword [rel {var_label}], {result_reg}")
    
    def visit_assignment(self, node: AssignmentNode):
        """Visit variable assignment"""
        # Use the new assignment code generator
        self._generate_assignment_code(node.name, node.value.strip())
    
    def visit_if(self, node: IfNode):
        """Visit if statement with proper nested if handling"""
        # Generate randomized labels for this if/else block
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
            
            # Else body - handle nested if statements properly
            for stmt in node.else_body:
                stmt.accept(self)
        
        self.text_section.append(f"{end_label}:")
    
    def visit_while(self, node: WhileNode):
        """Visit while loop"""
        # Generate randomized labels for while loop
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
        """Visit for loop (fix: declare and update loop variable, avoid label redefinition)"""
        # Generate labels and counter variable with randomized salt
        start_label = self._generate_label("for_start")
        end_label = self._generate_label("for_end")
        counter_var = self._generate_label("for_counter")

        # Use existing variable label if declared, otherwise create a new var label
        if node.variable in self.variable_labels:
            loop_var = self.variable_labels[node.variable]
        else:
            loop_var = f"var_{node.variable}_{self.naming_salt}"

        self.text_section.append(f"    ; for {node.variable} in range({node.count})")

        # Add counter variable and loop variable to data section if not already present
        if counter_var not in self.variable_labels.values():
            self.data_section.append(f"    {counter_var} dd 0")
        if loop_var not in self.variable_labels.values():
            self.data_section.append(f"    {loop_var} dd 0")
        # Ensure mapping for loop variable exists
        self.variable_labels.setdefault(node.variable, loop_var)

        # Initialize counter and loop variable
        self.text_section.append(f"    mov dword [rel {counter_var}], 0")
        self.text_section.append(f"    mov dword [rel {loop_var}], 0")
        self.text_section.append(f"{start_label}:")

        # Check condition - handle both numbers and variables
        self.text_section.append(f"    mov eax, dword [rel {counter_var}]")
        if node.count.isdigit():
            self.text_section.append(f"    cmp eax, {node.count}")
        else:
            if node.count in self.variable_labels:
                count_label = self.variable_labels[node.count]
                self.text_section.append(f"    cmp eax, dword [rel {count_label}]")
            else:
                count_label = self._get_var_label(node.count)
                self.text_section.append(f"    cmp eax, dword [rel {count_label}]")
        self.text_section.append(f"    jge {end_label}")


        # Set loop variable to current counter value (use register to avoid memory-to-memory move)
        self.text_section.append(f"    mov eax, dword [rel {counter_var}]")
        self.text_section.append(f"    mov dword [rel {loop_var}], eax")

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
        
        # Check if this is a variable reference
        if message in self.variable_labels:
            # This is a variable - print its value
            var_label = self.variable_labels[message]
            var_info = self.variable_info.get(message, {})
            var_type = var_info.get('type', 'int')
            
            if var_type in ['int', 'bool']:
                # Generate format string for integer
                format_label = self._generate_label("fmt_int")
                self.data_section.append(f'    {format_label} db "%d", 10, 0')
                
                # Generate printf call for integer
                self.text_section.append(f"    ; println {message} (int)")
                self.text_section.append(f"    mov edx, dword [rel {var_label}]")
                self.text_section.append(f"    lea rcx, [rel {format_label}]")
                self.text_section.append("    call printf")
            elif var_type in ['str', 'string']:
                # For strings, we need to check if it's a buffer or a string constant
                format_label = self._generate_label("fmt_str")
                self.data_section.append(f'    {format_label} db "%s", 10, 0')
                
                # Generate printf call for string
                self.text_section.append(f"    ; println {message} (string)")
                self.text_section.append(f"    lea rdx, [rel {var_label}]")
                self.text_section.append(f"    lea rcx, [rel {format_label}]")
                self.text_section.append("    call printf")
            else:
                # Fallback for other types
                format_label = self._generate_label("fmt_int")
                self.data_section.append(f'    {format_label} db "%d", 10, 0')
                
                self.text_section.append(f"    ; println {message}")
                self.text_section.append(f"    mov edx, dword [rel {var_label}]")
                self.text_section.append(f"    lea rcx, [rel {format_label}]")
                self.text_section.append("    call printf")
        else:
            # This is a string literal
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]  # Remove quotes
            
            # Reuse existing label for identical string literals to avoid
            # duplicate STR labels being emitted multiple times.
            if message in self.string_labels:
                string_label = self.string_labels[message]
            else:
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
        var_label = self._get_var_label(node.variable)
        
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
        # Processing logs for embedded C are noisy; use debug-only output
        from ..utils.colors import print_debug
        print_debug(f"Processing C code: '{node.c_code}'")
        
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
            # Emit original C code as comments so the assembly output shows the source C statements
            if node.c_code and node.c_code.strip():
                for c_line in node.c_code.split('\n'):
                    if c_line.strip():
                        self.text_section.append(f"    ; {c_line.strip()}")
            self.text_section.append(f"    nop  ; Start of C block {block_marker}")
            self.text_section.append(f"    ; {{{block_marker}}} ; Placeholder for assembly")
            self.text_section.append(f"    nop  ; End of C block {block_marker}")
            
        except ImportError:
            print_warning("C processor not available, skipping C code")
            self.text_section.append(f"    ; C code skipped: {node.c_code}")
        except Exception as e:
            print_error(f"C compilation error: {e}")
            self.text_section.append(f"    ; C code error: {node.c_code}")

        # Debug-only message to indicate collection
        from ..utils.colors import print_debug
        print_debug(f"C code collected: {node.c_code}")
    
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
        """Compile all C code and replace placeholders with actual assembly

        Exceptions from the C compilation step are intentionally propagated so the
        caller can abort the overall build when GCC fails. This avoids writing a
        potentially broken assembly file when embedded C cannot be compiled.
        """
        from ..utils.c_processor import c_processor

        print_info(f"Finalizing C code: {len(c_processor.c_code_blocks)} blocks collected")

        # Compile all collected C code blocks (may raise RuntimeError on failure)
        assembly_segments = c_processor.compile_all_c_code()

        # If extraction returned nothing, attempt a fallback: read the saved
        # combined GCC assembly file (if present) and re-run extraction. This
        # helps in cases where the in-memory segments were not set for some
        # reason but the raw assembly was saved for debugging.
        if not assembly_segments:
            try:
                out_dir = os.path.join(os.getcwd(), 'output')
                asm_out = os.path.join(out_dir, 'combined_asm_from_cprocessor.s')
                if os.path.exists(asm_out):
                    with open(asm_out, 'r', encoding='utf-8') as _af:
                        raw_asm = _af.read()
                    print_info('Attempting fallback extraction from saved combined assembly')
                    assembly_segments = c_processor._extract_assembly_segments(raw_asm)
                    # Also extract string constants if present
                    sc = c_processor._extract_string_constants(raw_asm)
                    if sc:
                        assembly_segments['_STRING_CONSTANTS'] = sc
            except Exception as e:
                print_warning(f'Fallback assembly extraction failed: {e}')

        if not assembly_segments:
            # No compiled C assembly segments. If there were blocks collected, this
            # means compilation produced nothing; surface a warning and continue.
            if c_processor.c_code_blocks:
                print_warning("No C code blocks were compiled into assembly")
            return

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
                        # Strings coming from C processor use CSTR prefix already
                        self.data_section.insert(0, f"    {line.strip()}")
            del assembly_segments['_STRING_CONSTANTS']

        # Replace placeholders with actual assembly
        updated_sections = []
        for line in self.text_section:
            if "{CASM_BLOCK_" in line and "} ; Placeholder for assembly" in line:
                # Extract marker from placeholder
                marker_match = re.search(r'\{(CASM_BLOCK_\d+)\}', line)
                if marker_match:
                    marker = marker_match.group(1)
                    if marker in assembly_segments:
                        # Replace with actual assembly
                        assembly_code = assembly_segments[marker]
                        if assembly_code.strip():
                            # Remove the previously emitted C-block header/comments/NOPs
                            header = f"    ; C code block: {marker}"
                            # search backwards for the header and trim anything from it onward
                            for i in range(len(updated_sections)-1, -1, -1):
                                if updated_sections[i] == header:
                                    # remove header and anything after it
                                    del updated_sections[i:]
                                    break

                            # Insert only the cleaned assembly instructions (no generated comment)
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
        
        # If this extern is a C include (e.g. <math.h> or "mylib.h"), forward
        # it to the C processor so it becomes a #include in the combined C file.
        try:
            from ..utils.c_processor import c_processor

            # Normalize header string (strip surrounding whitespace)
            raw = header.strip() if header is not None else ''

            # If the parser flagged this as a C include, honor it.
            is_c_include_flag = bool(getattr(node, 'is_c_include', False))
            use_angle_flag = bool(getattr(node, 'use_angle', False))

            # Heuristic: if the header string contains angle brackets or looks
            # like an include (e.g. '< math .h>' or '<math.h>'), treat it as
            # a C include even if the parser didn't set the flag.
            if (is_c_include_flag) or ('<' in raw or '>' in raw):
                # Remove angle brackets and extra spaces from header name
                cleaned = raw.replace('<', '').replace('>', '').replace(' ', '')
                # If parser suggested use_angle, prefer that; otherwise assume angle when original had <>
                use_angle = use_angle_flag or ('<' in raw and '>' in raw)
                c_processor.add_header(cleaned, use_angle=use_angle)
                # Add a brief comment in assembly that the include was forwarded
                self.text_section.append(f"    ; forwarded C include: {cleaned}")
            else:
                # Treat as assembler extern symbol and emit an extern directive
                self.text_section.append(f"    extern {raw}")
        except ImportError:
            print_warning("C processor not available for header processing")
            # Fallback: always emit as assembler extern
            self.text_section.append(f"    extern {header}")
    
    def _generate_label(self, prefix: str) -> str:
        """Generate unique label"""
        # Deterministic, human-friendly label generation
        # Special-case common prefixes
        if prefix.startswith("fmt") or prefix == "fmt_int" or prefix == "fmt_str" or prefix == "fmt":
            self.fmt_counter += 1
            return f"FMT{self.fmt_counter - 1 if self.fmt_counter>0 else 0}"
        if prefix == "str" or prefix.startswith("str"):
            # STR labels for string literals
            self.str_counter += 1
            return f"STR{self.str_counter}"
        if prefix.startswith("LC") or prefix == "LC":
            self.lc_counter += 1
            return f"LC{self.lc_counter - 1}"

        # Generic prefix: use prefix capitalized + counter
        self.label_counter += 1
        return f"{prefix.upper()}{self.label_counter}"
    
    def _generate_unique_id(self) -> int:
        """Generate unique ID for complex constructs like if/while/for"""
        self.label_counter += 1
        return self.label_counter

    def _get_var_label(self, name: str) -> str:
        """Return the assembly label for a given CASM variable name.

        If the variable was declared, return the canonical label. Otherwise
        fall back to the predictable var_<name>_salt form so references remain
        consistent with randomized naming.
        """
        if name in self.variable_labels:
            return self.variable_labels[name]
        # Fallback: allocate a new deterministic variable label
        self.var_counter += 1
        label = f"V{self.var_counter}"
        # Ensure mapping for future references
        self.variable_labels[name] = label
        return label
    
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
                    left_label = self._get_var_label(left)
                    self.text_section.append(f"    mov eax, dword [rel {left_label}]")
                
                # Compare with right operand
                if right.isdigit():
                    self.text_section.append(f"    cmp eax, {right}")
                elif right in self.variable_labels:
                    var_label = self.variable_labels[right]
                    self.text_section.append(f"    cmp eax, dword [rel {var_label}]")
                else:
                    # Try to treat as variable name
                    right_label = self._get_var_label(right)
                    self.text_section.append(f"    cmp eax, dword [rel {right_label}]")
                
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
            cond_label = self._get_var_label(condition)
            self.text_section.append(f"    mov eax, dword [rel {cond_label}]")
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
                    left_label = self._get_var_label(left)
                    self.text_section.append(f"    mov eax, dword [rel {left_label}]")
                
                # Compare with right operand
                if right.isdigit():
                    self.text_section.append(f"    cmp eax, {right}")
                elif right in self.variable_labels:
                    var_label = self.variable_labels[right]
                    self.text_section.append(f"    cmp eax, dword [rel {var_label}]")
                else:
                    # Try to treat as variable name
                    right_label = self._get_var_label(right)
                    self.text_section.append(f"    cmp eax, dword [rel {right_label}]")
                
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
            cond_label = self._get_var_label(condition)
            self.text_section.append(f"    mov eax, dword [rel {cond_label}]")
            self.text_section.append(f"    cmp eax, 0")