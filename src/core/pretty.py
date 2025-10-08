#!/usr/bin/env python3

"""
Pretty printer for CASM - Context-aware assembly cleanup
Intelligently organizes combined C and assembly code while preserving structure
"""

import re
from typing import List, Dict, Tuple, Optional
from .ast_nodes import *
from ..utils.colors import print_info, print_success, print_warning, print_debug

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
        self.control_counter = 0  # Counter for control flow labels
        self.extra_variables = []  # Variables to add to data section
        
    def prettify(self, assembly_code: str, ast: ProgramNode, variable_labels: Dict = None, gcc_blocks: Dict = None) -> str:
        """Format assembly code with proper structure and comments"""
        # Initialize tracking
        self.current_block_id = 0
        self.extra_variables = []
        self.variable_labels = variable_labels or {}
        self.variable_map = variable_labels or {}  # Use passed variable labels instead of empty map
        self.collected_strings = {}
        self.c_assembly_segments = gcc_blocks or {}
        
        # Extract existing labels from assembly before regenerating
        self.existing_labels = self._extract_labels(assembly_code)
        
        # Collect string literals from AST for cross-referencing
        self._collect_all_strings(ast)
        
        # Pre-scan AST to collect for loop counters
        self._collect_for_loop_counters(ast)
        
        lines = assembly_code.split('\n')
        sections = self._parse_assembly_sections(assembly_code)
        
        output = []
        
        # Header with comments
        if 'header' in sections:
            output.extend(self._format_header(sections['header']))
            output.append("")
        
        # External declarations - always ensure they are included
        output.append("; External function declarations")
        
        # Always ensure we have the necessary external function declarations
        required_functions = ['abort', 'cos', 'exit', 'free', 'malloc', 'pow', 
                             'printf', 'putchar', 'puts', 'rand', 'scanf', 'sin', 
                             'sqrt', 'srand', 'strcmp', 'strcpy', 'strlen']
        
        # Add existing external declarations first
        existing_externs = set()
        if sections.get('externals'):
            for line in sections['externals']:
                if line.strip() and line.strip().startswith('extern'):
                    output.append(line)
                    # Extract function name from extern declaration
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        existing_externs.add(parts[1])
        
        # Add any missing required external functions
        for func in required_functions:
            if func not in existing_externs:
                output.append(f"extern {func}")
        
        # Debug: force at least some extern declarations for testing
        from ..utils.colors import print_debug
        if not existing_externs:
            print_debug(f"No existing externs found, adding all required functions: {required_functions}")
        else:
            print_debug(f"Found existing externs: {existing_externs}")
        
        output.append("")
        
        # Data section - organized
        # Detect string declarations that were mistakenly placed in .bss and
        # move them into the data section so strings are defined in .data.
        misplaced_strings = []
        if sections.get('bss'):
            for line in sections['bss']:
                stripped = line.strip()
                if ' db ' in stripped and (stripped.startswith('str_') or stripped.startswith('LC')):
                    misplaced_strings.append(f"    {stripped}")

        data_section = self._format_data_section(sections['data'])
        # If we have misplaced strings, insert them after the 'section .data' header
        if misplaced_strings:
            insert_index = 1 if data_section and data_section[0].strip().startswith('section .data') else 0
            for i, string_line in enumerate(misplaced_strings):
                data_section.insert(insert_index + i, string_line)

        output.extend(data_section)
        output.append("")

        # BSS section - uninitialized data (string lines removed by formatter)
        if sections.get('bss') and any(line.strip() for line in sections['bss']):
            output.extend(self._format_bss_section(sections['bss']))
            output.append("")
        
        # Text section with context-aware C code placement
        output.extend(self._format_text_section_with_context(sections, ast))
        
        return '\n'.join(output)
    
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
        print_debug(f"Collected {len(self.collected_strings)} string literals")
    
    def _collect_for_loop_counters(self, ast: ProgramNode):
        """Pre-scan AST to collect all for loop counter variables"""
        def visit_statements(statements):
            for stmt in statements:
                if isinstance(stmt, IfNode):
                    visit_statements(stmt.if_body)
                    if hasattr(stmt, 'else_body') and stmt.else_body:
                        visit_statements(stmt.else_body)
                elif isinstance(stmt, WhileNode):
                    visit_statements(stmt.body)
                elif isinstance(stmt, ForNode):
                    # Generate deterministic counter variable name
                    counter_hash = hash(f"{stmt.variable}_{stmt.count}") % 1000
                    if counter_hash < 0:
                        counter_hash = -counter_hash
                    counter_var = f"for_counter_{counter_hash}"
                    
                    # Add to extra variables for data section
                    self.extra_variables.append(f"    {counter_var} dd 0")
                    
                    # Visit nested statements
                    visit_statements(stmt.body)
        
        visit_statements(ast.statements)
        print_debug(f"Pre-collected {len(self.extra_variables)} for loop counters")
    
    def _extract_metadata(self, assembly_code: str):
        """Extract metadata from the assembly code"""
        lines = assembly_code.split('\n')
        
        for line in lines:
            # Extract string constants
            if '.ascii' in line or 'db "' in line:
                self._extract_string_constant(line)
            
            # Extract C variable references
            if 'extern' in line:
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
        # Match: extern <type> <label>;
        match = re.search(r'extern\s+(\w+)\s+(\w+)', line)
        if match:
            var_type, var_label = match.groups()
            # Find original CASM variable name by matching label values
            for casm_var, label_info in self.variable_map.items():
                # label_info may be a dict or a string
                if isinstance(label_info, dict) and label_info.get('label') == var_label:
                    matched = casm_var
                elif isinstance(label_info, str) and label_info == var_label:
                    matched = casm_var
                else:
                    matched = None

                if matched:
                    self.c_variables[var_label] = {
                        'casm_name': matched,
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
            'bss': [],
            'text': [],
            'gcc_blocks': []
        }
        
        current_section = 'header'
        current_gcc_block = None
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('extern'):
                sections['externals'].append(line)
                continue
            
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
            elif stripped.startswith('section .data'):
                current_section = 'data'
                sections['data'].append(line)
            elif stripped.startswith('section .bss'):
                current_section = 'bss'
                sections['bss'].append(line)
            elif stripped.startswith('section .text'):
                current_section = 'text'
                sections['text'].append(line)
            elif current_section in sections:
                # If we're in the bss section but see a string/data declaration
                # that looks like an initialized string (db), treat it as data
                # so it will be emitted under section .data instead of .bss.
                if current_section == 'bss':
                    if ' db ' in stripped and (stripped.startswith('str_') or stripped.startswith('LC')):
                        sections['data'].append(line)
                        continue
                sections[current_section].append(line)
            else:
                # Default to header if no section is set
                sections['header'].append(line)
        
        return sections
    
    def _rebuild_with_context(self, sections: Dict, ast: ProgramNode) -> str:
        """Rebuild assembly with proper context and structure"""
        output = []
        
        # Header section
        output.extend(self._format_header(sections['header']))
        output.append("")
        
        # External declarations - cleaned up
        output.append("; External function declarations")
        
        # Always ensure we have the necessary external function declarations
        required_functions = ['abort', 'cos', 'exit', 'free', 'malloc', 'pow', 
                             'printf', 'putchar', 'puts', 'rand', 'scanf', 'sin', 
                             'sqrt', 'srand', 'strcmp', 'strcpy', 'strlen']
        
        # Add existing external declarations first
        existing_externs = set()
        if sections['externals']:
            for line in sections['externals']:
                if line.strip() and line.strip().startswith('extern'):
                    output.append(line)
                    # Extract function name from extern declaration
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        existing_externs.add(parts[1])
        
        # Add any missing required external functions
        for func in required_functions:
            if func not in existing_externs:
                output.append(f"extern {func}")
        
        # Debug: force at least some extern declarations for testing
        from ..utils.colors import print_debug
        if not existing_externs:
            print_debug(f"No existing externs found, adding all required functions: {required_functions}")
        else:
            print_debug(f"Found existing externs: {existing_externs}")
        
        output.append("")
        
        # Data section - organized, including any misplaced strings from BSS
        misplaced_strings = []
        if sections.get('bss'):
            for line in sections['bss']:
                stripped = line.strip()
                if ' db ' in stripped and (stripped.startswith('str_') or stripped.startswith('LC')):
                    misplaced_strings.append(f"    {stripped}")
        
        data_section = self._format_data_section(sections['data'])
        if misplaced_strings:
            # Insert misplaced strings into the data section
            data_insert_index = 1  # After "section .data"
            for i, string_line in enumerate(misplaced_strings):
                data_section.insert(data_insert_index + i, string_line)
        
        output.extend(data_section)
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
                    formatted.append(f"    {stripped}")
            elif not in_data_section:
                formatted.append(line)
        
        # Add extra variables (like for loop counters) to the data section
        if self.extra_variables:
            for var_line in self.extra_variables:
                formatted.append(var_line)
        
        # Add additional data items that might be needed (string constants, loop counters)
        self._add_generated_data_items(formatted)
        
        # Add all collected strings at the end of data section
        for string_label, content in self.collected_strings.items():
            formatted.append(f"    {string_label} db {content}")
        
        return formatted
    
    def _format_bss_section(self, bss_lines: List[str]) -> List[str]:
        """Format the BSS section with proper organization"""
        formatted = []
        in_bss_section = False
        
        for line in bss_lines:
            stripped = line.strip()
            
            if stripped.startswith('section .bss'):
                formatted.append("section .bss")
                in_bss_section = True
                continue
            
            if in_bss_section and stripped:
                # Check if this is a string declaration that belongs in .data section
                if ' db ' in stripped and (stripped.startswith('str_') or stripped.startswith('LC')):
                    # This is a string declaration that was incorrectly placed in .bss
                    # We'll move it to the data section in the main rebuild method
                    continue
                elif stripped.startswith('var_') and ' resb ' in stripped:
                    # This is a proper BSS declaration (uninitialized buffer)
                    formatted.append(f"    {stripped}")
                elif stripped.startswith('var_'):
                    # Other variable declarations
                    formatted.append(f"    {stripped}")
                else:
                    # Other BSS items (but not string data)
                    formatted.append(f"    {stripped}")
            elif not in_bss_section:
                formatted.append(line)
        
        return formatted
    
    def _extract_labels(self, assembly_code: str) -> Dict:
        """Extract existing control flow labels from assembly"""
        labels = {}
        lines = assembly_code.split('\n')
        
        # Look for control flow labels (if/else/while/for)
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith(';'):
                label = line.replace(':', '').strip()
                if any(prefix in label for prefix in ['if_end_', 'else_', 'while_start_', 'while_end_', 'for_start_', 'for_end_']):
                    # Extract the number from the label 
                    parts = label.split('_')
                    if len(parts) >= 3:
                        try:
                            label_num = int(parts[-1])
                            label_type = '_'.join(parts[:-1])
                            if label_type not in labels:
                                labels[label_type] = []
                            labels[label_type].append((label_num, label))
                        except ValueError:
                            pass
        
        return labels

    def _add_generated_data_items(self, formatted: List[str]):
        """Add generated data items like string constants and loop counters"""
        # Add any extra variables that were collected during processing
        if hasattr(self, 'extra_variables') and self.extra_variables:
            for var_line in self.extra_variables:
                formatted.append(var_line)
    
    def _format_text_section_with_context(self, sections: Dict, ast: ProgramNode) -> List[str]:
        """Format text section with context-aware C code placement"""
        # Debug: Check what's in the text section
        from ..utils.colors import print_debug
        print_debug(f"Text section has {len(sections.get('text', []))} lines")
        if sections.get('text'):
            print_debug(f"First few text lines: {sections['text'][:5]}")

        # Check if we have C code placeholders
        text_content = sections.get('text', [])
        has_placeholders = any('CASM_BLOCK_' in line and 'Placeholder' in line for line in text_content)
        print_debug(f"Has C placeholders: {has_placeholders}")
        
        # Use the original text section but insert C code where needed
        if 'text' in sections and sections['text']:
            formatted = []
            text_lines = sections['text']
            gcc_blocks = sections.get('gcc_blocks', [])
            
            # Find C code placeholders and replace them
            block_index = 0
            for line in text_lines:
                if 'CASM_BLOCK_' in line and 'Placeholder' in line:
                    # Insert actual C code assembly
                    if block_index < len(gcc_blocks):
                        formatted.append(f"    ; C code block: CASM_BLOCK_{block_index}")
                        formatted.extend(gcc_blocks[block_index]['assembly'])
                        block_index += 1
                    else:
                        formatted.append(line)  # Keep placeholder if no C code
                else:
                    formatted.append(line)
            
            # Remove duplicate labels from the formatted assembly
            from ..utils.colors import print_debug
            print_debug("About to remove duplicate labels")
            cleaned_assembly = self._remove_duplicate_labels('\n'.join(formatted))
            print_debug("Finished removing duplicate labels")
            return cleaned_assembly.split('\n')
        else:
            from ..utils.colors import print_debug
            print_debug("No text section found, falling back to AST processing")
            # Fallback to AST processing if no original text section
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
            # Generate actual variable initialization assembly (remove verbose comments)
            var_label = self.variable_map.get(stmt.name, f"var_{stmt.name}")
            try:
                # Try to parse as integer
                value = int(stmt.value)
                output.append(f"    mov dword [rel {var_label}], {value}")
            except ValueError:
                # Handle variable references or expressions
                if stmt.value in self.variable_map:
                    src_label = self.variable_map.get(stmt.value, f"var_{stmt.value}")
                    output.append(f"    mov eax, dword [rel {src_label}]")
                    output.append(f"    mov dword [rel {var_label}], eax")
                # Skip complex expression comments
            
        elif isinstance(stmt, AssignmentNode):
            # Generate assignment assembly
            var_label = self.variable_map.get(stmt.name, f"var_{stmt.name}")
            expression = stmt.value.strip()
            
            # Handle simple arithmetic like "i + 1" or "i - 1"
            if '+' in expression:
                parts = expression.split('+')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Load left operand
                    if left == stmt.name:
                        output.append(f"    mov eax, dword [rel {var_label}]")
                    elif left.isdigit():
                        output.append(f"    mov eax, {left}")
                    else:
                        left_label = self.variable_map.get(left, f"var_{left}")
                        output.append(f"    mov eax, dword [rel {left_label}]")
                    
                    # Add right operand
                    if right.isdigit():
                        output.append(f"    add eax, {right}")
                    else:
                        right_label = self.variable_map.get(right, f"var_{right}")
                        output.append(f"    add eax, dword [rel {right_label}]")
                    
                    # Store result
                    output.append(f"    mov dword [rel {var_label}], eax")
            elif expression.isdigit():
                # Simple numeric assignment
                output.append(f"    mov dword [rel {var_label}], {expression}")
            else:
                # Variable assignment
                if expression in self.variable_map:
                    src_label = self.variable_map.get(expression, f"var_{expression}")
                    output.append(f"    mov eax, dword [rel {src_label}]")
                    output.append(f"    mov dword [rel {var_label}], eax")
            
        elif isinstance(stmt, IfNode):
            # Use existing labels if available, otherwise generate new ones
            if hasattr(self, 'existing_labels') and 'if_end' in self.existing_labels:
                # Find the next available label pair from existing labels
                if_pairs = []
                for label_type in ['if_end', 'else']:
                    if label_type in self.existing_labels:
                        if_pairs.extend(self.existing_labels[label_type])
                
                # Use the control_counter to index into existing labels
                if if_pairs:
                    # Sort by label number and use the current counter
                    if_pairs.sort(key=lambda x: x[0])
                    if self.control_counter < len(if_pairs):
                        # Find corresponding else label if it exists
                        current_if_num = if_pairs[self.control_counter][0] if self.control_counter < len(if_pairs) else self.control_counter + 1
                        end_label = f"if_end_{current_if_num}"
                        else_label = f"else_{current_if_num}" if stmt.else_body and 'else' in self.existing_labels else None
                    else:
                        # Fallback to generated labels
                        self.control_counter += 1
                        end_label = f"if_end_{self.control_counter}"
                        else_label = f"else_{self.control_counter}" if stmt.else_body else None
                else:
                    # Fallback to generated labels
                    self.control_counter += 1
                    end_label = f"if_end_{self.control_counter}"
                    else_label = f"else_{self.control_counter}" if stmt.else_body else None
            else:
                # Generate new labels if no existing ones found
                self.control_counter += 1
                end_label = f"if_end_{self.control_counter}"
                else_label = f"else_{self.control_counter}" if stmt.else_body else None
            
            output.append(f"    ; if {stmt.condition}")
            
            # Generate condition evaluation assembly
            condition_asm, jump_instruction = self._generate_condition_assembly(stmt.condition)
            output.extend(condition_asm)
            
            if else_label:
                output.append(f"    {jump_instruction} {else_label}")
            else:
                output.append(f"    {jump_instruction} {end_label}")
            
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
            # Generate consistent labels
            self.control_counter += 1
            start_label = f"while_start_{self.control_counter}"
            end_label = f"while_end_{self.control_counter}"
            
            output.append(f"    ; while {stmt.condition}")
            output.append(f"{start_label}:")
            
            # Generate condition evaluation assembly
            condition_asm, jump_instruction = self._generate_condition_assembly(stmt.condition)
            output.extend(condition_asm)
            output.append(f"    {jump_instruction} {end_label}")
            
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
            # Use a deterministic counter based on loop count and variable name
            # This ensures consistency across different compilation phases
            counter_hash = hash(f"{stmt.variable}_{stmt.count}") % 1000  # Keep it small
            if counter_hash < 0:
                counter_hash = -counter_hash
            
            start_label = f"for_start_{counter_hash}"
            end_label = f"for_end_{counter_hash}"
            counter_var = f"for_counter_{counter_hash}"
            
            output.append(f"    ; for {stmt.variable} in range({stmt.count})")
            
            # Initialize counter (variable already defined in data section)
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
            # Generate consistent format label
            self.control_counter += 1
            format_label = f"fmt_scanf_{self.control_counter}"
            var_label = self.variable_map.get(stmt.variable, f"var_{stmt.variable}")
            
            # Generate scanf call assembly (Windows x64 calling convention)
            output.append(f"    lea rcx, [rel {format_label}]")
            output.append(f"    lea rdx, [rel {var_label}]")
            output.append(f"    call scanf")
            
        elif isinstance(stmt, CCodeBlockNode):
            # Handle the new marker-based C compilation system
            indent = self.base_indent * (self.indent_level + 1)
            
            # Get the actual compiled assembly for this block
            block_id = getattr(stmt, '_block_id', None)
            if block_id is not None:
                marker_key = f"CASM_BLOCK_{block_id}"
                if marker_key in self.c_assembly_segments:
                    assembly_code = self.c_assembly_segments[marker_key]
                    if assembly_code.strip():
                        # Add assembly instructions directly without extra comments
                        for asm_line in assembly_code.split('\n'):
                            if asm_line.strip():
                                output.append(f"{indent}{asm_line}")
                    # Don't add anything for empty assembly blocks
                # Don't add anything for missing assembly
            # Don't add anything for missing block ID
        
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
    
    def _generate_condition_assembly(self, condition: str) -> tuple[List[str], str]:
        """Generate assembly code for condition evaluation and return the appropriate jump instruction"""
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
                    var_label = self.variable_map.get(left, f"var_{left}")
                    asm_lines.append(f"    mov eax, dword [rel {var_label}]")
                elif left.isdigit():
                    asm_lines.append(f"    mov eax, {left}")
                else:
                    asm_lines.append(f"    ; TODO: Complex left operand: {left}")
                    asm_lines.append(f"    mov eax, 0  ; placeholder")
                
                # Generate assembly for right operand and comparison
                if right in self.variable_map:
                    var_label = self.variable_map.get(right, f"var_{right}")
                    asm_lines.append(f"    cmp eax, dword [rel {var_label}]")
                elif right.isdigit():
                    asm_lines.append(f"    cmp eax, {right}")
                else:
                    asm_lines.append(f"    ; TODO: Complex right operand: {right}")
                    asm_lines.append(f"    cmp eax, 0  ; placeholder")
                
                return asm_lines, comparison_ops[op]
        
        # Fallback for simple boolean conditions
        if condition in self.variable_map:
            var_label = self.variable_map.get(condition, f"var_{condition}")
            asm_lines.append(f"    mov eax, dword [rel {var_label}]")
            asm_lines.append(f"    cmp eax, 0")
        elif condition.isdigit():
            asm_lines.append(f"    mov eax, {condition}")
            asm_lines.append(f"    cmp eax, 0")
        else:
            asm_lines.append(f"    ; TODO: Complex condition: {condition}")
            asm_lines.append(f"    mov eax, 1  ; assume true")
            asm_lines.append(f"    cmp eax, 0")
        
        return asm_lines, 'je'
    
    def _remove_duplicate_labels(self, assembly_text: str) -> str:
        """Remove duplicate label definitions from assembly code"""
        lines = assembly_text.split('\n')
        seen_labels = set()
        filtered_lines = []
        duplicate_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Check if this is a label definition (ends with ':' and doesn't contain other content)
            if ':' in stripped and not stripped.startswith(';'):
                # Extract just the label part (before the colon)
                potential_label = stripped.split(':')[0].strip()
                
                # If this looks like a control flow label (else_, if_end_, etc.)
                if any(pattern in potential_label for pattern in ['else_', 'if_end_', 'while_', 'for_']):
                    if potential_label in seen_labels:
                        # Skip this duplicate label
                        print_info(f"Removing duplicate label: {potential_label}")
                        duplicate_count += 1
                        continue
                    else:
                        seen_labels.add(potential_label)
                        print_info(f"Keeping first occurrence of label: {potential_label}")
            
            filtered_lines.append(line)
        
        from ..utils.colors import print_debug
        print_debug(f"Removed {duplicate_count} duplicate labels")
        return '\n'.join(filtered_lines)

# Global instance
pretty_printer = CASMPrettyPrinter()