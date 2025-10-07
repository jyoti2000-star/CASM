#!/usr/bin/env python3

import platform
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from parser import (
    ASTNode, ProgramNode, IfNode, WhileNode, DoWhileNode, ForNode,
    SwitchNode, CaseNode, DefaultNode, FunctionNode, CallNode, ReturnNode,
    VarNode, ConstNode, PrintNode, InputNode, StdlibCallNode, ExitNode,
    BreakNode, ContinueNode, AssemblyNode, CommentNode, SectionNode
)

# Cross-platform configuration
class PlatformDetector:
    """Automatically detect and configure platform-specific settings"""
    
    @staticmethod
    def detect_platform():
        """Auto-detect the current platform (Windows only, x86-64 only)"""
        import platform as plt
        
        # Only support Windows
        system = plt.system().lower()
        if system == 'windows':
            os_name = 'windows'
        else:
            os_name = 'windows'  # Default to Windows for all other systems
        
        # Force x86-64 architecture only
        arch = 'x86_64'
        
        return os_name, arch
    
    @staticmethod
    def get_platform_config(os_name=None, arch=None):
        """Get platform configuration, auto-detecting if not specified"""
        if os_name is None or arch is None:
            detected_os, detected_arch = PlatformDetector.detect_platform()
            os_name = os_name or detected_os
            arch = arch or detected_arch
        
        return Platform(os_name, arch)

# Platform-specific configurations
class Platform:
    """Enhanced platform-specific configuration with auto-detection"""
    def __init__(self, os_name: str = None, architecture: str = None):
        # Auto-detect if not specified
        if os_name is None:
            detected_os, detected_arch = PlatformDetector.detect_platform()
            self.os_name = os_name or detected_os
        else:
            self.os_name = os_name.lower()
        
        # Force x86-64 architecture only
        self.architecture = 'x86_64'
        
        # Set x86-64 defaults
        self.word_size = 8
        self.pointer_size = 8
        self.reg_prefix = 'r'  # 64-bit registers
        
        self.addressing_mode = 'rip_relative'  # Default to modern addressing
        self.supports_32bit_abs = False  # Most modern platforms don't
        
        # Configure based on OS
        self._configure_platform()
    
    def _configure_platform(self):
        """Configure platform-specific settings"""
        if self.os_name == 'windows':
            self._configure_windows()
        else:
            self._configure_windows()  # Default to Windows for all systems
    

        

    
    def _configure_windows(self):
        """Configure for Windows x86-64"""
        self.supports_32bit_abs = False
        self.addressing_mode = 'rip_relative'
        self.registers = {
            'syscall_num': 'rax',
            'arg1': 'rcx', 'arg2': 'rdx', 'arg3': 'r8',  # Windows x64 calling convention
            'arg4': 'r9', 'arg5': 'rsp+32', 'arg6': 'rsp+40',
            'return': 'rax', 'stack_ptr': 'rsp', 'frame_ptr': 'rbp'
        }
        # Windows native syscalls (NT system calls)
        self.syscalls = {
            'sys_read': '3', 'sys_write': '4', 'sys_exit': '44'  # NtReadFile, NtWriteFile, NtTerminateProcess
        }
        self.object_format = 'win64'
        self.section_prefix = 'section'
        self.data_directive = 'section .data'
        self.text_directive = 'section .text'
        self.string_directive = 'db'
        self.quad_directive = 'dq'
        
        self.calling_convention = 'Microsoft x64'
        self.entry_point = 'main'
        self.global_directive = 'global main'
    

    
    def get_syscall_instruction(self):
        """Get the appropriate syscall instruction for x86-64"""
        return 'syscall'
    
    def get_register_for_purpose(self, purpose: str):
        """Get register name for a specific purpose"""
        return self.registers.get(purpose, 'rax')  # Default fallback
    
    def adapt_instruction(self, instruction: str, operands: List[str]):
        """Adapt instruction for x86-64 (no adaptation needed)"""
        return f"{instruction} {', '.join(operands)}"



@dataclass
class Context:
    """Maintains compilation context"""
    label_counter: int = 0
    loop_stack: List[Tuple[str, str]] = None
    string_literals: Dict[str, str] = field(default_factory=dict)
    variables: Dict[str, int] = field(default_factory=dict)
    functions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    loop_variables: Dict[str, str] = field(default_factory=dict)  # Maps loop var names to memory locations
    
    def __post_init__(self):
        if self.loop_stack is None:
            self.loop_stack = []
    
    def new_label(self, prefix: str = "L") -> str:
        """Generate unique label"""
        self.label_counter += 1
        return f"__{prefix}{self.label_counter}"
    
    def push_loop(self, continue_lbl: str, break_lbl: str):
        self.loop_stack.append((continue_lbl, break_lbl))
    
    def pop_loop(self):
        if self.loop_stack:
            self.loop_stack.pop()
    
    def get_current_loop(self) -> Tuple[str, str]:
        if self.loop_stack:
            return self.loop_stack[-1]
        return None, None
    
    def set_loop_variable(self, loop_var: str, memory_location: str):
        """Map a loop variable name to its memory location"""
        self.loop_variables[loop_var] = memory_location
    
    def get_loop_variable(self, loop_var: str) -> str:
        """Get the memory location for a loop variable"""
        return self.loop_variables.get(loop_var, loop_var)

class CodeGenerator:
    def __init__(self, target_os: str = None, target_arch: str = None):
        self.ctx = Context()
        self.output_lines = []
        self.data_section = []
        self.bss_section = []
        self.stdlib_used = set()
        self.function_calls = set()  # Track function calls for automatic extern generation
        
        # Use enhanced platform detection (x86-64 only)
        if target_os is None:
            # Auto-detect current platform, but use Windows as default when on macOS
            detected_os, detected_arch = PlatformDetector.detect_platform()
            # If detected OS is not supported (like macOS), default to Windows
            if detected_os not in ['windows']:
                detected_os = 'windows'
                print(f"[Auto-detected] Running on unsupported platform, defaulting to: windows-x86_64")
            else:
                print(f"[Auto-detected] Target: {detected_os}-x86_64")
            self.platform = Platform(detected_os, 'x86_64')
        else:
            # Use specified target
            self.platform = Platform(target_os, 'x86_64')
            print(f"[Target] Target: {target_os}-x86_64")
        
        # Add universal compatibility directives
        self._add_compatibility_directives()
    
    def _add_compatibility_directives(self):
        """Add directives for cross-platform compatibility"""
        # Platform-specific assembly directives will be added in generate()
        pass
    
    def add_function_call(self, function_name: str):
        """Track a function call for automatic extern generation"""
        self.function_calls.add(function_name)
    
    def generate(self, ast: ProgramNode) -> str:
        """Generate assembly code from AST"""
        # Process the AST
        self.visit(ast)
        
        # Assemble final output using the same format as c_codegen.py
        result = []
        result.append(f"; Generated by Advanced High-Level Assembly Preprocessor")
        result.append(f"; Target platform: {self.platform.os_name} ({self.platform.calling_convention})")
        result.append("")
        
        # Add global directive for entry point
        result.append(self.platform.global_directive)
        result.append("")
        
        # Add function call tracking comment for debugging
        if self.function_calls:
            result.append(f"; Function calls detected: {', '.join(sorted(self.function_calls))}")
            result.append("")
        
        # Add standard library functions if used
        if self.stdlib_used:
            result.extend(self.generate_stdlib())
            result.append("")
        
        # Only create data section if we have data and there's no existing one
        if self.data_section or self.bss_section:
            result.append("; DATA START")
            
            # Add data section content
            if self.data_section:
                result.extend(self.data_section)
            
            # Add bss section content 
            if self.bss_section:
                result.extend(self.bss_section)
            
            result.append("; DATA END")
            result.append("")
        
        # Add entry point
        result.append(f"{self.platform.entry_point}:")
        
        # Add platform-specific entry point setup
        if self.platform.os_name == 'windows':
            # Windows x64 ABI requires 16-byte stack alignment
            result.append("    ; Set up stack frame and ensure 16-byte alignment")
            result.append("    push rbp")
            result.append("    mov rbp, rsp")
            result.append("    sub rsp, 48  ; Reserve shadow space (32) + space for 5th param (8) + alignment (8)")
        
        # Add code
        result.extend(self.output_lines)
        
        # Add platform-specific exit if no explicit exit in code
        if not any('exit' in line.lower() for line in self.output_lines):
            result.extend(self._generate_default_exit())
        
        return "\n".join(result)
    
    def visit(self, node: ASTNode):
        """Visit an AST node and generate code"""
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node: ASTNode):
        """Default visitor for unknown nodes"""
        pass
    
    def visit_ProgramNode(self, node: ProgramNode):
        """Visit program node"""
        for statement in node.statements:
            self.visit(statement)
    
    def visit_IfNode(self, node: IfNode):
        """Visit if node"""
        else_label = self.ctx.new_label("else")
        end_label = self.ctx.new_label("endif")
        
        self.add_line(f"; %if {node.condition}")
        self.emit_condition_check(node.condition, else_label, False)
        
        for stmt in node.if_block:
            self.visit(stmt)
        
        if node.else_block:
            self.add_line(f"    jmp {end_label}")
            self.add_line(f"{else_label}:")
            for stmt in node.else_block:
                self.visit(stmt)
            self.add_line(f"{end_label}:")
        else:
            self.add_line(f"{else_label}:")
    
    def visit_WhileNode(self, node: WhileNode):
        """Visit while node"""
        start_label = self.ctx.new_label("while_start")
        end_label = self.ctx.new_label("while_end")
        
        self.ctx.push_loop(start_label, end_label)
        
        self.add_line(f"; %while {node.condition}")
        self.add_line(f"{start_label}:")
        self.emit_condition_check(node.condition, end_label, False)
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.add_line(f"    jmp {start_label}")
        self.add_line(f"{end_label}:")
        
        self.ctx.pop_loop()
    
    def visit_DoWhileNode(self, node: DoWhileNode):
        """Visit do-while node"""
        start_label = self.ctx.new_label("do_start")
        end_label = self.ctx.new_label("do_end")
        
        self.ctx.push_loop(start_label, end_label)
        
        self.add_line(f"; %do...%while {node.condition}")
        self.add_line(f"{start_label}:")
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.emit_condition_check(node.condition, start_label, True)
        self.add_line(f"{end_label}:")
        
        self.ctx.pop_loop()
    
    def visit_ForNode(self, node: ForNode):
        """Visit for node"""
        start_label = self.ctx.new_label("for_start")
        end_label = self.ctx.new_label("for_end")
        
        self.ctx.push_loop(start_label, end_label)
        
        # Create a memory location for the loop variable
        loop_var_name = f"__loop_var_{node.variable}"
        if loop_var_name not in self.ctx.variables:
            self.data_section.append(f"    {loop_var_name} dq 0  ; Loop variable for {node.variable}")
            self.ctx.variables[loop_var_name] = 0
        
        # Map the loop variable name to its memory location
        self.ctx.set_loop_variable(node.variable, loop_var_name)
        
        # For loop implementation
        self.add_line(f"; %for {node.variable} in range {node.count_expr}")
        
        # x86-64 implementation only
        self.add_line(f"    mov qword [{loop_var_name}], 0  ; Initialize {node.variable}")
        self.add_line(f"{start_label}:")
        self.add_line(f"    mov rax, [{loop_var_name}]")
        self.add_line(f"    cmp rax, {node.count_expr}")
        self.add_line(f"    jge {end_label}")
        
        for stmt in node.body:
            self.visit(stmt)
        
        # x86-64 increment only
        self.add_line(f"    mov rax, [{loop_var_name}]")
        self.add_line(f"    add rax, 1")
        self.add_line(f"    mov [{loop_var_name}], rax  ; Increment {node.variable}")
        self.add_line(f"    jmp {start_label}")
        
        self.add_line(f"{end_label}:")
        
        self.ctx.pop_loop()
    
    def visit_SwitchNode(self, node: SwitchNode):
        """Visit switch node"""
        end_label = self.ctx.new_label("switch_end")
        
        self.add_line(f"; %switch {node.expression}")
        
        # Generate case comparisons
        for case in node.cases:
            case_label = self.ctx.new_label("case")
            self.add_line(f"    cmp {node.expression}, {case.value}")
            self.add_line(f"    jne {case_label}")
            
            for stmt in case.body:
                self.visit(stmt)
            
            self.add_line(f"    jmp {end_label}")
            self.add_line(f"{case_label}:")
        
        # Generate default case if present
        if node.default_case:
            for stmt in node.default_case.body:
                self.visit(stmt)
        
        self.add_line(f"{end_label}:")
    
    def visit_FunctionNode(self, node: FunctionNode):
        """Visit function node"""
        self.add_line(f"; %function {node.name}")
        self.add_line(f"{node.name}:")
        
        # Platform-specific function prologue
        if self.platform.os_name == 'windows':
            self.add_line(f"    push rbp")
            self.add_line(f"    mov rbp, rsp")
            self.add_line(f"    sub rsp, 32  ; Shadow space for Windows x64")
        else:
            self.add_line(f"    push rbp")
            self.add_line(f"    mov rbp, rsp")
        
        for stmt in node.body:
            self.visit(stmt)
        
        # Platform-specific function epilogue
        if self.platform.os_name == 'windows':
            self.add_line(f"    add rsp, 32  ; Restore shadow space")
            self.add_line(f"    mov rsp, rbp")
            self.add_line(f"    pop rbp")
            self.add_line(f"    ret")
        else:
            self.add_line(f"    mov rsp, rbp")
            self.add_line(f"    pop rbp")
            self.add_line(f"    ret")
    
    def visit_CallNode(self, node: CallNode):
        """Visit call node"""
        self.add_function_call(node.name)  # Track the function call
        self.add_line(f"    call {node.name}  ; %call {node.name}")
    
    def visit_ReturnNode(self, node: ReturnNode):
        """Visit return node"""
        if self.platform.os_name == 'windows':
            self.add_line(f"    add rsp, 32  ; Restore shadow space")
        self.add_line(f"    mov rsp, rbp")
        self.add_line(f"    pop rbp")
        self.add_line(f"    ret  ; %return")
    
    def visit_VarNode(self, node: VarNode):
        """Visit variable declaration node"""
        if node.var_type == "array":
            # Array declaration with initialization
            if hasattr(node, 'init_values') and node.init_values:
                # Initialize array in data section
                init_str = ', '.join(str(v) for v in node.init_values)
                self.data_section.append(f"    {node.name}: {self.platform.quad_directive} {init_str}  ; %var {node.name}[{node.size}]")
            else:
                # Uninitialized array in bss section
                element_size = 8  # qword size
                self.bss_section.append(f"    {node.name} resb {node.size * element_size}  ; %var {node.name}[{node.size}]")
        elif node.var_type == "string":
            # String variable
            str_content = node.value[1:-1]  # Remove quotes
            self.data_section.append(f"    {node.name} {self.platform.string_directive} {node.value}, 0  ; %var {node.name} {node.value}")
        elif node.var_type == "number":
            # Numeric variable
            self.data_section.append(f"    {node.name}: {self.platform.quad_directive} {node.value}  ; %var {node.name} {node.value}")
        elif node.var_type == "boolean":
            # Boolean variable (0 for false, 1 for true)
            bool_value = "1" if node.value.lower() == "true" else "0"
            self.data_section.append(f"    {node.name} db {bool_value}  ; %var {node.name} {node.value}")
        else:
            # Default behavior - uninitialized variable
            self.bss_section.append(f"    {node.name} resq 1  ; %var {node.name}")
            if node.value:
                self.add_line(f"    mov {self.platform.registers['arg1']}, #{node.value}  ; initialize {node.name}")
                self.add_line(f"    adrp x9, {node.name}@PAGE")
                self.add_line(f"    add x9, x9, {node.name}@PAGEOFF")
                self.add_line(f"    str {self.platform.registers['arg1']}, [x9]")
    
    def visit_ConstNode(self, node: ConstNode):
        """Visit constant declaration node"""
        self.data_section.append(f"    {node.name} equ {node.value}  ; %const {node.name}")
    
    def visit_PrintNode(self, node: PrintNode):
        """Visit print node"""
        self._print_helper(node.argument, node.newline)
    
    def visit_InputNode(self, node: InputNode):
        """Visit input node"""
        self.add_line(f"    ; %input {node.destination}")
        self.add_line(f"    mov {self.platform.registers['syscall_num']}, {self.platform.syscalls['sys_read']}")
        self.add_line(f"    mov {self.platform.registers['arg1']}, 0")
        self.add_line(f"    mov {self.platform.registers['arg2']}, {node.destination}")
        self.add_line(f"    mov {self.platform.registers['arg3']}, 256")
        self.add_line(f"    syscall")
    
    def visit_StdlibCallNode(self, node: StdlibCallNode):
        """Visit standard library call node"""
        if node.function == 'scanf':
            self.process_scanf(node.arguments)
        elif node.function == 'strlen':
            self.process_strlen(node.arguments)
        elif node.function == 'strcpy':
            self.process_strcpy(node.arguments)
        elif node.function == 'strcmp':
            self.process_strcmp(node.arguments)
        elif node.function == 'memset':
            self.process_memset(node.arguments)
        elif node.function == 'memcpy':
            self.process_memcpy(node.arguments)
        elif node.function == 'atoi':
            self.process_atoi(node.arguments)
        elif node.function == 'itoa':
            self.process_itoa(node.arguments)
    
    def visit_ExitNode(self, node: ExitNode):
        """Visit exit node"""
        self.add_line(f"    ; %exit {node.code}")
        if self.platform.os_name == 'windows':
            # Windows API call to ExitProcess
            self.add_line(f"    mov rcx, {node.code}  ; exit code")
            self.add_function_call("ExitProcess")  # Track the function call
            self.add_line(f"    call ExitProcess")
        else:
            self.add_line(f"    mov {self.platform.registers['syscall_num']}, {self.platform.syscalls['sys_exit']}")
            self.add_line(f"    mov {self.platform.registers['arg1']}, {node.code}")
            self.add_line(f"    {self.platform.get_syscall_instruction()}")
    
    def visit_BreakNode(self, node: BreakNode):
        """Visit break node"""
        _, break_label = self.ctx.get_current_loop()
        if break_label:
            self.add_line(f"    jmp {break_label}  ; %break")
        else:
            self.add_line(f"    ; ERROR: %break outside loop")
    
    def visit_ContinueNode(self, node: ContinueNode):
        """Visit continue node"""
        continue_label, _ = self.ctx.get_current_loop()
        if continue_label:
            self.add_line(f"    jmp {continue_label}  ; %continue")
        else:
            self.add_line(f"    ; ERROR: %continue outside loop")
    
    def visit_AssemblyNode(self, node: AssemblyNode):
        """Visit assembly node"""
        instruction = node.instruction.strip()
        
        # Only x86-64 assembly
        self.add_line(instruction)
    
    def visit_CommentNode(self, node: CommentNode):
        """Visit comment node"""
        self.add_line(node.text)
    
    def visit_SectionNode(self, node: SectionNode):
        """Visit section node - skip text section as it's handled automatically"""
        # Skip text section directive as it's handled automatically
        if 'text' not in node.directive.lower():
            self.add_line(node.directive)
    
    def add_line(self, line: str):
        """Add line to output with automatic addressing fixes and variable substitution"""
        # Fix 32-bit absolute addressing for platforms that don't support it
        if hasattr(self, 'platform') and not self.platform.supports_32bit_abs:
            line = self._fix_addressing(line)
        
        # Replace loop variable references with their memory locations
        line = self._replace_loop_variables(line)
        
        self.output_lines.append(line)
    
    def _replace_loop_variables(self, line: str) -> str:
        """Replace loop variable references in assembly code"""
        for loop_var, memory_location in self.ctx.loop_variables.items():
            # Replace [variable_name] with [__loop_var_variable_name]
            pattern = f'[{loop_var}]'
            if pattern in line:
                line = line.replace(pattern, f'[{memory_location}]')
            
            # Also replace bare variable name references
            import re
            # Replace variable_name when it appears as a standalone word (not part of another identifier)
            pattern = rf'\b{re.escape(loop_var)}\b'
            replacement = memory_location
            line = re.sub(pattern, replacement, line)
            
        return line
    
    def _fix_addressing(self, line: str) -> str:
        """Fix addressing modes for platform compatibility"""
        line = line.strip()
        
        # Skip comments and labels
        if line.startswith(';') or line.endswith(':') or not line:
            return line
            
        # Fix complex array indexing that causes 32-bit absolute addressing
        if 'lea' in line.lower() and '[' in line and '+' in line:
            return self._fix_complex_lea(line)
        
        # Fix direct symbol references for Windows (RIP-relative addressing)
        if self.platform.os_name == 'windows':
            # Fix mov instructions with direct symbol references
            import re
            
            # 2. INTELLIGENT PARSING: Comprehensive register list to avoid false conversions
            registers = {
                # 64-bit registers
                'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
                # 32-bit registers
                'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
                'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
                # 16-bit registers
                'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
                'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
                # 8-bit registers
                'al', 'ah', 'bl', 'bh', 'cl', 'ch', 'dl', 'dh',
                'sil', 'dil', 'bpl', 'spl', 'r8b', 'r9b', 'r10b', 'r11b',
                'r12b', 'r13b', 'r14b', 'r15b'
            }
            
            # Pattern: mov reg, [rel reg] - this is incorrect, should be mov reg, [reg]
            mov_rel_reg_pattern = r'(\s*)mov\s+(\w+),\s+\[rel\s+(\w+)\]\s*(;.*)?$'
            match = re.match(mov_rel_reg_pattern, line)
            if match:
                indent, dest_reg, src_reg, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                # If source is a register, don't use RIP-relative
                if src_reg.lower() in registers:
                    return f"{indent}mov {dest_reg}, [{src_reg}]{comment}"
            
            # Pattern: mov reg, symbol_name (where symbol_name starts with letter/underscore and is not a register)
            mov_pattern = r'(\s*)mov\s+(\w+),\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(;.*)?$'
            match = re.match(mov_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                # Only convert if the symbol is not a register name
                if symbol.lower() not in registers:
                    return f"{indent}lea {reg}, [rel {symbol}]{comment}"
            
            # Pattern: lea reg, [rel symbol] - ensure this doesn't get double-converted
            lea_rel_pattern = r'(\s*)lea\s+(\w+),\s+\[rel\s+([a-zA-Z_][a-zA-Z0-9_]*)\]\s*(;.*)?$'
            match = re.match(lea_rel_pattern, line)
            if match:
                # Already has RIP-relative addressing, don't modify
                return line
            
            # Pattern: mov reg, [symbol_name] (with possible spaces)
            mov_mem_pattern = r'(\s*)mov\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(mov_mem_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                # Only apply RIP-relative if symbol is not a register
                if symbol.lower() not in registers:
                    return f"{indent}mov {reg}, [rel {symbol}]{comment}"
            
            # Pattern: add [symbol_name], reg - rewrite to avoid RIP-relative memory destination
            add_mem_pattern = r'(\s*)add\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\],\s+(\w+)\s*(;.*)?$'
            match = re.match(add_mem_pattern, line)
            if match:
                indent, symbol, reg, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                # Only apply fix if symbol is not a register
                if symbol.lower() not in registers:
                    # Rewrite as: mov temp_reg, [rel symbol]; add temp_reg, reg; mov [rel symbol], temp_reg
                    return f"{indent}mov r11, [rel {symbol}]  ; Load for add operation\n{indent}add r11, {reg}\n{indent}mov [rel {symbol}], r11{comment}"
            
            # Pattern: mov [symbol_name], reg (with possible spaces)
            mov_to_mem_pattern = r'(\s*)mov\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\],\s+(\w+)\s*(;.*)?$'
            match = re.match(mov_to_mem_pattern, line)
            if match:
                indent, symbol, reg, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                # Only apply RIP-relative if symbol is not a register
                if symbol.lower() not in registers:
                    return f"{indent}mov [rel {symbol}], {reg}{comment}"
            
            # Pattern: mov qword [symbol_name], immediate (with possible spaces)
            mov_qword_pattern = r'(\s*)mov\s+qword\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\],\s+(\w+)\s*(;.*)?$'
            match = re.match(mov_qword_pattern, line)
            if match:
                indent, symbol, immediate, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                return f"{indent}mov qword [rel {symbol}], {immediate}{comment}"
            
            # 3. COMPLETE COVERAGE: Additional instruction patterns
            
            # Pattern: sub [symbol], reg → complex rewrite
            sub_mem_pattern = r'(\s*)sub\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\],\s+(\w+)\s*(;.*)?$'
            match = re.match(sub_mem_pattern, line)
            if match:
                indent, symbol, reg, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}mov r11, [rel {symbol}]  ; Load for sub operation\n{indent}sub r11, {reg}\n{indent}mov [rel {symbol}], r11{comment}"
            
            # Pattern: inc [symbol] → complex rewrite
            inc_mem_pattern = r'(\s*)inc\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(inc_mem_pattern, line)
            if match:
                indent, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}mov r11, [rel {symbol}]  ; Load for inc operation\n{indent}inc r11\n{indent}mov [rel {symbol}], r11{comment}"
            
            # Pattern: dec [symbol] → complex rewrite
            dec_mem_pattern = r'(\s*)dec\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(dec_mem_pattern, line)
            if match:
                indent, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}mov r11, [rel {symbol}]  ; Load for dec operation\n{indent}dec r11\n{indent}mov [rel {symbol}], r11{comment}"
            
            # Pattern: cmp reg, [symbol] → cmp reg, [rel symbol]
            cmp_mem_pattern = r'(\s*)cmp\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(cmp_mem_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}cmp {reg}, [rel {symbol}]{comment}"
            
            # Pattern: cmp [symbol], reg → cmp [rel symbol], reg
            cmp_mem2_pattern = r'(\s*)cmp\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\],\s+(\w+)\s*(;.*)?$'
            match = re.match(cmp_mem2_pattern, line)
            if match:
                indent, symbol, reg, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}cmp [rel {symbol}], {reg}{comment}"
            
            # Pattern: cmp qword/dword/word/byte [symbol], immediate → cmp qword/dword/word/byte [rel symbol], immediate
            cmp_sized_pattern = r'(\s*)cmp\s+(qword|dword|word|byte)\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\],\s+(\w+)\s*(;.*)?$'
            match = re.match(cmp_sized_pattern, line)
            if match:
                indent, size, symbol, immediate, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}cmp {size} [rel {symbol}], {immediate}{comment}"
            
            # Pattern: cmp [symbol], immediate (without size specifier) → cmp qword [rel symbol], immediate
            cmp_no_size_pattern = r'(\s*)cmp\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\],\s+(\w+)\s*(;.*)?$'
            match = re.match(cmp_no_size_pattern, line)
            if match:
                indent, symbol, immediate, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}cmp qword [rel {symbol}], {immediate}{comment}"
            
            # Pattern: test reg, [symbol] → test reg, [rel symbol]
            test_mem_pattern = r'(\s*)test\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(test_mem_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}test {reg}, [rel {symbol}]{comment}"
            
            # Pattern: push [symbol] → push qword [rel symbol]
            push_mem_pattern = r'(\s*)push\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(push_mem_pattern, line)
            if match:
                indent, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}push qword [rel {symbol}]{comment}"
            
            # Pattern: mul [symbol] → mul qword [rel symbol]
            mul_mem_pattern = r'(\s*)mul\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(mul_mem_pattern, line)
            if match:
                indent, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}mul qword [rel {symbol}]{comment}"
            
            # Pattern: div [symbol] → div qword [rel symbol]
            div_mem_pattern = r'(\s*)div\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(div_mem_pattern, line)
            if match:
                indent, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}div qword [rel {symbol}]{comment}"
            
            # Pattern: add reg, [symbol] → add reg, [rel symbol]
            add_reg_mem_pattern = r'(\s*)add\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(add_reg_mem_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}add {reg}, [rel {symbol}]{comment}"
            
            # Pattern: sub reg, [symbol] → sub reg, [rel symbol]
            sub_reg_mem_pattern = r'(\s*)sub\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(sub_reg_mem_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}sub {reg}, [rel {symbol}]{comment}"
            
            # Pattern: and reg, [symbol] → and reg, [rel symbol]
            and_reg_mem_pattern = r'(\s*)and\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(and_reg_mem_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}and {reg}, [rel {symbol}]{comment}"
            
            # Pattern: or reg, [symbol] → or reg, [rel symbol]
            or_reg_mem_pattern = r'(\s*)or\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(or_reg_mem_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}or {reg}, [rel {symbol}]{comment}"
            
            # Pattern: xor reg, [symbol] → xor reg, [rel symbol]
            xor_reg_mem_pattern = r'(\s*)xor\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*(;.*)?$'
            match = re.match(xor_reg_mem_pattern, line)
            if match:
                indent, reg, symbol, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    return f"{indent}xor {reg}, [rel {symbol}]{comment}"
            
            # Pattern: mov [symbol + offset], reg → mov [rel symbol + offset], reg
            mov_array_to_pattern = r'(\s*)mov\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\+\s*(\w+)\s*\],\s+(\w+)\s*(;.*)?$'
            match = re.match(mov_array_to_pattern, line)
            if match:
                indent, symbol, offset, reg, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    # Convert to RIP-relative addressing with offset
                    return f"{indent}lea r10, [rel {symbol}]  ; Load base address\n{indent}mov [r10 + {offset}], {reg}{comment}"
            
            # Pattern: mov reg, [symbol + offset] → mov reg, [rel symbol + offset]
            mov_array_from_pattern = r'(\s*)mov\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\+\s*(\w+)\s*\]\s*(;.*)?$'
            match = re.match(mov_array_from_pattern, line)
            if match:
                indent, reg, symbol, offset, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    # Convert to RIP-relative addressing with offset
                    return f"{indent}lea r10, [rel {symbol}]  ; Load base address\n{indent}mov {reg}, [r10 + {offset}]{comment}"
            
            # Pattern: lea reg, [symbol + offset] → RIP-relative base + offset
            lea_array_pattern = r'(\s*)lea\s+(\w+),\s+\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\+\s*(\w+)\s*\]\s*(;.*)?$'
            match = re.match(lea_array_pattern, line)
            if match:
                indent, reg, symbol, offset, comment = match.groups()
                comment = f"  {comment}" if comment else ""
                if symbol.lower() not in registers:
                    # Convert to RIP-relative addressing with offset calculation
                    return f"{indent}lea {reg}, [rel {symbol}]  ; Load base address\n{indent}add {reg}, {offset}  ; Add offset{comment}"
            
            # 4. PE64 COMPATIBILITY: Already converted patterns - don't double-convert
            if '[rel ' in line:
                return line
            
        return line
    
    def _fix_complex_lea(self, line: str) -> str:
        """Fix LEA instructions with complex addressing"""
        # For platforms that don't support 32-bit absolute addressing
        if not self.platform.supports_32bit_abs and '[' in line and '+' in line:
            
            # Extract components: lea rsi, [array + rax]
            try:
                # Parse the instruction
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].lower() == 'lea':
                    dest_reg = parts[1].rstrip(',')
                    
                    # Extract address expression
                    addr_start = line.find('[')
                    addr_end = line.find(']')
                    if addr_start != -1 and addr_end != -1:
                        addr_expr = line[addr_start+1:addr_end]
                        
                        if '+' in addr_expr:
                            parts = addr_expr.split('+')
                            base = parts[0].strip()
                            offset = parts[1].strip() if len(parts) > 1 else '0'
                            
                            # Generate safe alternative using RIP-relative addressing
                            safe_lines = []
                            safe_lines.append(f"    mov {dest_reg}, {base}  ; Load base address")
                            if offset != '0' and offset:
                                safe_lines.append(f"    add {dest_reg}, {offset}  ; Add offset")
                            
                            return '\n'.join(safe_lines)
            except:
                # If parsing fails, return original line
                pass
        
        return line
    
    def emit_condition_check(self, condition: str, jump_label: str, jump_if_true: bool):
        """Emit assembly for condition checking"""
        condition = condition.strip()
        
        # Order operators from longest to shortest to avoid partial matches
        comparison_ops = [
            ('>=', ('jge', 'jl')),
            ('<=', ('jle', 'jg')),
            ('==', ('je', 'jne')),
            ('!=', ('jne', 'je')),
            ('>', ('jg', 'jle')),
            ('<', ('jl', 'jge')),
        ]
        
        for op, (true_jmp, false_jmp) in comparison_ops:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Only x86-64 supported
                    self._emit_x86_condition_check(left, right, true_jmp, false_jmp, jump_label, jump_if_true)
                    return
        
        # For non-comparison conditions, test the value - x86-64 only
        if condition.startswith('[') and condition.endswith(']'):
            self.add_line(f"    mov rax, {condition}")
            self.add_line(f"    test rax, rax")
        else:
            self.add_line(f"    test {condition}, {condition}")
        jump_instr = "jnz" if jump_if_true else "jz"
        
        self.add_line(f"    {jump_instr} {jump_label}")
    

    
    def _emit_x86_condition_check(self, left: str, right: str, true_jmp: str, false_jmp: str, jump_label: str, jump_if_true: bool):
        """Emit x86-64 condition check"""
        # Normalize memory references by removing spaces within brackets
        left = self._normalize_memory_ref(left)
        right = self._normalize_memory_ref(right)
        
        # For x86-64, we need to handle memory-to-memory comparisons
        # and ensure proper sizing for memory operands
        left_is_memory = left.startswith('[') and left.endswith(']')
        right_is_memory = right.startswith('[') and right.endswith(']')
        
        if left_is_memory and right_is_memory:
            # Both are memory - load left into register
            self.add_line(f"    mov rax, {left}")
            self.add_line(f"    cmp rax, {right}")
        elif left.isdigit() and right.isdigit():
            # Both are immediate values
            self.add_line(f"    mov rax, {left}")
            self.add_line(f"    cmp rax, {right}")
        elif left_is_memory and right.isdigit():
            # Memory compared to immediate - load memory into register
            self.add_line(f"    mov rax, {left}")
            self.add_line(f"    cmp rax, {right}")
        elif right_is_memory and left.isdigit():
            # Immediate compared to memory - load memory into register
            self.add_line(f"    mov rax, {right}")
            self.add_line(f"    cmp rax, {left}")
            # Flip the condition since we swapped operands
            true_jmp, false_jmp = false_jmp, true_jmp
        else:
            # Normal case - at most one is memory, make sure to use proper sizing
            if left_is_memory or right_is_memory:
                # Ensure memory operand uses qword size
                if left_is_memory:
                    self.add_line(f"    mov rax, {left}")
                    self.add_line(f"    cmp rax, {right}")
                else:
                    self.add_line(f"    mov rax, {right}")
                    self.add_line(f"    cmp {left}, rax")
            else:
                self.add_line(f"    cmp {left}, {right}")
        
        jump_instr = true_jmp if jump_if_true else false_jmp
        self.add_line(f"    {jump_instr} {jump_label}")
    
    def _normalize_memory_ref(self, operand: str) -> str:
        """Normalize memory references by removing spaces within brackets"""
        operand = operand.strip()
        if operand.startswith('[') and operand.endswith(']'):
            # Remove spaces inside brackets
            inner = operand[1:-1].strip()
            return f'[{inner}]'
        return operand
    
    def _print_helper(self, arg: str, add_newline: bool):
        """Helper for print operations"""
        arg = arg.strip()
        
        if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
            str_content = arg[1:-1]
            str_label = self.ctx.new_label("str")
            
            if add_newline:
                self.data_section.append(f"    {str_label} {self.platform.string_directive} {arg[0]}{str_content}{arg[0]}, 10")
                length = len(str_content) + 1
            else:
                self.data_section.append(f"    {str_label} {self.platform.string_directive} {arg[0]}{str_content}{arg[0]}")
                length = len(str_content)
            
            self.add_line(f"    ; %print{('ln' if add_newline else '')} {arg}")
            
            if self.platform.os_name == 'windows':
                # Windows API call to WriteConsole
                self.add_line(f"    ; Get console handle")
                self.add_line(f"    mov rcx, -11  ; STD_OUTPUT_HANDLE")
                self.add_function_call("GetStdHandle")  # Track the function call
                self.add_line(f"    call GetStdHandle")
                self.add_line(f"    mov r15, rax  ; Save handle")
                self.add_line(f"    ; Call WriteConsole")
                self.add_line(f"    mov rcx, r15  ; console handle")
                self.add_line(f"    mov rdx, {str_label}  ; text buffer")
                self.add_line(f"    mov r8, {length}  ; number of chars")
                self.add_line(f"    mov r9, 0  ; written count (can be NULL)")
                self.add_line(f"    mov qword [rsp+32], 0  ; 5th parameter (reserved)")
                self.add_function_call("WriteConsoleA")  # Track the function call
                self.add_line(f"    call WriteConsoleA")
            else:
                self.add_line(f"    mov {self.platform.registers['syscall_num']}, {self.platform.syscalls['sys_write']}")
                self.add_line(f"    mov {self.platform.registers['arg1']}, 1")
                self.add_line(f"    mov {self.platform.registers['arg2']}, {str_label}")
                self.add_line(f"    mov {self.platform.registers['arg3']}, {length}")
                self.add_line(f"    {self.platform.get_syscall_instruction()}")
        else:
            # Handle variable references
            self.add_line(f"    ; %print{('ln' if add_newline else '')} {arg}")
            if self.platform.os_name == 'windows':
                # Windows API call for variables
                self.add_line(f"    mov rcx, -11  ; STD_OUTPUT_HANDLE")
                self.add_function_call("GetStdHandle")  # Track the function call
                self.add_line(f"    call GetStdHandle")
                self.add_line(f"    mov r15, rax  ; Save handle")
                self.add_line(f"    mov rcx, r15  ; console handle")
                self.add_line(f"    mov rdx, {arg}  ; text buffer")
                if add_newline:
                    self.add_line(f"    mov r8, 20  ; estimated length + newline")
                else:
                    self.add_line(f"    mov r8, 19  ; estimated length")
                self.add_line(f"    mov r9, 0  ; written count")
                self.add_line(f"    mov qword [rsp+32], 0  ; 5th parameter (reserved)")
                self.add_function_call("WriteConsoleA")  # Track the function call
                self.add_line(f"    call WriteConsoleA")
            else:
                self.add_line(f"    mov {self.platform.registers['syscall_num']}, {self.platform.syscalls['sys_write']}")
                self.add_line(f"    mov {self.platform.registers['arg1']}, 1")
                self.add_line(f"    mov {self.platform.registers['arg2']}, {arg}")
                self.add_line(f"    ; {self.platform.registers['arg3']} must contain length")
                self.add_line(f"    {self.platform.get_syscall_instruction()}")
    
    def process_scanf(self, args: List[str]):
        """Process scanf call"""
        if len(args) >= 1:
            dest = args[0] if len(args) > 1 else args[0]
            self.add_line(f"    ; %scanf")
            self.add_line(f"    mov {self.platform.registers['syscall_num']}, {self.platform.syscalls['sys_read']}")
            self.add_line(f"    mov {self.platform.registers['arg1']}, 0")
            self.add_line(f"    mov {self.platform.registers['arg2']}, {dest}")
            self.add_line(f"    mov {self.platform.registers['arg3']}, 256")
            self.add_line(f"    syscall")
    
    def process_strlen(self, args: List[str]):
        """Process strlen call"""
        if len(args) >= 2:
            str_ptr, result = args[0], args[1]
            self.stdlib_used.add('strlen')
            self.add_line(f"    ; %strlen {str_ptr}, {result}")
            self.add_line(f"    mov rdi, {str_ptr}")
            self.add_function_call("__strlen")  # Track the function call
            self.add_line(f"    call __strlen")
            self.add_line(f"    mov {result}, rax")
    
    def process_strcpy(self, args: List[str]):
        """Process strcpy call"""
        if len(args) >= 2:
            dest, src = args[0], args[1]
            self.stdlib_used.add('strcpy')
            self.add_line(f"    ; %strcpy {dest}, {src}")
            self.add_line(f"    mov rdi, {dest}")
            self.add_line(f"    mov rsi, {src}")
            self.add_function_call("__strcpy")  # Track the function call
            self.add_line(f"    call __strcpy")
    
    def process_strcmp(self, args: List[str]):
        """Process strcmp call"""
        if len(args) >= 2:
            str1, str2 = args[0], args[1]
            self.stdlib_used.add('strcmp')
            self.add_line(f"    ; %strcmp {str1}, {str2}")
            self.add_line(f"    mov rdi, {str1}")
            self.add_line(f"    mov rsi, {str2}")
            self.add_function_call("__strcmp")  # Track the function call
            self.add_line(f"    call __strcmp")
    
    def process_memset(self, args: List[str]):
        """Process memset call"""
        if len(args) >= 3:
            dest, value, count = args[0], args[1], args[2]
            self.stdlib_used.add('memset')
            self.add_line(f"    ; %memset {dest}, {value}, {count}")
            self.add_line(f"    mov rdi, {dest}")
            self.add_line(f"    mov rsi, {value}")
            self.add_line(f"    mov rdx, {count}")
            self.add_function_call("__memset")  # Track the function call
            self.add_line(f"    call __memset")
    
    def process_memcpy(self, args: List[str]):
        """Process memcpy call"""
        if len(args) >= 3:
            dest, src, count = args[0], args[1], args[2]
            self.stdlib_used.add('memcpy')
            self.add_line(f"    ; %memcpy {dest}, {src}, {count}")
            self.add_line(f"    mov rdi, {dest}")
            self.add_line(f"    mov rsi, {src}")
            self.add_line(f"    mov rdx, {count}")
            self.add_function_call("__memcpy")  # Track the function call
            self.add_line(f"    call __memcpy")
    
    def process_atoi(self, args: List[str]):
        """Process atoi call"""
        if len(args) >= 2:
            str_ptr, result = args[0], args[1]
            self.stdlib_used.add('atoi')
            self.add_line(f"    ; %atoi {str_ptr}, {result}")
            self.add_line(f"    mov rdi, {str_ptr}")
            self.add_line(f"    call __atoi")
            self.add_line(f"    mov {result}, rax")
    
    def process_itoa(self, args: List[str]):
        """Process itoa call"""
        if len(args) >= 2:
            num, buffer = args[0], args[1]
            self.stdlib_used.add('itoa')
            self.add_line(f"    ; %itoa {num}, {buffer}")
            self.add_line(f"    mov rdi, {num}")
            self.add_line(f"    mov rsi, {buffer}")
            self.add_line(f"    call __itoa")
    
    def generate_stdlib(self) -> List[str]:
        """Generate standard library functions"""
        stdlib = []
        
        if 'strlen' in self.stdlib_used:
            stdlib.extend([
                "; Standard Library: strlen",
                "__strlen:",
                "    push rbp",
                "    mov rbp, rsp",
                "    xor rax, rax",
                ".loop:",
                "    cmp byte [rdi + rax], 0",
                "    je .done",
                "    inc rax",
                "    jmp .loop",
                ".done:",
                "    pop rbp",
                "    ret",
                ""
            ])
        
        if 'strcpy' in self.stdlib_used:
            stdlib.extend([
                "; Standard Library: strcpy",
                "__strcpy:",
                "    push rbp",
                "    mov rbp, rsp",
                "    xor rcx, rcx",
                ".loop:",
                "    mov al, byte [rsi + rcx]",
                "    mov byte [rdi + rcx], al",
                "    test al, al",
                "    jz .done",
                "    inc rcx",
                "    jmp .loop",
                ".done:",
                "    pop rbp",
                "    ret",
                ""
            ])
        
        if 'strcmp' in self.stdlib_used:
            stdlib.extend([
                "; Standard Library: strcmp",
                "__strcmp:",
                "    push rbp",
                "    mov rbp, rsp",
                "    xor rcx, rcx",
                ".loop:",
                "    mov al, byte [rdi + rcx]",
                "    mov bl, byte [rsi + rcx]",
                "    cmp al, bl",
                "    jne .not_equal",
                "    test al, al",
                "    jz .equal",
                "    inc rcx",
                "    jmp .loop",
                ".equal:",
                "    xor rax, rax",
                "    pop rbp",
                "    ret",
                ".not_equal:",
                "    movzx rax, al",
                "    movzx rbx, bl",
                "    sub rax, rbx",
                "    pop rbp",
                "    ret",
                ""
            ])
        
        if 'memset' in self.stdlib_used:
            stdlib.extend([
                "; Standard Library: memset",
                "__memset:",
                "    push rbp",
                "    mov rbp, rsp",
                "    xor rcx, rcx",
                ".loop:",
                "    cmp rcx, rdx",
                "    jge .done",
                "    mov byte [rdi + rcx], sil",
                "    inc rcx",
                "    jmp .loop",
                ".done:",
                "    pop rbp",
                "    ret",
                ""
            ])
        
        if 'memcpy' in self.stdlib_used:
            stdlib.extend([
                "; Standard Library: memcpy",
                "__memcpy:",
                "    push rbp",
                "    mov rbp, rsp",
                "    xor rcx, rcx",
                ".loop:",
                "    cmp rcx, rdx",
                "    jge .done",
                "    mov al, byte [rsi + rcx]",
                "    mov byte [rdi + rcx], al",
                "    inc rcx",
                "    jmp .loop",
                ".done:",
                "    pop rbp",
                "    ret",
                ""
            ])
        
        if 'atoi' in self.stdlib_used:
            stdlib.extend([
                "; Standard Library: atoi",
                "__atoi:",
                "    push rbp",
                "    mov rbp, rsp",
                "    xor rax, rax",
                "    xor rcx, rcx",
                "    xor rdx, rdx",
                "    mov r8, 1  ; sign",
                "    cmp byte [rdi], '-'",
                "    jne .loop",
                "    mov r8, -1",
                "    inc rdi",
                ".loop:",
                "    movzx rdx, byte [rdi + rcx]",
                "    cmp dl, 0",
                "    je .done",
                "    cmp dl, '0'",
                "    jb .done",
                "    cmp dl, '9'",
                "    ja .done",
                "    sub dl, '0'",
                "    imul rax, 10",
                "    add rax, rdx",
                "    inc rcx",
                "    jmp .loop",
                ".done:",
                "    imul rax, r8",
                "    pop rbp",
                "    ret",
                ""
            ])
        
        if 'itoa' in self.stdlib_used:
            stdlib.extend([
                "; Standard Library: itoa",
                "__itoa:",
                "    push rbp",
                "    mov rbp, rsp",
                "    push rbx",
                "    mov rax, rdi",
                "    mov rbx, 10",
                "    xor rcx, rcx",
                "    test rax, rax",
                "    jns .positive",
                "    mov byte [rsi], '-'",
                "    inc rsi",
                "    neg rax",
                ".positive:",
                "    test rax, rax",
                "    jnz .convert",
                "    mov byte [rsi], '0'",
                "    mov byte [rsi + 1], 0",
                "    pop rbx",
                "    pop rbp",
                "    ret",
                ".convert:",
                "    xor rdx, rdx",
                "    div rbx",
                "    add dl, '0'",
                "    push rdx",
                "    inc rcx",
                "    test rax, rax",
                "    jnz .convert",
                ".write:",
                "    pop rdx",
                "    mov byte [rsi], dl",
                "    inc rsi",
                "    loop .write",
                "    mov byte [rsi], 0",
                "    pop rbx",
                "    pop rbp",
                "    ret",
                ""
            ])
        
        return stdlib
    
    def _generate_safe_array_access(self, base: str, offset: str, dest_reg: str) -> List[str]:
        """Generate safe array access code that avoids 32-bit absolute addressing"""
        lines = []
        
        # Load base address into a register
        lines.append(f"    mov {dest_reg}, {base}  ; Load base address")
        
        # Add offset if it's not zero
        if offset.strip() != '0' and offset.strip():
            if offset.strip().isdigit():
                lines.append(f"    add {dest_reg}, {offset}  ; Add offset")
            else:
                lines.append(f"    add {dest_reg}, {offset}  ; Add offset")
        
        return lines
    
    def _generate_default_exit(self) -> List[str]:
        """Generate default exit code for the platform"""
        exit_code = []
        exit_code.append("    ; Default program exit")
        if self.platform.os_name == 'windows':
            exit_code.append(f"    mov rcx, 0  ; exit code")
            exit_code.append(f"    call ExitProcess")
        else:
            exit_code.append(f"    mov {self.platform.registers['syscall_num']}, {self.platform.syscalls['sys_exit']}")
            exit_code.append(f"    mov {self.platform.registers['arg1']}, 0")
            exit_code.append(f"    {self.platform.get_syscall_instruction()}")
        return exit_code
    

    
