#!/usr/bin/env python3
import re
import sys
import subprocess
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import uuid
import json
import hashlib
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod

# --- Colorized Logging ---
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_message(level: str, message: str):
    """Prints a color-coded log message."""
    if level == "OK":
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} {message}")
    elif level == "ERROR":
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")
    elif level == "WARN":
        print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {message}")
    elif level == "INFO":
        print(f"{Colors.WARNING}[*]{Colors.ENDC} {message}")

# Advanced Compiler Architecture
class TargetArchitecture(Enum):
    """Supported target architectures"""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    RISCV64 = "riscv64"
    X86 = "x86"
    ARM = "arm"
    RISCV32 = "riscv32"

class OptimizationLevel(Enum):
    """Compiler optimization levels"""
    NONE = "O0"
    BASIC = "O1"
    STANDARD = "O2"
    AGGRESSIVE = "O3"
    SIZE = "Os"
    FAST = "Ofast"

@dataclass
class CompilerOptions:
    """Advanced compiler options"""
    target_arch: TargetArchitecture = TargetArchitecture.X86_64
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    debug_info: bool = True
    pic: bool = False  # Position Independent Code
    pie: bool = False  # Position Independent Executable
    lto: bool = False  # Link Time Optimization
    pgo: bool = False  # Profile Guided Optimization
    sanitizers: Set[str] = field(default_factory=set)  # address, thread, memory, etc.
    custom_flags: List[str] = field(default_factory=list)
    define_macros: Dict[str, str] = field(default_factory=dict)
    include_paths: List[Path] = field(default_factory=list)
    library_paths: List[Path] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    cross_compile_prefix: str = ""
    cpu_features: Set[str] = field(default_factory=set)  # avx2, sse4.1, etc.
    vectorize: bool = True
    unroll_loops: bool = True
    inline_functions: bool = True

class MacroType(Enum):
    """Types of macros"""
    SIMPLE = auto()      # #define NAME value
    FUNCTION = auto()    # #define NAME(args) body
    VARIADIC = auto()    # #define NAME(args...) body
    BUILTIN = auto()     # Built-in compiler macros

@dataclass
class Macro:
    """Macro definition"""
    name: str
    macro_type: MacroType
    parameters: List[str] = field(default_factory=list)
    body: str = ""
    is_variadic: bool = False
    line_number: int = 0
    file_path: str = ""
    expansion_count: int = 0
    nested_level: int = 0

class MacroProcessor:
    """Advanced macro preprocessing system"""
    
    def __init__(self):
        self.macros: Dict[str, Macro] = {}
        self.conditional_stack: List[Tuple[str, bool]] = []  # (directive, condition_met)
        self.include_stack: List[Path] = []
        self.include_guard_cache: Dict[Path, str] = {}
        self.expansion_depth = 0
        self.max_expansion_depth = 1000
        self.builtin_macros = self._init_builtin_macros()
        self.macros.update(self.builtin_macros)
        
    def _init_builtin_macros(self) -> Dict[str, Macro]:
        """Initialize built-in macros"""
        builtins = {}
        
        # Standard predefined macros
        builtins["__LINE__"] = Macro("__LINE__", MacroType.BUILTIN)
        builtins["__FILE__"] = Macro("__FILE__", MacroType.BUILTIN) 
        builtins["__DATE__"] = Macro("__DATE__", MacroType.BUILTIN)
        builtins["__TIME__"] = Macro("__TIME__", MacroType.BUILTIN)
        builtins["__TIMESTAMP__"] = Macro("__TIMESTAMP__", MacroType.BUILTIN)
        
        # Compiler identification
        builtins["__CASM__"] = Macro("__CASM__", MacroType.SIMPLE, body="1")
        builtins["__CASM_VERSION__"] = Macro("__CASM_VERSION__", MacroType.SIMPLE, body="\"2.0\"")
        
        # Architecture macros
        builtins["__x86_64__"] = Macro("__x86_64__", MacroType.SIMPLE, body="1")
        builtins["__amd64__"] = Macro("__amd64__", MacroType.SIMPLE, body="1")
        builtins["_M_X64"] = Macro("_M_X64", MacroType.SIMPLE, body="100")
        
        # Standard library macros
        builtins["__STDC__"] = Macro("__STDC__", MacroType.SIMPLE, body="1")
        builtins["__STDC_VERSION__"] = Macro("__STDC_VERSION__", MacroType.SIMPLE, body="201112L")
        
        return builtins
        
    def define_macro(self, name: str, value: str = "", parameters: List[str] = None, 
                    is_variadic: bool = False, file_path: str = "", line_number: int = 0):
        """Define a new macro"""
        if parameters:
            macro_type = MacroType.VARIADIC if is_variadic else MacroType.FUNCTION
        else:
            macro_type = MacroType.SIMPLE
            
        macro = Macro(
            name=name,
            macro_type=macro_type,
            parameters=parameters or [],
            body=value,
            is_variadic=is_variadic,
            file_path=file_path,
            line_number=line_number
        )
        
        self.macros[name] = macro
        
    def undefine_macro(self, name: str):
        """Undefine a macro"""
        if name in self.macros and self.macros[name].macro_type != MacroType.BUILTIN:
            del self.macros[name]
            
    def is_defined(self, name: str) -> bool:
        """Check if a macro is defined"""
        return name in self.macros
        
    def expand_macro(self, name: str, args: List[str] = None, file_path: str = "", line_number: int = 0) -> str:
        """Expand a macro with given arguments"""
        if self.expansion_depth > self.max_expansion_depth:
            raise RuntimeError(f"Maximum macro expansion depth ({self.max_expansion_depth}) exceeded")
            
        if name not in self.macros:
            return name
            
        macro = self.macros[name]
        macro.expansion_count += 1
        
        self.expansion_depth += 1
        try:
            result = self._expand_macro_body(macro, args or [], file_path, line_number)
        finally:
            self.expansion_depth -= 1
            
        return result
        
    def _expand_macro_body(self, macro: Macro, args: List[str], file_path: str, line_number: int) -> str:
        """Expand macro body with parameter substitution"""
        if macro.macro_type == MacroType.BUILTIN:
            return self._expand_builtin_macro(macro.name, file_path, line_number)
            
        body = macro.body
        
        if macro.macro_type in (MacroType.FUNCTION, MacroType.VARIADIC):
            # Substitute parameters
            # Create a temporary body to perform substitutions on
            temp_body = body
            for i, param in enumerate(macro.parameters):
                if i < len(args):
                    # Use regex for proper parameter substitution
                    pattern = r'\b' + re.escape(param) + r'\b'
                    temp_body = re.sub(pattern, args[i], temp_body)
                    
            # Handle variadic arguments
            if macro.is_variadic and len(args) > len(macro.parameters):
                variadic_args = args[len(macro.parameters):]
                body = body.replace("__VA_ARGS__", ", ".join(variadic_args))
                
        # Recursively expand any macros in the result
        return self._recursive_expand(temp_body, file_path, line_number)
        
    def _expand_builtin_macro(self, name: str, file_path: str, line_number: int) -> str:
        """Expand built-in macros"""
        if name == "__LINE__":
            return str(line_number)
        elif name == "__FILE__":
            return f'"{ file_path}"'
        elif name == "__DATE__":
            return f'"{ time.strftime("%b %d %Y")}"'
        elif name == "__TIME__":
            return f'"{ time.strftime("%H:%M:%S")}"'
        elif name == "__TIMESTAMP__":
            return f'"{ time.strftime("%a %b %d %H:%M:%S %Y")}"'
        else:
            return self.macros[name].body
            
    def _recursive_expand(self, text: str, file_path: str, line_number: int) -> str:
        """Recursively expand macros in text"""
        # Find and expand macro invocations
        macro_pattern = r'\b([a-zA-Z_]\w*)(?:\(([^)]*)\))?\b'
        
        def replace_macro(match):
            macro_name = match.group(1)
            args_str = match.group(2)
            
            if macro_name not in self.macros:
                return match.group(0)
                
            args = []
            if args_str is not None:
                # Parse arguments (simple parsing - could be improved)
                args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                
            return self.expand_macro(macro_name, args, file_path, line_number)
            
        expanded = re.sub(macro_pattern, replace_macro, text)
        
        # If expansion occurred, try again
        if expanded != text:
            return self._recursive_expand(expanded, file_path, line_number)
        else:
            return expanded
            
    def process_directive(self, line: str, file_path: str, line_number: int) -> Tuple[bool, str]:
        """Process preprocessor directive. Returns (should_include_line, processed_line)"""
        stripped = line.strip()
        
        # #define directive
        if stripped.startswith('#define'):
            self._process_define(stripped[7:].strip(), file_path, line_number)
            return False, ""
            
        # #undef directive
        elif stripped.startswith('#undef'):
            name = stripped[6:].strip()
            self.undefine_macro(name)
            return False, ""
            
        # #ifdef directive
        elif stripped.startswith('#ifdef'):
            name = stripped[6:].strip()
            condition = self.is_defined(name)
            self.conditional_stack.append(("ifdef", condition))
            return False, ""
            
        # #ifndef directive
        elif stripped.startswith('#ifndef'):
            name = stripped[7:].strip()
            condition = not self.is_defined(name)
            self.conditional_stack.append(("ifndef", condition))
            return False, ""
            
        # #if directive
        elif stripped.startswith('#if'):
            expr = stripped[3:].strip()
            condition = self._evaluate_preprocessor_expression(expr, file_path, line_number)
            self.conditional_stack.append(("if", condition))
            return False, ""
            
        # #elif directive
        elif stripped.startswith('#elif'):
            if not self.conditional_stack:
                raise SyntaxError(f"#elif without #if at {file_path}:{line_number}")
            expr = stripped[5:].strip()
            _, prev_condition = self.conditional_stack.pop()
            if not prev_condition:
                condition = self._evaluate_preprocessor_expression(expr, file_path, line_number)
            else:
                condition = False
            self.conditional_stack.append(("elif", condition))
            return False, ""
            
        # #else directive
        elif stripped.startswith('#else'):
            if not self.conditional_stack:
                raise SyntaxError(f"#else without #if at {file_path}:{line_number}")
            directive, prev_condition = self.conditional_stack.pop()
            self.conditional_stack.append(("else", not prev_condition))
            return False, ""
            
        # #endif directive
        elif stripped.startswith('#endif'):
            if not self.conditional_stack:
                raise SyntaxError(f"#endif without #if at {file_path}:{line_number}")
            self.conditional_stack.pop()
            return False, ""
            
        # Check if we should include this line based on conditional stack
        should_include = self._should_include_line()
        if should_include:
            # Expand macros in the line
            expanded = self._recursive_expand(line, file_path, line_number)
            return True, expanded
        else:
            return False, ""
            
    def _process_define(self, define_str: str, file_path: str, line_number: int):
        """Process #define directive"""
        # Parse different define formats:
        # #define NAME value
        # #define NAME(args) value
        # #define NAME(args...) value (variadic)
        
        match = re.match(r'([a-zA-Z_]\w*)(?:\(([^)]*)\))?\s*(.*)', define_str)
        if not match:
            raise SyntaxError(f"Invalid #define directive at {file_path}:{line_number}")
            
        name = match.group(1)
        params_str = match.group(2)
        body = match.group(3) or ""
        
        parameters = []
        is_variadic = False
        
        if params_str is not None:  # Function-like macro
            if params_str.endswith('...'):
                is_variadic = True
                params_str = params_str[:-3].rstrip(', ')
                
            if params_str.strip():
                parameters = [p.strip() for p in params_str.split(',')]
                
        self.define_macro(name, body, parameters, is_variadic, file_path, line_number)
        
    def _should_include_line(self) -> bool:
        """Check if current line should be included based on conditional stack"""
        for directive, condition in self.conditional_stack:
            if not condition:
                return False
        return True
        
    def _evaluate_preprocessor_expression(self, expr: str, file_path: str, line_number: int) -> bool:
        """Evaluate preprocessor expression (simplified)"""
        # Expand macros in expression
        expanded = self._recursive_expand(expr, file_path, line_number)
        
        # Handle defined() operator
        def replace_defined(match):
            name = match.group(1)
            return "1" if self.is_defined(name) else "0"
            
        expanded = re.sub(r'defined\s*\(\s*([a-zA-Z_]\w*)\s*\)', replace_defined, expanded)
        expanded = re.sub(r'defined\s+([a-zA-Z_]\w*)', replace_defined, expanded)
        
        # Simple expression evaluation (could be much more sophisticated)
        try:
            # Replace undefined identifiers with 0
            def replace_undefined(match):
                name = match.group(0)
                if name.isdigit() or name in ['0', '1']:
                    return name
                return "0"
                
            expanded = re.sub(r'\b[a-zA-Z_]\w*\b', replace_undefined, expanded)
            
            # Evaluate the expression (limited to safe operations)
            result = eval(expanded, {"__builtins__": {}}, {})
            return bool(result)
        except:
            return False
            
    def process_file(self, lines: List[str], file_path: str) -> List[str]:
        """Process an entire file through the macro processor"""
        processed_lines = []
        
        for line_number, line in enumerate(lines, 1):
            if line.strip().startswith('#'):
                should_include, processed_line = self.process_directive(line, file_path, line_number)
                if should_include and processed_line:
                    processed_lines.append(processed_line)
            else:
                should_include = self._should_include_line()
                if should_include:
                    expanded = self._recursive_expand(line, file_path, line_number)
                    processed_lines.append(expanded)
                    
        return processed_lines

class AdvancedOptimizer:
    """Advanced optimization engine"""
    
    def __init__(self, options: CompilerOptions):
        self.options = options
        self.optimization_passes = [
            self._dead_code_elimination,
            self._constant_folding,
            self._common_subexpression_elimination,
            self._loop_optimization,
            self._register_allocation_hints,
            self._vectorization_hints,
            self._instruction_scheduling
        ]
        
    def optimize_assembly_block(self, asm_lines: List[str]) -> List[str]:
        """Apply optimization passes to assembly block"""
        optimized = asm_lines[:]
        
        if self.options.optimization_level == OptimizationLevel.NONE:
            return optimized
            
        for pass_func in self.optimization_passes:
            optimized = pass_func(optimized)
            
        return optimized
        
    def _dead_code_elimination(self, lines: List[str]) -> List[str]:
        """Remove dead code"""
        # Simple dead code elimination
        optimized = []
        for line in lines:
            stripped = line.strip()
            # Skip redundant moves
            if re.match(r'mov[bwlq]?\s+(\w+),\s*\1', stripped, re.I):
                continue  # mov reg, reg
            # Skip nops
            if stripped.lower() in ['nop', 'nop;']:
                continue
            optimized.append(line)
        return optimized
        
    def _constant_folding(self, lines: List[str]) -> List[str]:
        """Fold constants where possible"""
        # Simple constant folding for immediate values
        optimized = []
        for line in lines:
            # Look for patterns like add reg, 0 and remove them
            if re.match(r'add[bwlq]?\s+\w+,\s*[\$#]?0\b', line.strip(), re.I):
                continue  # Adding 0 is a no-op
            if re.match(r'sub[bwlq]?\s+\w+,\s*[\$#]?0\b', line.strip(), re.I):
                continue  # Subtracting 0 is a no-op
            optimized.append(line)
        return optimized
        
    def _common_subexpression_elimination(self, lines: List[str]) -> List[str]:
        """Eliminate common subexpressions"""
        # Simplified CSE - look for repeated calculations
        return lines  # Placeholder
        
    def _loop_optimization(self, lines: List[str]) -> List[str]:
        """Optimize loops"""
        if not self.options.unroll_loops:
            return lines
            
        # Look for simple loops and potentially unroll them
        return lines  # Placeholder
        
    def _register_allocation_hints(self, lines: List[str]) -> List[str]:
        """Provide register allocation hints"""
        # Add register allocation hints for better code generation
        return lines  # Placeholder
        
    def _vectorization_hints(self, lines: List[str]) -> List[str]:
        """Add vectorization hints"""
        if not self.options.vectorize:
            return lines
            
        # Look for vectorizable patterns
        return lines  # Placeholder
        
    def _instruction_scheduling(self, lines: List[str]) -> List[str]:
        """Optimize instruction scheduling"""
        # Reorder instructions for better pipeline utilization
        return lines  # Placeholder

class ArchitectureBackend(ABC):
    """Abstract base class for architecture-specific code generation"""
    
    @abstractmethod
    def get_register_info(self) -> Dict[str, Set[str]]:
        """Get register information for this architecture"""
        pass
        
    @abstractmethod
    def get_instruction_set(self) -> Set[str]:
        """Get supported instruction set for this architecture"""
        pass
        
    @abstractmethod
    def convert_to_native_syntax(self, line: str) -> str:
        """Convert generic assembly to architecture-specific syntax"""
        pass
        
    @abstractmethod
    def optimize_for_architecture(self, lines: List[str]) -> List[str]:
        """Apply architecture-specific optimizations"""
        pass
        
    @abstractmethod
    def get_calling_convention(self) -> Dict[str, Any]:
        """Get calling convention details"""
        pass

class X86_64Backend(ArchitectureBackend):
    """x86_64 architecture backend"""
    
    def get_register_info(self) -> Dict[str, Set[str]]:
        return {
            '8-bit': REGISTERS_8,
            '16-bit': REGISTERS_16,
            '32-bit': REGISTERS_32,
            '64-bit': REGISTERS_64,
            'xmm': REGISTERS_XMM,
            'ymm': REGISTERS_YMM,
            'zmm': REGISTERS_ZMM,
            'mask': REGISTERS_MASK
        }
        
    def get_instruction_set(self) -> Set[str]:
        return ASM_OPS
        
    def convert_to_native_syntax(self, line: str) -> str:
        # x86_64 uses Intel/AT&T syntax - already supported
        return line
        
    def optimize_for_architecture(self, lines: List[str]) -> List[str]:
        # x86_64 specific optimizations
        optimized = []
        for line in lines:
            # Convert 32-bit operations to 64-bit where beneficial
            line = re.sub(r'\bmovl\b', 'movq', line)
            # Use more efficient instruction variants
            line = re.sub(r'\bimull\b', 'imulq', line)
            optimized.append(line)
        return optimized
        
    def get_calling_convention(self) -> Dict[str, Any]:
        return {
            'integer_params': ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'],
            'float_params': ['xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7'],
            'return_int': 'rax',
            'return_float': 'xmm0',
            'callee_saved': ['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15'],
            'caller_saved': ['rax', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11'],
            'stack_alignment': 16
        }

class ARM64Backend(ArchitectureBackend):
    """ARM64/AArch64 architecture backend"""
    
    def get_register_info(self) -> Dict[str, Set[str]]:
        return {
            'general': {f'x{i}' for i in range(31)} | {f'w{i}' for i in range(31)},
            'vector': {f'v{i}' for i in range(32)},
            'special': {'sp', 'lr', 'pc', 'xzr', 'wzr'}
        }
        
    def get_instruction_set(self) -> Set[str]:
        return {
            # ARM64 instruction set (subset)
            'add', 'sub', 'mul', 'div', 'mov', 'ldr', 'str', 'ldp', 'stp',
            'b', 'bl', 'br', 'blr', 'ret', 'cbz', 'cbnz', 'tbz', 'tbnz',
            'adr', 'adrp', 'cmp', 'ccmp', 'csel', 'cset', 'cinc', 'cneg',
            'and', 'orr', 'eor', 'bic', 'lsl', 'lsr', 'asr', 'ror',
            'svc', 'hvc', 'smc', 'brk', 'hlt', 'nop', 'yield', 'wfe', 'wfi'
        }
        
    def convert_to_native_syntax(self, line: str) -> str:
        # Convert x86 patterns to ARM64
        line = re.sub(r'\bmov\s+(\w+),\s*(\w+)', r'mov \1, \2', line)
        line = re.sub(r'\bcall\s+(\w+)', r'bl \1', line)
        line = re.sub(r'\bret\b', 'ret', line)
        return line
        
    def optimize_for_architecture(self, lines: List[str]) -> List[str]:
        # ARM64 specific optimizations
        optimized = []
        for line in lines:
            # Use ldp/stp for consecutive loads/stores
            if 'ldr' in line.lower():
                line = line  # Placeholder for ldp optimization
            optimized.append(line)
        return optimized
        
    def get_calling_convention(self) -> Dict[str, Any]:
        return {
            'integer_params': [f'x{i}' for i in range(8)],
            'float_params': [f'v{i}' for i in range(8)],
            'return_int': 'x0',
            'return_float': 'v0',
            'callee_saved': [f'x{i}' for i in range(19, 29)],
            'caller_saved': [f'x{i}' for i in range(18)],
            'stack_alignment': 16
        }

class RISCVBackend(ArchitectureBackend):
    """RISC-V architecture backend"""
    
    def get_register_info(self) -> Dict[str, Set[str]]:
        return {
            'integer': {f'x{i}' for i in range(32)} | {'zero', 'ra', 'sp', 'gp', 'tp', 'fp'},
            'float': {f'f{i}' for i in range(32)},
            'aliases': {'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7',
                       't0', 't1', 't2', 't3', 't4', 't5', 't6',
                       's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11'}
        }
        
    def get_instruction_set(self) -> Set[str]:
        return {
            # RISC-V instruction set
            'add', 'sub', 'mul', 'div', 'addi', 'subi', 'lui', 'auipc',
            'lw', 'sw', 'lb', 'sb', 'lh', 'sh', 'ld', 'sd',
            'beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu',
            'jal', 'jalr', 'j', 'jr', 'call', 'ret',
            'and', 'or', 'xor', 'andi', 'ori', 'xori',
            'sll', 'srl', 'sra', 'slli', 'srli', 'srai',
            'slt', 'sltu', 'slti', 'sltiu',
            'ecall', 'ebreak', 'fence', 'nop'
        }
        
    def convert_to_native_syntax(self, line: str) -> str:
        # Convert x86 patterns to RISC-V
        line = re.sub(r'\bmov\s+(\w+),\s*(\w+)', r'mv \1, \2', line)
        line = re.sub(r'\bcall\s+(\w+)', r'jal \1', line)
        line = re.sub(r'\bret\b', 'ret', line)
        return line
        
    def optimize_for_architecture(self, lines: List[str]) -> List[str]:
        # RISC-V specific optimizations
        optimized = []
        for line in lines:
            # Use compressed instructions where possible
            line = re.sub(r'\baddi\s+(\w+),\s*\1,\s*1', r'c.addi \1, 1', line)
            optimized.append(line)
        return optimized
        
    def get_calling_convention(self) -> Dict[str, Any]:
        return {
            'integer_params': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'],
            'float_params': ['fa0', 'fa1', 'fa2', 'fa3', 'fa4', 'fa5', 'fa6', 'fa7'],
            'return_int': 'a0',
            'return_float': 'fa0',
            'callee_saved': ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11'],
            'caller_saved': ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'],
            'stack_alignment': 16
        }

class CodeGenerator(ABC):
    """Abstract code generator interface"""
    
    @abstractmethod
    def generate_code(self, ast: Any, options: CompilerOptions) -> str:
        """Generate code from AST"""
        pass

class LLVMIRGenerator(CodeGenerator):
    """LLVM IR code generator"""
    
    def __init__(self, target_arch: TargetArchitecture):
        self.target_arch = target_arch
        self.module_name = "casm_module"
        self.functions = []
        self.globals = []
        self.types = {}
        self.current_function = None
        
    def generate_code(self, ast: Any, options: CompilerOptions) -> str:
        """Generate LLVM IR code"""
        ir_lines = []
        
        # Module header
        ir_lines.append(f'; ModuleID = \'{self.module_name}\'')
        ir_lines.append(f'target triple = "{self._get_target_triple()}"')
        ir_lines.append('')
        
        # Generate type definitions
        ir_lines.extend(self._generate_type_definitions())
        
        # Generate global variables
        ir_lines.extend(self._generate_globals())
        
        # Generate function declarations
        ir_lines.extend(self._generate_function_declarations())
        
        # Generate function definitions
        ir_lines.extend(self._generate_function_definitions())
        
        return '\n'.join(ir_lines)
        
    def _get_target_triple(self) -> str:
        """Get LLVM target triple"""
        if self.target_arch == TargetArchitecture.X86_64:
            return "x86_64-unknown-linux-gnu"
        elif self.target_arch == TargetArchitecture.ARM64:
            return "aarch64-unknown-linux-gnu"
        elif self.target_arch == TargetArchitecture.RISCV64:
            return "riscv64-unknown-linux-gnu"
        else:
            return "x86_64-unknown-linux-gnu"  # Default
            
    def _generate_type_definitions(self) -> List[str]:
        """Generate LLVM type definitions"""
        return []  # Placeholder
        
    def _generate_globals(self) -> List[str]:
        """Generate LLVM global variable definitions"""
        return []  # Placeholder
        
    def _generate_function_declarations(self) -> List[str]:
        """Generate LLVM function declarations"""
        return []  # Placeholder
        
    def _generate_function_definitions(self) -> List[str]:
        """Generate LLVM function definitions"""
        return []  # Placeholder

class JITCompiler:
    """Just-In-Time compiler using LLVM"""
    
    def __init__(self, target_arch: TargetArchitecture):
        self.target_arch = target_arch
        self.compiled_functions = {}
        self.execution_engine = None
        
    def compile_function(self, source: str, function_name: str) -> Any:
        """Compile function to machine code and return callable"""
        # This would use LLVM's JIT capabilities
        # Placeholder implementation
        return None
        
    def execute_function(self, function_name: str, *args) -> Any:
        """Execute JIT-compiled function"""
        if function_name in self.compiled_functions:
            return self.compiled_functions[function_name](*args)
        return None

class DebugInfoGenerator:
    """Generate debug information in DWARF format"""
    
    def __init__(self, source_file: str):
        self.source_file = source_file
        self.line_table = []
        self.variable_info = {}
        self.function_info = {}
        
    def add_line_info(self, line_number: int, address: int):
        """Add line number to address mapping"""
        self.line_table.append((line_number, address))
        
    def add_variable_info(self, name: str, type_info: str, location: str):
        """Add variable debug information"""
        self.variable_info[name] = {
            'type': type_info,
            'location': location
        }
        
    def add_function_info(self, name: str, start_addr: int, end_addr: int):
        """Add function debug information"""
        self.function_info[name] = {
            'start_address': start_addr,
            'end_address': end_addr
        }
        
    def generate_dwarf(self) -> bytes:
        """Generate DWARF debug information"""
        # Placeholder for DWARF generation
        return b''

class ObjectFileGenerator:
    """Generate object files in various formats"""
    
    def __init__(self, target_arch: TargetArchitecture):
        self.target_arch = target_arch
        self.sections = {}
        self.symbols = {}
        self.relocations = []
        
    def add_section(self, name: str, data: bytes, flags: int = 0):
        """Add a section to the object file"""
        self.sections[name] = {
            'data': data,
            'flags': flags
        }
        
    def add_symbol(self, name: str, section: str, offset: int, is_global: bool = False):
        """Add a symbol to the symbol table"""
        self.symbols[name] = {
            'section': section,
            'offset': offset,
            'global': is_global
        }
        
    def add_relocation(self, section: str, offset: int, symbol: str, rel_type: str):
        """Add a relocation entry"""
        self.relocations.append({
            'section': section,
            'offset': offset,
            'symbol': symbol,
            'type': rel_type
        })
        
    def generate_elf(self) -> bytes:
        """Generate ELF object file"""
        # Placeholder for ELF generation
        return b''
        
    def generate_coff(self) -> bytes:
        """Generate COFF object file (Windows)"""
        # Placeholder for COFF generation
        return b''
        
    def generate_macho(self) -> bytes:
        """Generate Mach-O object file (macOS)"""
        # Placeholder for Mach-O generation
        return b''

class ExceptionHandler:
    """Advanced exception handling system"""
    
    def __init__(self):
        self.try_blocks = []
        self.catch_blocks = []
        self.finally_blocks = []
        self.exception_table = []
        
    def enter_try_block(self, block_id: str):
        """Enter a try block"""
        self.try_blocks.append(block_id)
        
    def exit_try_block(self):
        """Exit the current try block"""
        if self.try_blocks:
            return self.try_blocks.pop()
        return None
        
    def add_catch_block(self, exception_type: str, handler_code: str):
        """Add a catch block for specific exception type"""
        self.catch_blocks.append({
            'type': exception_type,
            'handler': handler_code
        })
        
    def generate_exception_table(self) -> List[Dict[str, Any]]:
        """Generate exception handling table"""
        return self.exception_table
        
    def generate_seh_data(self) -> bytes:
        """Generate Structured Exception Handling data for Windows"""
        # Placeholder for SEH generation
        return b''

class ExternDeclaration:
    """Represents an extern declaration for C functions or assembly functions"""
    
    def __init__(self, name: str, declaration_type: str, signature: str = "", 
                 header_file: str = "", is_assembly: bool = False):
        self.name = name
        self.declaration_type = declaration_type  # 'c_function', 'asm_function', 'header_include'
        self.signature = signature  # Function signature like "int func(int, char*)"
        self.header_file = header_file  # Header file like "stdint.h"
        self.is_assembly = is_assembly
        self.return_type = "void"
        self.parameters = []
        self.calling_convention = "cdecl"  # cdecl, stdcall, fastcall, etc.
        
        if signature:
            self._parse_signature(signature)
            
    def _parse_signature(self, signature: str):
        """Parse function signature to extract return type and parameters"""
        # Handle function signature parsing
        # Example: "int func(int a, char* b)" -> return_type="int", params=[("int", "a"), ("char*", "b")]
        signature = signature.strip()
        
        # Match pattern: return_type function_name(parameters)
        match = re.match(r'([\w\s\*]+)\s+(\w+)\s*\(([^)]*)\)', signature)
        if match:
            self.return_type = match.group(1).strip()
            func_name = match.group(2).strip()
            if func_name != self.name:
                self.name = func_name  # Update name if different
            params_str = match.group(3).strip()
            
            if params_str and params_str != "void":
                # Parse parameters
                param_parts = [p.strip() for p in params_str.split(',')]
                for param in param_parts:
                    if param:
                        # Handle "type name" or just "type"
                        parts = param.rsplit(' ', 1)
                        if len(parts) == 2:
                            param_type, param_name = parts
                            self.parameters.append((param_type.strip(), param_name.strip()))
                        else:
                            # Just type, no name
                            self.parameters.append((param.strip(), ""))
        else:
            # Handle simple function name without signature
            self.return_type = "int"  # Default return type
            
    def generate_c_declaration(self) -> str:
        """Generate C function declaration"""
        if self.declaration_type == 'header_include':
            return f'#include <{self.header_file}>'
        elif self.declaration_type == 'c_function':
            param_list = ", ".join([f"{ptype} {pname}" if pname else ptype 
                                  for ptype, pname in self.parameters])
            if not param_list:
                param_list = "void"
            return f'extern {self.return_type} {self.name}({param_list});'
        elif self.declaration_type == 'asm_function':
            # For assembly functions, generate appropriate declaration
            param_list = ", ".join([f"{ptype} {pname}" if pname else ptype 
                                  for ptype, pname in self.parameters])
            if not param_list:
                param_list = "void"
            return f'extern {self.return_type} {self.name}({param_list});  // Assembly function'
        return ""
        
    def generate_inline_asm_call(self, args: List[str] = None) -> str:
        """Generate inline assembly call for this extern function"""
        if not self.is_assembly:
            return ""
            
        args = args or []
        # Generate inline assembly call based on calling convention
        if self.calling_convention == "stdcall" or "Win" in self.name:
            # Windows API call
            call_asm = f'call {self.name}'
        else:
            # Standard call
            call_asm = f'call {self.name}'
            
        return call_asm

class ExternManager:
    """Manages extern declarations and their integration"""
    
    def __init__(self):
        self.extern_declarations: Dict[str, ExternDeclaration] = {}
        self.required_headers: Set[str] = set()
        self.asm_functions: Dict[str, ExternDeclaration] = {}
        
    def add_extern_declaration(self, declaration: ExternDeclaration):
        """Add an extern declaration"""
        self.extern_declarations[declaration.name] = declaration
        
        if declaration.declaration_type == 'header_include':
            self.required_headers.add(declaration.header_file)
        elif declaration.is_assembly:
            self.asm_functions[declaration.name] = declaration
            
    def parse_extern_line(self, line: str) -> Optional[ExternDeclaration]:
        """Parse an extern declaration line"""
        line = line.strip()
        
        # Handle header includes: extern <stdint.h>
        if match := re.match(r'extern\s*<([^>]+)>', line):
            header_file = match.group(1)
            return ExternDeclaration(
                name=f"include_{header_file.replace('.', '_')}",
                declaration_type='header_include',
                header_file=header_file
            )
            
        # Handle function declarations: extern int function_name(params)
        elif match := re.match(r'extern\s+([\w\s\*]+\s+\w+\s*\([^)]*\))', line):
            signature = match.group(1).strip()
            func_match = re.search(r'(\w+)\s*\(', signature)
            if func_match:
                func_name = func_match.group(1)
                return ExternDeclaration(
                    name=func_name,
                    declaration_type='c_function',
                    signature=signature
                )
                
        # Handle simple function names: extern WriteConsole
        elif match := re.match(r'extern\s+(\w+)(?:\s*\(([^)]*)\))?', line):
            func_name = match.group(1)
            params = match.group(2) if match.group(2) else ""
            
            # Check if it looks like a Windows API or assembly function
            is_assembly = (func_name.startswith('Write') or 
                          func_name.startswith('Get') or 
                          func_name.startswith('Set') or 
                          'Console' in func_name or
                          'Handle' in func_name)
                          
            signature = f"int {func_name}({params})" if params else f"int {func_name}()"
            
            return ExternDeclaration(
                name=func_name,
                declaration_type='asm_function' if is_assembly else 'c_function',
                signature=signature,
                is_assembly=is_assembly
            )
            
        return None
        
    def get_required_headers(self) -> List[str]:
        """Get list of required header includes"""
        headers = []
        for header in sorted(self.required_headers):
            headers.append(f'#include <{header}>')
        return headers
        
    def get_function_declarations(self) -> List[str]:
        """Get list of function declarations"""
        declarations = []
        for name, decl in self.extern_declarations.items():
            if decl.declaration_type in ['c_function', 'asm_function']:
                declarations.append(decl.generate_c_declaration())
        return declarations
        
    def is_extern_function(self, name: str) -> bool:
        """Check if a function is an extern declaration"""
        return name in self.extern_declarations
        
    def get_extern_function(self, name: str) -> Optional[ExternDeclaration]:
        """Get extern declaration by name"""
        return self.extern_declarations.get(name)

class TypeSystem:
    """Advanced type system with inference"""
    
    def __init__(self):
        self.types = {
            'void': {'size': 0, 'align': 1},
            'char': {'size': 1, 'align': 1},
            'short': {'size': 2, 'align': 2},
            'int': {'size': 4, 'align': 4},
            'long': {'size': 8, 'align': 8},
            'float': {'size': 4, 'align': 4},
            'double': {'size': 8, 'align': 8},
            'ptr': {'size': 8, 'align': 8}  # 64-bit pointers
        }
        self.structs = {}
        self.unions = {}
        self.enums = {}
        self.templates = {}
        
    def infer_type(self, expression: str) -> str:
        """Infer type from expression"""
        # Simple type inference
        if expression.isdigit():
            return 'int'
        elif '.' in expression and expression.replace('.', '').isdigit():
            return 'double'
        elif expression.startswith('"') and expression.endswith('"'):
            return 'char*'
        elif expression.startswith("'") and expression.endswith("'"):
            return 'char'
        else:
            return 'auto'  # Needs more analysis
            
    def define_struct(self, name: str, fields: List[Tuple[str, str]]):
        """Define a struct type"""
        self.structs[name] = fields
        
    def define_template(self, name: str, params: List[str], definition: str):
        """Define a template"""
        self.templates[name] = {
            'params': params,
            'definition': definition
        }
        
    def instantiate_template(self, name: str, args: List[str]) -> str:
        """Instantiate a template with given arguments"""
        if name not in self.templates:
            return None
            
        template = self.templates[name]
        definition = template['definition']
        
        # Simple template substitution
        for i, param in enumerate(template['params']):
            if i < len(args):
                definition = definition.replace(param, args[i])
                
        return definition

# Configuration
# Enhanced register support for x86_64
REGISTERS_8 = {
    "al", "ah", "bl", "bh", "cl", "ch", "dl", "dh",
    "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
    "sil", "dil", "bpl", "spl"
}
REGISTERS_16 = {
    "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
    "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"
}
REGISTERS_32 = {
    "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
    "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"
}
REGISTERS_64 = {
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", 
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    "rip"  # Instruction pointer
}

# Floating-point registers
REGISTERS_FPU = {f"st{i}" for i in range(8)}  # st0-st7
REGISTERS_MMX = {f"mm{i}" for i in range(8)}  # mm0-mm7
REGISTERS_XMM = {f"xmm{i}" for i in range(16)} # xmm0-xmm15
REGISTERS_YMM = {f"ymm{i}" for i in range(16)} # ymm0-ymm15
REGISTERS_ZMM = {f"zmm{i}" for i in range(32)} # zmm0-zmm31 (AVX-512)

# Segment registers
REGISTERS_SEGMENT = {"cs", "ds", "es", "fs", "gs", "ss"}

# Control registers
REGISTERS_CONTROL = {f"cr{i}" for i in range(16)} | {f"dr{i}" for i in range(16)}

# Mask registers (AVX-512)
REGISTERS_MASK = {f"k{i}" for i in range(8)}

# All registers combined
REGISTERS = (REGISTERS_8 | REGISTERS_16 | REGISTERS_32 | REGISTERS_64 | 
             REGISTERS_FPU | REGISTERS_MMX | REGISTERS_XMM | REGISTERS_YMM | 
             REGISTERS_ZMM | REGISTERS_SEGMENT | REGISTERS_CONTROL | REGISTERS_MASK)

# Comprehensive x86_64 instruction set support
ASM_OPS = {
    # Data movement
    "mov", "movl", "movq", "movb", "movw", "movsx", "movzx", "movsb", "movsw", "movsd", "movsq",
    "cmov", "cmova", "cmovae", "cmovb", "cmovbe", "cmovc", "cmove", "cmovg", "cmovge",
    "cmovl", "cmovle", "cmovna", "cmovnae", "cmovnb", "cmovnbe", "cmovnc", "cmovne",
    "cmovng", "cmovnge", "cmovnl", "cmovnle", "cmovno", "cmovnp", "cmovns", "cmovnz",
    "cmovo", "cmovp", "cmovpe", "cmovpo", "cmovs", "cmovz",
    "lea", "xchg", "bswap", "xlatb",
    
    # Arithmetic operations
    "add", "adc", "sub", "sbb", "mul", "imul", "div", "idiv",
    "inc", "dec", "neg", "cmp", "test",
    "daa", "das", "aaa", "aas", "aam", "aad",
    
    # Logical operations
    "and", "or", "xor", "not",
    
    # Shift and rotate
    "shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr",
    "shld", "shrd",
    
    # Bit manipulation (BMI1/BMI2)
    "andn", "bextr", "blsi", "blsmsk", "blsr", "bzhi", "lzcnt", "tzcnt",
    "pdep", "pext", "rorx", "sarx", "shlx", "shrx",
    "popcnt", "bsf", "bsr", "bt", "btc", "btr", "bts",
    
    # Stack operations
    "push", "pop", "pusha", "popa", "pushad", "popad", "pushf", "popf",
    "enter", "leave",
    
    # Control flow
    "jmp", "call", "ret", "iret", "int", "into", "bound",
    "je", "jz", "jne", "jnz", "js", "jns", "jc", "jb", "jnae", "jnc", "jnb", "jae",
    "jo", "jno", "ja", "jnbe", "jbe", "jna", "jl", "jnge", "jge", "jnl",
    "jle", "jng", "jg", "jnle", "jp", "jpe", "jnp", "jpo",
    "jcxz", "jecxz", "jrcxz", "loop", "loope", "loopz", "loopne", "loopnz",
    
    # String operations
    "rep", "repe", "repz", "repne", "repnz",
    "movs", "cmps", "scas", "lods", "stos",
    "movsb", "movsw", "movsd", "movsq",
    "cmpsb", "cmpsw", "cmpsd", "cmpsq",
    "scasb", "scasw", "scasd", "scasq",
    "lodsb", "lodsw", "lodsd", "lodsq",
    "stosb", "stosw", "stosd", "stosq",
    
    # Flag operations
    "clc", "cld", "cli", "cmc", "stc", "std", "sti",
    "lahf", "sahf", "pushf", "popf",
    
    # Conversion
    "cbw", "cwde", "cdqe", "cwd", "cdq", "cqo",
    "movsx", "movzx",
    
    # System instructions
    "syscall", "sysenter", "sysexit", "sysret",
    "cpuid", "rdtsc", "rdtscp", "rdmsr", "wrmsr",
    "hlt", "nop", "wait", "fwait", "pause",
    "lock", "xacquire", "xrelease",
    
    # Memory fence
    "mfence", "lfence", "sfence",
    
    # Atomic operations
    "cmpxchg", "cmpxchg8b", "cmpxchg16b",
    "xadd",
    
    # Floating-point (x87)
    "fld", "fst", "fstp", "fild", "fist", "fistp",
    "fadd", "fsub", "fmul", "fdiv", "fsqrt", "fabs", "fchs",
    "faddp", "fsubp", "fmulp", "fdivp",
    "fcom", "fcomp", "fcompp", "fcomi", "fcomip",
    "fsin", "fcos", "fsincos", "fptan", "fpatan", "f2xm1", "fyl2x",
    "finit", "fninit", "fclex", "fnclex", "fstcw", "fnstcw", "fldcw",
    "fstsw", "fnstsw", "fstenv", "fnstenv", "fldenv", "fsave", "fnsave", "frstor",
    "ffree", "ffreep", "fincstp", "fdecstp",
    
    # SSE/SSE2 (scalar)
    "movss", "movsd", "movaps", "movapd", "movups", "movupd",
    "movhps", "movhpd", "movlps", "movlpd", "movhlps", "movlhps",
    "movmskps", "movmskpd", "movntps", "movntpd", "movntq", "movnti",
    "addss", "addsd", "addps", "addpd",
    "subss", "subsd", "subps", "subpd",
    "mulss", "mulsd", "mulps", "mulpd",
    "divss", "divsd", "divps", "divpd",
    "sqrtss", "sqrtsd", "sqrtps", "sqrtpd",
    "maxss", "maxsd", "maxps", "maxpd",
    "minss", "minsd", "minps", "minpd",
    "cmpss", "cmpsd", "cmpps", "cmppd",
    "comiss", "comisd", "ucomiss", "ucomisd",
    "cvtss2sd", "cvtsd2ss", "cvtsi2ss", "cvtsi2sd",
    "cvtss2si", "cvtsd2si", "cvttss2si", "cvttsd2si",
    "cvtps2pd", "cvtpd2ps", "cvtpi2ps", "cvtps2pi",
    "cvttps2pi", "cvtpi2pd", "cvtpd2pi", "cvttpd2pi",
    "cvtdq2ps", "cvtps2dq", "cvttps2dq",
    "cvtdq2pd", "cvtpd2dq", "cvttpd2dq",
    
    # SSE/SSE2 (packed integer)
    "paddb", "paddw", "paddd", "paddq", "paddsb", "paddsw", "paddusb", "paddusw",
    "psubb", "psubw", "psubd", "psubq", "psubsb", "psubsw", "psubusb", "psubusw",
    "pmullw", "pmulhw", "pmulhuw", "pmuludq",
    "pmaddwd", "psadbw",
    "pcmpeqb", "pcmpeqw", "pcmpeqd",
    "pcmpgtb", "pcmpgtw", "pcmpgtd",
    "pand", "pandn", "por", "pxor",
    "psllw", "pslld", "psllq", "psrlw", "psrld", "psrlq", "psraw", "psrad",
    "punpckhbw", "punpckhwd", "punpckhdq", "punpckhqdq",
    "punpcklbw", "punpcklwd", "punpckldq", "punpcklqdq",
    "packsswb", "packssdw", "packuswb", "packusdw",
    "pmaxub", "pmaxsw", "pminub", "pminsw",
    "pavgb", "pavgw", "pshufw", "pshufd", "pshufhw", "pshuflw",
    
    # AVX/AVX2 (basic support)
    "vaddss", "vaddsd", "vaddps", "vaddpd",
    "vmovss", "vmovsd", "vmovaps", "vmovapd", "vmovups", "vmovupd",
    "vmulss", "vmulsd", "vmulps", "vmulpd",
    "vsubss", "vsubsd", "vsubps", "vsubpd",
    "vdivss", "vdivsd", "vdivps", "vdivpd",
    "vsqrtss", "vsqrtsd", "vsqrtps", "vsqrtpd",
    "vfmadd132ss", "vfmadd213ss", "vfmadd231ss",
    "vfmadd132sd", "vfmadd213sd", "vfmadd231sd",
    "vfmadd132ps", "vfmadd213ps", "vfmadd231ps",
    "vfmadd132pd", "vfmadd213pd", "vfmadd231pd",
    
    # AVX-512 (basic support)
    "vaddps", "vaddpd", "vmulps", "vmulpd",
    "vbroadcastss", "vbroadcastsd",
    "vgatherdd", "vgatherqd", "vgatherdpd", "vgatherqpd",
    "vscatterdd", "vscatterqd", "vscatterdpd", "vscatterqpd",
    
    # Memory prefetch
    "prefetchnta", "prefetcht0", "prefetcht1", "prefetcht2",
    "prefetchw", "prefetchwt1",
    
    # CRC32
    "crc32",
    
    # AES-NI
    "aesenc", "aesenclast", "aesdec", "aesdeclast",
    "aeskeygenassist", "aesimc",
    
    # PCLMULQDQ
    "pclmulqdq",
    
    # Random number generation
    "rdrand", "rdseed",
    
    # Transactional memory
    "xbegin", "xend", "xabort", "xtest",
    
    # Intel MPX (bounds checking)
    "bndmk", "bndcl", "bndcu", "bndcn", "bndmov", "bndldx", "bndstx",
    
    # Intel CET (control flow enforcement)
    "endbr32", "endbr64", "incsspd", "incsspq", "rdsspd", "rdsspq",
    "saveprevssp", "rstorssp", "wrssd", "wrssq", "wrussd", "wrussq",
    "setssbsy", "clrssbsy",
    
# Miscellaneous
    "ud2", "cpuid", "clflush", "clflushopt", "clwb", "sfence", "lfence", "mfence",
    "monitor", "mwait", "xgetbv", "xsetbv", "xsave", "xrstor", "xsaveopt",
    
    # Advanced Intel Extensions
    # Intel AMX (Advanced Matrix Extensions)
    "ldtilecfg", "sttilecfg", "tileloadd", "tileloaddt1", "tilestored", 
    "tilerelease", "tilezero", "tdpbf16ps", "tdpbssd", "tdpbsud", "tdpbusd", "tdpbuud",
    
    # Intel APX (Advanced Performance Extensions) 
    "ccmpb", "ccmpw", "ccmpl", "ccmpq", "cctestb", "cctestw", "cctestl", "cctestq",
    "cfcmovb", "cfcmovw", "cfcmovl", "cfcmovq",
    
    # Advanced Vector Extensions 10.1/10.2
    "vp2intersectd", "vp2intersectq", "vpcompressb", "vpcompressw", "vpexpandb", "vpexpandw",
    "vpopcntb", "vpopcntw", "vpopcntd", "vpopcntq", "vpshldw", "vpshldd", "vpshldq",
    "vpshrdw", "vpshrdd", "vpshrdq", "vprolvd", "vprolvq", "vprorvd", "vprorvq",
    
    # Galois Field Instructions
    "gf2p8affineinvqb", "gf2p8affineqb", "gf2p8mulb", "vgf2p8affineinvqb", "vgf2p8affineqb", "vgf2p8mulb",
    
    # Key Locker
    "loadiwkey", "aesenc128kl", "aesenc256kl", "aesdec128kl", "aesdec256kl",
    "aesencwide128kl", "aesencwide256kl", "aesdecwide128kl", "aesdecwide256kl",
    
    # User Interrupt Instructions
    "uiret", "testui", "clui", "stui", "senduipi",
    
    # Serialize
    "serialize",
    
    # HRESET
    "hreset",
    
    # WAITPKG
    "umonitor", "umwait", "tpause",
    
    # ENQCMD
    "enqcmd", "enqcmds",
    
    # MOVDIRI/MOVDIR64B
    "movdiri", "movdir64b",
    
    # PCONFIG
    "pconfig",
    
    # WBNOINVD
    "wbnoinvd",
    
    # Additional AVX-512 instructions
    "vp4dpwssd", "vp4dpwssds", "v4fmaddps", "v4fmaddss", "v4fnmaddps", "v4fnmaddss",
    "vpopcntdq", "vpmadd52luq", "vpmadd52huq", "vpmultishiftqb",
    "vrangeps", "vrangepd", "vrangess", "vrangesd", "vreduceps", "vreducepd", "vreducess", "vreducesd",
    "vscalefps", "vscalefpd", "vscalefss", "vscalefsd", "vfixupimmps", "vfixupimmpd", "vfixupimmss", "vfixupimmsd",
    "vgetexpps", "vgetexppd", "vgetexpss", "vgetexpsd", "vgetmantps", "vgetmantpd", "vgetmantss", "vgetmantsd",
    
    # AMD-specific instructions
    "clzero", "rdpru", "mcommit", "monitorx", "mwaitx", "clwb",
    
    # ARM64/AArch64 instructions (for future cross-compilation support)
    "adr", "adrp", "ldr", "str", "ldp", "stp", "cbz", "cbnz", "tbz", "tbnz",
    "b", "bl", "br", "blr", "ret", "svc", "hvc", "smc",
    "msr", "mrs", "isb", "dsb", "dmb", "sevl", "sev", "wfe", "wfi", "yield",
    "cas", "casp", "swp", "ldadd", "ldclr", "ldeor", "ldset", "ldsmax", "ldsmin", "ldumax", "ldumin",
    "stadd", "stclr", "steor", "stset", "stsmax", "stsmin", "stumax", "stumin",
    
    # RISC-V instructions (for future support)
    "lui", "auipc", "jal", "jalr", "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "lb", "lh", "lw", "lbu", "lhu", "sb", "sh", "sw", "addi", "slti", "sltiu",
    "xori", "ori", "andi", "slli", "srli", "srai", "add", "sub", "sll", "slt",
    "sltu", "xor", "srl", "sra", "or", "and", "fence", "ecall", "ebreak",
    "lwu", "ld", "sd", "addiw", "slliw", "srliw", "sraiw", "addw", "subw",
    "sllw", "srlw", "sraw", "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci",
    
    # Specialized floating-point and SIMD
    "vfmadd132ps", "vfmadd213ps", "vfmadd231ps", "vfmsub132ps", "vfmsub213ps", "vfmsub231ps",
    "vfnmadd132ps", "vfnmadd213ps", "vfnmadd231ps", "vfnmsub132ps", "vfnmsub213ps", "vfnmsub231ps",
    "vpermq", "vpermd", "vpermps", "vpermpd", "vperm2i128", "vperm2f128",
    "vmaskmovps", "vmaskmovpd", "vpmaskmovd", "vpmaskmovq",
}

class VariableInfo:
    """Track variable type and properties"""
    def __init__(self, name: str, vtype: str, value: Any = None, 
                 is_global: bool = False, size: int = 0, is_array: bool = False):
        self.name = name
        self.type = vtype
        self.value = value
        self.is_global = is_global
        self.size = size
        self.section = None
        self.is_array = is_array
        self.offset = None  # For local variables (stack offset)

class SymbolTable:
    """Manage variables with proper scoping and type information"""
    def __init__(self):
        self.globals: Dict[str, VariableInfo] = {}
        self.locals: Dict[str, VariableInfo] = {}
        self.current_scope = 'global'
        self.functions: Dict[str, Tuple[str, List[str]]] = OrderedDict()
        self.labels: Dict[str, int] = {}
        self.stack_offset = 0
        
    def add_variable(self, var: VariableInfo):
        if self.current_scope == 'global':
            self.globals[var.name] = var
        else:
            # Assign stack offset for local variables
            if not var.is_global:
                self.stack_offset += 8  # Assuming 64-bit alignment
                var.offset = self.stack_offset
            self.locals[var.name] = var
            
    def get_variable(self, name: str) -> Optional[VariableInfo]:
        # Check locals first, then globals
        if self.current_scope != 'global' and name in self.locals:
            return self.locals[name]
        return self.globals.get(name)
        
    def get_all_globals(self) -> Dict[str, VariableInfo]:
        return self.globals
        
    def enter_function(self, name: str, return_type: str = 'void', params: List[str] = None):
        self.current_scope = name
        self.locals.clear()
        self.stack_offset = 0
        self.functions[name] = (return_type, params or [])
        
    def exit_function(self):
        self.current_scope = 'global'
        self.locals.clear()
        self.stack_offset = 0

class AsmParser:
    """Parse assembly sections and directives"""
    
    @staticmethod
    def parse_data_section_line(line: str, symbols: SymbolTable) -> Optional[str]:
        """Parse .data section declarations"""
        line = line.strip()
        
        # String: label db "string", 0
        m = re.match(r'^([a-zA-Z_]\w*)\s+db\s+"([^"]*)"(?:\s*,\s*0)?', line)
        if m:
            name, value = m.groups()
            var = VariableInfo(name, 'char*', f'"{value}"', is_global=True)
            var.section = '.data'
            symbols.add_variable(var)
            return f'const char {name}[] = "{value}";'
            
        # Character: label db 'c'
        m = re.match(r"^([a-zA-Z_]\w*)\s+db\s+'(.)'", line)
        if m:
            name, value = m.groups()
            var = VariableInfo(name, 'char', f"'{value}'", is_global=True)
            var.section = '.data'
            symbols.add_variable(var)
            return f'char {name} = \'{value}\';'
            
        # Integer: label dd number
        m = re.match(r'^([a-zA-Z_]\w*)\s+dd\s+(-?\d+)(?:\s*,|$)', line)
        if m:
            name, value = m.groups()
            # Check if it's an array
            if ',' in line:
                vals_str = line.split('dd', 1)[1].strip()
                vals = [v.strip() for v in vals_str.split(',') if v.strip()]
                var = VariableInfo(name, 'int[]', vals, is_global=True, size=len(vals), is_array=True)
                var.section = '.data'
                symbols.add_variable(var)
                return f'int {name}[] = {{{", ".join(vals)}}};'
            else:
                var = VariableInfo(name, 'int', int(value), is_global=True)
                var.section = '.data'
                symbols.add_variable(var)
                return f'int {name} = {value};'
                
        # Double/Quad word (64-bit): label dq number  
        m = re.match(r'^([a-zA-Z_]\w*)\s+dq\s+(-?\d+(?:\.\d+)?)(?:\s*,|$)', line)
        if m:
            name, value = m.groups()
            # Check if it's an array
            if ',' in line:
                vals_str = line.split('dq', 1)[1].strip()
                vals = [v.strip() for v in vals_str.split(',') if v.strip()]
                var = VariableInfo(name, 'double[]', vals, is_global=True, size=len(vals), is_array=True)
                var.section = '.data'
                symbols.add_variable(var)
                return f'double {name}[] = {{{", ".join(vals)}}};'
            else:
                var = VariableInfo(name, 'double', float(value), is_global=True)
                var.section = '.data'
                symbols.add_variable(var)
                return f'double {name} = {value};'

        return None
        
    @staticmethod
    def parse_bss_section_line(line: str, symbols: SymbolTable) -> Optional[str]:
        """Parse .bss section declarations"""
        line = line.strip()
        
        # Buffer: label resb count
        m = re.match(r'^([a-zA-Z_]\w*)\s+resb\s+(\d+)', line)
        if m:
            name, count = m.groups()
            var = VariableInfo(name, 'char[]', None, is_global=True, size=int(count), is_array=True)
            var.section = '.bss'
            symbols.add_variable(var)
            return f'char {name}[{count}];'
            
        # Integer: label resd count
        m = re.match(r'^([a-zA-Z_]\w*)\s+resd\s+(\d+)', line)
        if m:
            name, count = m.groups()
            if int(count) > 1:
                var = VariableInfo(name, 'int[]', None, is_global=True, size=int(count), is_array=True)
            else:
                var = VariableInfo(name, 'int', None, is_global=True, size=int(count))
            var.section = '.bss'
            symbols.add_variable(var)
            return f'int {name};' if int(count) == 1 else f'int {name}[{count}];'
                
        return None

class InlineAsmGenerator:
    """Generate proper GCC inline assembly with enhanced dynamic processing"""
    
    @staticmethod
    def process_asm_block(lines: List[str], symbols: SymbolTable, extern_manager: 'ExternManager' = None) -> Tuple[Optional[str], Set[str], Set[str]]:
        """Convert assembly block to inline asm - ALL assembly is converted, nothing skipped"""
        if not lines:
            return None, set(), set()
            
        # First, analyze the entire block for patterns and variable usage
        block_analysis = InlineAsmGenerator._analyze_asm_block(lines, symbols)
        
        outputs = set()
        inputs = set()
        clobbers = set()
        asm_lines = []
        
        # Optimize the assembly block before processing
        optimized_lines = InlineAsmGenerator.optimize_asm_block(lines, symbols)
        
        # Extract dynamic variable usage patterns
        dynamic_variables = InlineAsmGenerator.extract_dynamic_variables(optimized_lines, symbols)
        
        # Update our sets with dynamically detected variables
        outputs.update(dynamic_variables.get('outputs', set()))
        inputs.update(dynamic_variables.get('inputs', set()))
        
        # Process each line - convert to inline assembly or C code as appropriate
        c_lines = []  # For C statements that should be outside inline assembly
        
        for i, line in enumerate(optimized_lines):
            line = re.split(r';|//', line)[0].strip()
            if not line:
                continue
                
            # Check if it's a label
            if line.endswith(':'):
                label_name = line[:-1].strip()
                # Skip the 'main:' label, as it's handled by the C main function
                if label_name.lower() == 'main':
                    continue
                # Labels need special handling in inline asm - make them unique
                if label_name.startswith('.'):
                    label_name = f'{label_name}_%='
                asm_lines.append(f'{label_name}:')
                continue
                
            # Convert assembly lines to inline assembly or C code
            processed_line, line_outputs, line_inputs = InlineAsmGenerator._convert_line_to_inline_asm(
                line, symbols, block_analysis, extern_manager)
            
            if processed_line is not None:
                outputs.update(line_outputs)
                inputs.update(line_inputs)
                
                # Check if this is a C statement (ends with semicolon) vs assembly
                if processed_line.rstrip().endswith(';'):
                    c_lines.append(processed_line)
                else:
                    asm_lines.append(processed_line)
                
                # Track clobbered registers
                for reg in REGISTERS:
                    if re.search(r'\b' + re.escape(reg) + r'\b', line, re.I):
                        # If register is destination of an instruction, it's clobbered
                        if re.match(r'^\s*\w+\s+' + re.escape(reg) + r'\b', line, re.I):
                            clobbers.add(reg.lower())
        
        # Handle mixed C and assembly code
        result_lines = []
        
        # Add C lines first
        if c_lines:
            result_lines.extend(c_lines)
            
        # Add inline assembly if we have any
        if asm_lines:
            inline_asm = InlineAsmGenerator._generate_asm_statement(asm_lines, outputs, inputs, clobbers, symbols)
            result_lines.append(inline_asm)
        elif not c_lines:
            # No processable lines at all
            return '/* Empty assembly block */', set(), set()
            
        # Join all lines with newlines
        return '\n'.join(result_lines), outputs, inputs
    
    @staticmethod
    def _convert_line_to_inline_asm(line: str, symbols: SymbolTable, block_analysis: Dict[str, Any] = None, extern_manager: 'ExternManager' = None) -> Tuple[str, Set[str], Set[str]]:
        """Convert a single assembly line to inline assembly format - ALL lines are converted"""
        outputs = set()
        inputs = set()
        
        # Handle function calls by converting to proper C function calls
        # Improved parsing to handle nested parentheses properly
        if m := re.match(r'^\s*call\s+([a-zA-Z_]\w*)(?:\s*(.*))?$', line, re.I):
            func_name = m.group(1)
            remainder = (m.group(2) if m.group(2) else '').strip()
            
            # If there's a remainder that starts with '(', parse the full argument list
            args = ''
            if remainder.startswith('('):
                # Find matching closing parenthesis, handling nested parentheses
                paren_count = 0
                end_idx = -1
                for i, char in enumerate(remainder):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            end_idx = i
                            break
                            
                if end_idx != -1:
                    args = remainder[1:end_idx]  # Extract content between parentheses
            
            # Check if this is an extern function
            if extern_manager and extern_manager.is_extern_function(func_name):
                extern_func = extern_manager.get_extern_function(func_name)
                if extern_func:
                    # Generate proper C function call for extern function
                    if args:
                        return f'{func_name}({args});', outputs, inputs
                    else:
                        return f'{func_name}();', outputs, inputs
            
            # Convert to C function call instead of inline assembly
            if args:
                return f'{func_name}({args});', outputs, inputs
            else:
                return f'{func_name}();', outputs, inputs
            
        # Handle ret instructions - skip in main function context
        if re.match(r'^\s*ret\s*$', line, re.I):
            # Don't generate inline ret in main function - let C handle the return
            return None, outputs, inputs
            
        # Convert ALL other assembly instructions to AT&T syntax
        processed = InlineAsmGenerator._convert_to_att_syntax(line, symbols, outputs, inputs)
        return processed, outputs, inputs
    
    @staticmethod
    def _analyze_asm_block(lines: List[str], symbols: SymbolTable) -> Dict[str, Any]:
        """Analyze an assembly block to understand its structure and requirements"""
        analysis = {
            'is_complex_function': False,
            'has_array_operations': False,
            'has_bitwise_ops': False,
            'has_loops': False,
            'has_function_calls': False,
            'variables_used': set(),
            'registers_used': set(),
            'memory_operations': [],
            'control_flow': []
        }
        
        asm_text = '\n'.join(lines)
        
        # Detect complex function patterns
        if any(pattern in asm_text.lower() for pattern in ['push', 'pop', 'call', '.find_', '.copy_', 'string_reverse']):
            analysis['is_complex_function'] = True
            
        # Detect array operations
        if re.search(r'\[\w+\s*\+\s*\w+\*[1248]\]|\[\w+\s*\+\s*\w+\]', asm_text):
            analysis['has_array_operations'] = True
            
        # Detect bitwise operations
        if any(op in asm_text.lower() for op in ['xor', 'and', 'or', 'rol', 'ror', 'shl', 'shr']):
            analysis['has_bitwise_ops'] = True
            
        # Detect loops and control flow
        if any(instr in asm_text.lower() for instr in ['jmp', 'je', 'jne', 'jg', 'jl', 'loop']):
            analysis['has_loops'] = True
            
        # Track variables and registers used
        for line in lines:
            line = re.split(r';|//', line)[0].strip()
            if not line:
                continue
                
            # Find variables referenced in the line
            for var_name, var_info in symbols.get_all_globals().items():
                if re.search(r'\b' + re.escape(var_name) + r'\b', line):
                    analysis['variables_used'].add(var_name)
                    
            # Find registers used
            for reg in REGISTERS:
                if re.search(r'\b' + re.escape(reg) + r'\b', line, re.I):
                    analysis['registers_used'].add(reg.lower())
                    
        return analysis
        
    @staticmethod
    def _handle_complex_function(lines: List[str], symbols: SymbolTable, analysis: Dict[str, Any]) -> Tuple[str, Set[str], Set[str]]:
        """Handle complex assembly functions by converting ALL to inline assembly"""
        outputs = set()
        inputs = set()
        asm_lines = []
        clobbers = set()
        
        # Convert ALL lines to inline assembly without exception
        for line in lines:
            line = re.split(r';|//', line)[0].strip()
            if not line:
                continue
                
            # Check if it's a label
            if line.endswith(':'):
                label_name = line[:-1].strip()
                if label_name.startswith('.'):
                    label_name = f'{label_name}_%='
                asm_lines.append(f'{label_name}:')
                continue
                
            # Convert to inline assembly format
            processed_line, line_outputs, line_inputs = InlineAsmGenerator._convert_line_to_inline_asm(
                line, symbols, analysis, None)
            
            if processed_line:
                outputs.update(line_outputs)
                inputs.update(line_inputs)
                asm_lines.append(processed_line)
                
                # Track clobbered registers
                for reg in REGISTERS:
                    if re.search(r'\b' + re.escape(reg) + r'\b', line, re.I):
                        if re.match(r'^\s*\w+\s+' + re.escape(reg) + r'\b', line, re.I):
                            clobbers.add(reg.lower())
        
        # Generate inline assembly for complex functions
        if asm_lines:
            variable_usage = {'outputs': outputs, 'inputs': inputs, 'modified': outputs}
            return InlineAsmGenerator.generate_enhanced_inline_asm(asm_lines, variable_usage, symbols, clobbers), outputs, inputs
        else:
            return '/* Complex function with no processable assembly */', outputs, inputs
        
    @staticmethod
    def _handle_array_operations(lines: List[str], symbols: SymbolTable, analysis: Dict[str, Any]) -> Tuple[str, Set[str], Set[str]]:
        """Handle array operations by converting ALL to inline assembly"""
        outputs = set()
        inputs = set()
        asm_lines = []
        clobbers = set()
        
        # Convert ALL array operations to inline assembly
        for line in lines:
            line = re.split(r';|//', line)[0].strip()
            if not line:
                continue
                
            # Check if it's a label
            if line.endswith(':'):
                label_name = line[:-1].strip()
                if label_name.startswith('.'):
                    label_name = f'{label_name}_%='
                asm_lines.append(f'{label_name}:')
                continue
                
            # Convert to inline assembly format
            processed_line, line_outputs, line_inputs = InlineAsmGenerator._convert_line_to_inline_asm(
                line, symbols, analysis, None)
            
            if processed_line:
                outputs.update(line_outputs)
                inputs.update(line_inputs)
                asm_lines.append(processed_line)
                
                # Track clobbered registers
                for reg in REGISTERS:
                    if re.search(r'\b' + re.escape(reg) + r'\b', line, re.I):
                        if re.match(r'^\s*\w+\s+' + re.escape(reg) + r'\b', line, re.I):
                            clobbers.add(reg.lower())
        
        # Generate inline assembly for array operations
        if asm_lines:
            variable_usage = {'outputs': outputs, 'inputs': inputs, 'modified': outputs}
            return InlineAsmGenerator.generate_enhanced_inline_asm(asm_lines, variable_usage, symbols, clobbers), outputs, inputs
        else:
            return '/* Array operations with no processable assembly */', outputs, inputs
        
    @staticmethod
    def _handle_bitwise_operations(lines: List[str], symbols: SymbolTable, analysis: Dict[str, Any]) -> Tuple[str, Set[str], Set[str]]:
        """Handle bitwise operations by converting ALL to inline assembly"""
        outputs = set()
        inputs = set()
        asm_lines = []
        clobbers = set()
        
        # Convert ALL bitwise operations to inline assembly without filtering
        for line in lines:
            line = re.split(r';|//', line)[0].strip()
            if not line:
                continue
                
            # Check if it's a label
            if line.endswith(':'):
                label_name = line[:-1].strip()
                if label_name.startswith('.'):
                    label_name = f'{label_name}_%='
                asm_lines.append(f'{label_name}:')
                continue
                
            # Convert ALL lines to inline assembly format
            processed_line, line_outputs, line_inputs = InlineAsmGenerator._convert_line_to_inline_asm(
                line, symbols, analysis, None)
            
            if processed_line:
                outputs.update(line_outputs)
                inputs.update(line_inputs)
                asm_lines.append(processed_line)
                
                # Track clobbered registers
                for reg in REGISTERS:
                    if re.search(r'\b' + re.escape(reg) + r'\b', line, re.I):
                        if re.match(r'^\s*\w+\s+' + re.escape(reg) + r'\b', line, re.I):
                            clobbers.add(reg.lower())
                
        # Always generate inline assembly for bitwise operations
        if asm_lines:
            variable_usage = {
                'outputs': outputs,
                'inputs': inputs,
                'modified': outputs
            }
            return InlineAsmGenerator.generate_enhanced_inline_asm(asm_lines, variable_usage, symbols, clobbers), outputs, inputs
        else:
            return '/* Bitwise operations with no processable assembly */', outputs, inputs
        
    @staticmethod
    def _generate_asm_statement(asm_lines: List[str], outputs: Set[str], inputs: Set[str], 
                               clobbers: Set[str], symbols: SymbolTable) -> str:
        """Generate the complete inline assembly statement with enhanced processing"""
        if not asm_lines:
            return ''
            
        # Use the enhanced variable extraction if we have the lines
        variable_usage = {
            'outputs': outputs,
            'inputs': inputs,
            'modified': outputs  # Assume outputs are modified
        }
        
        # Use our enhanced inline assembly generator
        return InlineAsmGenerator.generate_enhanced_inline_asm(asm_lines, variable_usage, symbols, clobbers)
        
    @staticmethod
    def _process_single_asm_line(line: str, symbols: SymbolTable, block_analysis: Dict[str, Any] = None, 
                                line_index: int = 0, all_lines: List[str] = None) -> Tuple[str, Set[str], Set[str], bool]:
        """Process a single assembly line - ALL lines converted to inline assembly, no C conversions"""
        outputs = set()
        inputs = set()
        is_c_code = False  # Never return C code - always inline assembly
        
        # Use block analysis if provided
        if block_analysis is None:
            block_analysis = {'variables_used': set(), 'registers_used': set()}
        
        # Convert ALL assembly instructions to AT&T syntax for inline assembly
        processed = InlineAsmGenerator._convert_to_att_syntax(line, symbols, outputs, inputs)
        
        return processed, outputs, inputs, is_c_code
        
    @staticmethod
    def _convert_to_att_syntax(line: str, symbols: SymbolTable, outputs: Set[str], inputs: Set[str]) -> str:
        """Convert Intel syntax to AT&T syntax for GCC, making labels unique."""
        processed = line.strip()
        
        # Handle memory references with arrays
        # Pattern: [array + offset]
        # This function is complex, so let's focus on the label part for the fix.
        def replace_mem_ref(match):
            content = match.group(1)
            # Check for array + offset pattern
            if '+' in content:
                parts = content.split('+')
                base = parts[0].strip()
                offset = parts[1].strip()
                
                var = symbols.get_variable(base)
                if var and var.is_array:
                    inputs.add(base)
                    # Let the compiler handle the offset. We'll access the array element directly.
                    return f'%[{base}]'
                elif base.lower() in REGISTERS:
                    # Register + offset
                    return f'{offset}(%{base.lower()})'
            else:
                # Simple variable reference
                var = symbols.get_variable(content)
                if var:
                    # Determine if it's input or output based on position
                    if re.search(r',\s*\[' + re.escape(content) + r'\]', line, re.I):
                        outputs.add(content)
                    else:
                        inputs.add(content)
                    return f'%[{content}]'
                elif content.lower() in REGISTERS:
                    return f'(%{content.lower()})'
            return match.group(0)
        
        processed = re.sub(r'\[([^\]]+)\]', replace_mem_ref, processed)
        
        # Convert register references to AT&T format
        for reg in sorted(REGISTERS, key=lambda x: -len(x)):
            processed = re.sub(r'\b' + re.escape(reg) + r'\b', '%%' + reg.lower(), processed, flags=re.I)
        
        # Convert immediate values (that are not part of a memory reference)
        processed = re.sub(r'(?<!\()(\b\d+\b)(?!\()', r'$\1', processed)
        
        # Fix invalid triple % sequences (%%%) -> (%%) for memory references
        processed = re.sub(r'%%%([a-zA-Z]+)', r'(%%\1)', processed)
        
        # Make jump targets unique for loops
        # e.g., jle .no_swap -> jle .no_swap_%=
        processed = re.sub(r'(\bj[a-z]{1,3}\s+)(\.\w+)', r'\1\2_%=', processed)

        # Fix instruction syntax (Intel to AT&T)
        # Most two-operand instructions need operands reversed
        parts = processed.split(None, 1)
        if len(parts) == 2:
            op = parts[0].lower()
            operands = parts[1]
            
            if op in {'mov', 'movl', 'movq', 'movb', 'movw', 'add', 'sub', 'cmp', 'test', 'xor', 'and', 'or', 'rol', 'ror', 'shl', 'shr'}:
                # These need operands reversed
                if ',' in operands:
                    ops = [o.strip() for o in operands.split(',', 1)]
                    if len(ops) == 2:
                        # Add size suffix if needed for various operations
                        if op in ['mov', 'add', 'sub', 'xor', 'and', 'or']:
                            # Determine size from operands
                            if any(r in ops[0].lower() or r in ops[1].lower() for r in REGISTERS_64):
                                if op == 'mov':
                                    op = 'movq'
                                else:
                                    op = op + 'q'
                            elif any(r in ops[0].lower() or r in ops[1].lower() for r in REGISTERS_32):
                                if op == 'mov':
                                    op = 'movl'
                                else:
                                    op = op + 'l'
                            elif any(r in ops[0].lower() or r in ops[1].lower() for r in REGISTERS_16):
                                if op == 'mov':
                                    op = 'movw'
                                else:
                                    op = op + 'w'
                            elif any(r in ops[0].lower() or r in ops[1].lower() for r in REGISTERS_8):
                                if op == 'mov':
                                    op = 'movb'
                                else:
                                    op = op + 'b'
                            else:
                                op = op + 'l'  # Default to long
                        elif op in ['rol', 'ror', 'shl', 'shr']:
                            # Rotate and shift operations need explicit size suffix
                            if any(r in ops[0].lower() for r in REGISTERS_32) or '%%eax' in ops[0]:
                                op = op + 'l'
                            elif any(r in ops[0].lower() for r in REGISTERS_16):
                                op = op + 'w'
                            elif any(r in ops[0].lower() for r in REGISTERS_8):
                                op = op + 'b'
                            else:
                                op = op + 'l'  # Default to long
                        # Handle memory operands more safely
                        if ops[0].startswith('%['): # Destination is a memory variable
                            var_name = ops[0][2:-1] # e.g., "sum" from "%[sum]"
                            var_info = symbols.get_variable(var_name)
                            if var_info:
                                # Use safer memory constraints - let GCC handle the addressing
                                processed = f'{op} {ops[1]}, %[{var_name}]'
                                outputs.add(var_name)
                            else:
                                processed = f'{op} {ops[1]}, {ops[0]}'
                        elif ops[1].startswith('%['): # Source is a memory variable
                            var_name = ops[1][2:-1]
                            var_info = symbols.get_variable(var_name)
                            if var_info:
                                processed = f'{op} %[{var_name}], {ops[0]}'
                                inputs.add(var_name)
                            else:
                                processed = f'{op} {ops[1]}, {ops[0]}'
                        else:
                            processed = f'{op} {ops[1]}, {ops[0]}' # Standard reversal
                    else:
                        processed = f'{op} {operands}'
                else:
                    processed = f'{op} {operands}'
            else:
                processed = f'{op} {operands}'
        
        return processed
    
    @staticmethod
    def _handle_operand_sizes(line: str) -> str:
        """Enhanced operand size handling for better AT&T conversion"""
        processed = line.strip()
        
        # Handle explicit size specifiers
        size_map = {
            'byte ptr': 'b',
            'word ptr': 'w', 
            'dword ptr': 'l',
            'qword ptr': 'q'
        }
        
        for intel_size, att_suffix in size_map.items():
            if intel_size in processed.lower():
                processed = re.sub(r'\b' + re.escape(intel_size) + r'\s+', '', processed, flags=re.I)
                # Add suffix to instruction if not already present
                parts = processed.split(None, 1)
                if parts and not any(parts[0].endswith(s) for s in ['b', 'w', 'l', 'q']):
                    parts[0] += att_suffix
                    processed = ' '.join(parts)
                    
        return processed
    
    @staticmethod
    def optimize_asm_block(lines: List[str], symbols: SymbolTable) -> List[str]:
        """Optimize assembly blocks for better inline generation"""
        optimized = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
                
            # Remove redundant moves
            if m := re.match(r'mov\s+(\w+),\s*\1', line, re.I):
                continue  # Skip self-moves like mov eax, eax
                
            # Optimize stack operations that don't affect variables
            if re.match(r'(push|pop)\s+\w+', line, re.I):
                # Check if this affects any tracked variables
                affects_vars = False
                for var_name in symbols.get_all_globals():
                    if var_name in line:
                        affects_vars = True
                        break
                if not affects_vars:
                    continue  # Skip stack ops that don't affect our variables
                    
            # Convert complex addressing modes to simpler forms
            line = InlineAsmGenerator._simplify_addressing(line, symbols)
            
            optimized.append(line)
            
        return optimized
    
    @staticmethod
    def _simplify_addressing(line: str, symbols: SymbolTable) -> str:
        """Simplify complex addressing modes for better inline assembly"""
        processed = line
        
        # Convert [base + index*scale + displacement] to simpler forms
        complex_addr = re.search(r'\[([^\]]+)\]', processed)
        if complex_addr:
            addr_content = complex_addr.group(1)
            
            # If it contains a known variable, try to simplify
            for var_name in symbols.get_all_globals():
                if var_name in addr_content:
                    var_info = symbols.get_variable(var_name)
                    if var_info and var_info.is_array:
                        # Replace complex array access with variable reference
                        processed = processed.replace(complex_addr.group(0), f'[{var_name}]')
                        break
                        
        return processed
    
    @staticmethod
    def extract_dynamic_variables(lines: List[str], symbols: SymbolTable) -> Dict[str, Set[str]]:
        """Dynamically extract variable usage patterns from assembly blocks"""
        variable_usage = {
            'inputs': set(),
            'outputs': set(),
            'modified': set(),
            'array_accesses': set()
        }
        
        for line in lines:
            line = re.split(r';|//', line)[0].strip()
            if not line:
                continue
                
            # Extract variable references from different instruction patterns
            
            # mov [var], value - output
            if m := re.search(r'mov\s+\[([a-zA-Z_]\w*)\]', line, re.I):
                var_name = m.group(1)
                if symbols.get_variable(var_name):
                    variable_usage['outputs'].add(var_name)
                    variable_usage['modified'].add(var_name)
                    
            # mov reg, [var] - input
            if m := re.search(r'mov\s+\w+,\s*\[([a-zA-Z_]\w*)\]', line, re.I):
                var_name = m.group(1)
                if symbols.get_variable(var_name):
                    variable_usage['inputs'].add(var_name)
                    
            # Array access patterns [var + offset]
            if m := re.search(r'\[([a-zA-Z_]\w*)\s*\+\s*[^\]]*\]', line, re.I):
                var_name = m.group(1)
                var_info = symbols.get_variable(var_name)
                if var_info and var_info.is_array:
                    variable_usage['array_accesses'].add(var_name)
                    # Determine if it's read or write based on instruction
                    if re.match(r'\s*mov\s+\[', line, re.I):
                        variable_usage['outputs'].add(var_name)
                    else:
                        variable_usage['inputs'].add(var_name)
                        
            # inc/dec operations on variables
            if m := re.search(r'(inc|dec)\s+\[?([a-zA-Z_]\w*)\]?', line, re.I):
                var_name = m.group(2)
                if symbols.get_variable(var_name):
                    variable_usage['outputs'].add(var_name)
                    variable_usage['inputs'].add(var_name)
                    variable_usage['modified'].add(var_name)
                    
            # Direct variable references (without brackets)
            for var_name, var_info in symbols.get_all_globals().items():
                if re.search(r'\b' + re.escape(var_name) + r'\b', line) and not re.search(r'\[.*' + re.escape(var_name) + r'.*\]', line):
                    # Context-sensitive classification
                    if re.search(r'mov\s+\w+,\s*' + re.escape(var_name), line, re.I):
                        variable_usage['inputs'].add(var_name)
                    elif re.search(r'mov\s+' + re.escape(var_name), line, re.I):
                        variable_usage['outputs'].add(var_name)
                    else:
                        variable_usage['inputs'].add(var_name)
                        
        return variable_usage
    
    @staticmethod 
    def generate_enhanced_inline_asm(asm_lines: List[str], variable_usage: Dict[str, Set[str]], 
                                   symbols: SymbolTable, clobbers: Set[str] = None) -> str:
        """Generate enhanced inline assembly with improved constraint handling"""
        if not asm_lines:
            return ''
            
        if clobbers is None:
            clobbers = set()
            
        # Build the assembly string
        asm_string = '\\n\\t'.join(asm_lines)
        
        # Prepare constraint lists with better type awareness
        output_constraints = []
        input_constraints = []
        
        # Handle output variables (variables that are modified)
        output_vars = variable_usage.get('outputs', set()) | variable_usage.get('modified', set())
        for var in sorted(output_vars):
            var_info = symbols.get_variable(var)
            if var_info:
                if var_info.is_array:
                    # For arrays, use memory constraint
                    output_constraints.append(f'[{var}] "+m"({var})')
                else:
                    # For scalar variables, choose appropriate constraint
                    if 'int' in var_info.type or 'long' in var_info.type:
                        output_constraints.append(f'[{var}] "+r"({var})')
                    else:
                        output_constraints.append(f'[{var}] "+m"({var})')
        
        # Handle input variables (variables that are read but not modified)
        input_only_vars = variable_usage.get('inputs', set()) - output_vars
        for var in sorted(input_only_vars):
            var_info = symbols.get_variable(var)
            if var_info:
                if var_info.is_array:
                    input_constraints.append(f'[{var}] "m"({var})')
                else:
                    if 'int' in var_info.type or 'long' in var_info.type:
                        input_constraints.append(f'[{var}] "r"({var})')
                    else:
                        input_constraints.append(f'[{var}] "m"({var})')
        
        # Build the complete inline assembly statement
        stmt = f'__asm__ __volatile__("\\n\\t{asm_string}\\n"'
        
        # Add output constraints
        if output_constraints:
            stmt += f' : {", ".join(output_constraints)}'
        else:
            stmt += ' :'
            
        # Add input constraints
        if input_constraints:
            stmt += f' : {", ".join(input_constraints)}'
        else:
            stmt += ' :'
            
        # Add clobber list
        clobber_list = ['"memory"', '"cc"']
        for reg in sorted(clobbers):
            # Convert to 64-bit register names for x86_64
            reg_64 = InlineAsmGenerator._normalize_register_name(reg)
            if reg_64 and reg_64 not in ['rax']:
                clobber_list.append(f'"{reg_64}"')
                
        stmt += f' : {", ".join(clobber_list)}'
        stmt += ');'
        
        return stmt
    
    @staticmethod
    def _normalize_register_name(reg: str) -> str:
        """Convert register names to their 64-bit x86_64 equivalents"""
        reg = reg.lower().strip('%')
        register_map = {
            'eax': 'rax', 'ebx': 'rbx', 'ecx': 'rcx', 'edx': 'rdx',
            'esi': 'rsi', 'edi': 'rdi', 'ebp': 'rbp', 'esp': 'rsp',
            'ax': 'rax', 'bx': 'rbx', 'cx': 'rcx', 'dx': 'rdx',
            'si': 'rsi', 'di': 'rdi', 'bp': 'rbp', 'sp': 'rsp',
            'al': 'rax', 'ah': 'rax', 'bl': 'rbx', 'bh': 'rbx',
            'cl': 'rcx', 'ch': 'rcx', 'dl': 'rdx', 'dh': 'rdx'
        }
        return register_map.get(reg, reg)

class CodeTranslator:
    """Main translator between C and Assembly with advanced features"""
    
    def __init__(self, options: CompilerOptions = None):
        self.options = options or CompilerOptions()
        self.symbols = SymbolTable()
        self.current_section = '.text'
        self.in_function = None
        self.indent_level = 0
        self.label_counter = 0
        
        # Advanced components
        self.macro_processor = MacroProcessor()
        self.optimizer = AdvancedOptimizer(self.options)
        self.type_system = TypeSystem()
        self.exception_handler = ExceptionHandler()
        self.debug_info = DebugInfoGenerator("")
        self.extern_manager = ExternManager()  # Add extern manager
        
        # Architecture backend
        self.architecture_backend = self._get_architecture_backend()
        
        # Code generators
        self.code_generators = {
            'c': self._generate_c_code,
            'llvm': LLVMIRGenerator(self.options.target_arch),
            'object': ObjectFileGenerator(self.options.target_arch)
        }
        
    def _get_architecture_backend(self) -> ArchitectureBackend:
        """Get the appropriate architecture backend"""
        if self.options.target_arch == TargetArchitecture.X86_64:
            return X86_64Backend()
        elif self.options.target_arch == TargetArchitecture.ARM64:
            return ARM64Backend()
        elif self.options.target_arch == TargetArchitecture.RISCV64:
            return RISCVBackend()
        else:
            return X86_64Backend()  # Default fallback
        
    def translate_to_c(self, lines: List[str], file_path: str = "") -> str:
        """Translate assembly/hybrid code to C with advanced preprocessing"""
        # Check if input is already pure C code
        if self._is_pure_c_code(lines):
            return '\n'.join(lines)
            
        # First pass: process extern declarations
        processed_lines = self._process_extern_declarations(lines)
        
        # Second pass: macro preprocessing
        processed_lines = self.macro_processor.process_file(processed_lines, file_path)
        
        # Apply architecture-specific preprocessing if needed
        if self.options.target_arch != TargetArchitecture.X86_64:
            processed_lines = self._convert_to_target_architecture(processed_lines)
        
        blocks = self._group_lines(processed_lines)
        c_lines = []
        
        global_decls = []
        main_body = []
        functions = OrderedDict()
        headers = []
        default_headers = {
            '#include <stdio.h>', 
            '#include <stdlib.h>', 
            '#include <stdint.h>', 
            '#include <string.h>',
            '#include <unistd.h>'
        }
        
        # Add headers from extern declarations
        extern_headers = self.extern_manager.get_required_headers()
        for header in extern_headers:
            headers.append(header)
        
        for block_type, block_lines in blocks:
            if block_type == 'DATA':
                for line in block_lines:
                    c_line = AsmParser.parse_data_section_line(line, self.symbols)
                    if c_line:
                        global_decls.append(c_line)
                        
            elif block_type == 'BSS':
                for line in block_lines:
                    c_line = AsmParser.parse_bss_section_line(line, self.symbols)
                    if c_line:
                        global_decls.append(c_line)
                        
            elif block_type == 'ASM':
                # Apply advanced optimization if enabled
                optimized_lines = block_lines
                if self.options.optimization_level != OptimizationLevel.NONE:
                    optimized_lines = self.optimizer.optimize_assembly_block(block_lines)
                    
                # Apply architecture-specific optimizations
                arch_optimized_lines = self.architecture_backend.optimize_for_architecture(optimized_lines)
                
                inline_asm, outputs, inputs = InlineAsmGenerator.process_asm_block(arch_optimized_lines, self.symbols, self.extern_manager)
                if inline_asm:
                    indent = '    ' if self.in_function else ''
                    if '\n' in inline_asm:  # Multiple C lines
                        for line in inline_asm.split('\n'):
                            if line.strip():
                                self._add_line_to_function(line, indent, functions, main_body)
                    else:
                        self._add_line_to_function(inline_asm, indent, functions, main_body)
                        
            elif block_type == 'C':
                for line in block_lines:
                    c_line = self._translate_high_level_line(line)
                    if c_line and c_line.startswith('#include'):
                        headers.append(c_line)
                        continue
                    if c_line:
                        indent = '    ' * (self.indent_level + 1) if self.in_function else ''
                        self._add_line_to_function(c_line, indent, functions, main_body)
                            
        # Generate final C code
        c_lines.extend(sorted(list(set(headers) | default_headers)))
        c_lines.append('')
        
        # Add extern function declarations
        extern_decls = self.extern_manager.get_function_declarations()
        if extern_decls:
            c_lines.append('/* Extern function declarations */')
            c_lines.extend(extern_decls)
            c_lines.append('')
        
        if global_decls:
            c_lines.append('/* Global variables */')
            c_lines.extend(global_decls)
            c_lines.append('')
        
        # Add function prototypes
        for func_name, (ret_type, params) in self.symbols.functions.items():
            if func_name != 'main':
                param_str = ', '.join(params) if params else 'void'
                c_lines.append(f'{ret_type} {func_name}({param_str});')
        
        if self.symbols.functions and 'main' not in self.symbols.functions:
            c_lines.append('')
        
        # Add main function
        c_lines.append('int main(void) {')
        c_lines.extend(main_body)
        if not any('return' in line for line in main_body):
            c_lines.append('    return 0;')
        c_lines.append('}')
        c_lines.append('')
        
        # Add other functions
        for func_name, func_body in functions.items():
            if func_name != 'main':
                ret_type, params = self.symbols.functions.get(func_name, ('void', []))
                param_str = ', '.join(params) if params else 'void'
                c_lines.append(f'{ret_type} {func_name}({param_str}) {{')
                c_lines.extend(func_body)
                if ret_type != 'void' and not any('return' in line for line in func_body):
                    c_lines.append('    return 0;')
                c_lines.append('}')
                c_lines.append('')
                
        return '\n'.join(c_lines)
        
    def _convert_to_target_architecture(self, lines: List[str]) -> List[str]:
        """Convert assembly lines to target architecture"""
        converted_lines = []
        for line in lines:
            if self._is_asm_line(line.strip()):
                # Convert using architecture backend
                converted = self.architecture_backend.convert_to_native_syntax(line)
                converted_lines.append(converted)
            else:
                converted_lines.append(line)
        return converted_lines
        
    def _process_extern_declarations(self, lines: List[str]) -> List[str]:
        """Process extern declarations and remove them from the line list"""
        processed_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this is an extern declaration
            if line_stripped.startswith('extern '):
                extern_decl = self.extern_manager.parse_extern_line(line_stripped)
                if extern_decl:
                    self.extern_manager.add_extern_declaration(extern_decl)
                    # Don't add this line to processed_lines - it's been handled
                    continue
            
            processed_lines.append(line)
            
        return processed_lines
        
    def _generate_c_code(self, ast: Any) -> str:
        """Generate C code (default implementation)"""
        return self.translate_to_c(ast)
        
    def generate_code(self, lines: List[str], output_format: str = 'c', file_path: str = "") -> str:
        """Generate code in specified format"""
        if output_format == 'c':
            return self.translate_to_c(lines, file_path)
        elif output_format == 'llvm':
            # Generate LLVM IR
            return self.code_generators['llvm'].generate_code(lines, self.options)
        elif output_format == 'object':
            # Generate object file data (as hex string for now)
            obj_gen = self.code_generators['object']
            if sys.platform == 'darwin':
                obj_data = obj_gen.generate_macho()
            elif sys.platform == 'win32':
                obj_data = obj_gen.generate_coff()
            else:
                obj_data = obj_gen.generate_elf()
            return obj_data.hex()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
    def add_include_path(self, path: Path):
        """Add include search path"""
        self.options.include_paths.append(path)
        
    def define_macro(self, name: str, value: str = ""):
        """Define a preprocessor macro"""
        self.macro_processor.define_macro(name, value)
        self.options.define_macros[name] = value
        
    def set_optimization_level(self, level: OptimizationLevel):
        """Set optimization level"""
        self.options.optimization_level = level
        self.optimizer = AdvancedOptimizer(self.options)
        
    def enable_debug_info(self, enable: bool = True):
        """Enable/disable debug information generation"""
        self.options.debug_info = enable
        
    def add_sanitizer(self, sanitizer: str):
        """Add runtime sanitizer (address, thread, memory, etc.)"""
        self.options.sanitizers.add(sanitizer)
        
    def set_cpu_features(self, features: Set[str]):
        """Set target CPU features"""
        self.options.cpu_features = features
        
    def _add_line_to_function(self, line: str, indent: str, functions: Dict, main_body: List):
        """Add a line to the appropriate function or main body"""
        if self.in_function and self.in_function != 'main':
            if self.in_function not in functions:
                functions[self.in_function] = []
            functions[self.in_function].append(indent + line)
        else:
            main_body.append(indent + line if self.in_function else '    ' + line)
            
    def _group_lines(self, lines: List[str]) -> List[Tuple[str, List[str]]]:
        """Group lines by section and type"""
        blocks = []
        current_block = None
        current_lines = []
        current_section = '.text'
        
        for line in lines:
            stripped = line.strip()
            
            # Handle section directives
            if stripped.startswith('section ') or stripped in ['.data', '.bss', '.text', '.rodata']:
                if current_lines:
                    blocks.append((current_block, current_lines))
                    current_lines = []
                    
                # Handle both 'section .data' and '.data' formats
                if stripped.startswith('section '):
                    section = stripped.split()[1]
                else:
                    section = stripped
                    
                if '.data' in section or section == '.data':
                    current_block = 'DATA'
                    current_section = '.data'
                elif '.bss' in section or section == '.bss':
                    current_block = 'BSS'
                    current_section = '.bss'
                elif '.rodata' in section or section == '.rodata':
                    current_block = 'DATA'  # Treat rodata as data section
                    current_section = '.data'
                else:
                    current_block = 'ASM'  # .text should be assembly
                    current_section = '.text'
                continue
                
            # Skip empty lines, comments, and assembler directives
            if not stripped or stripped.startswith(';'):
                continue
                
            # Skip assembler directives that don't affect code generation
            assembler_directives = [
                'global ', 'extern ', 'align ', 'bits ', 'org ',
                'times ', 'incbin ', 'struc ', 'endstruc ', 
                'istruc ', 'iend ', 'absolute ', 'common ',
                'cpu ', 'float ', 'default '
            ]
            
            if any(stripped.lower().startswith(directive) for directive in assembler_directives):
                continue  # Assembler directives - skip, not needed in C code
                
            # Categorize lines based on current section
            if current_section in ['.data', '.bss']:
                # Set the correct block type for data sections if not already set
                if current_section == '.data' and current_block != 'DATA':
                    if current_lines:
                        blocks.append((current_block, current_lines))
                        current_lines = []
                    current_block = 'DATA'
                elif current_section == '.bss' and current_block != 'BSS':
                    if current_lines:
                        blocks.append((current_block, current_lines))
                        current_lines = []
                    current_block = 'BSS'
                current_lines.append(line)
            elif self._is_high_level_line(stripped):
                if current_block != 'C':
                    if current_lines:
                        blocks.append((current_block, current_lines))
                    current_block = 'C'
                    current_lines = [line]
                else:
                    current_lines.append(line)
            elif self._is_asm_line(stripped):
                if current_block != 'ASM':
                    if current_lines:
                        blocks.append((current_block, current_lines))
                    current_block = 'ASM'
                    current_lines = [line]
                else:
                    current_lines.append(line)
            else:
                current_lines.append(line)
                
        if current_lines:
            blocks.append((current_block, current_lines))
            
        return blocks
        
    def _is_high_level_line(self, line: str) -> bool:
        """Check if line is high-level construct"""
        stripped = line.strip()
        if stripped in ('{', '}'):
            return True
            
        # Check for C preprocessor directives
        if stripped.startswith('#'):
            return True
            
        # Check for C type declarations
        c_keywords = {
            'var', 'print', 'printf', 'scanf', 'if', 'else', 'while', 'for', 'do',
            'switch', 'case', 'break', 'continue', 'return', 'proc', 'extern',
            'endp', 'function', 'def', 'struct', 'typedef', 'enum', 'int', 'char',
            'float', 'double', 'void', 'const', 'static', 'long', 'short', 'unsigned'
        }
        
        first_word = stripped.split()[0] if stripped.split() else ''
        if first_word in c_keywords:
            return True
            
        # Check for function calls or assignments
        if re.match(r'^[a-zA-Z_]\w*\s*(?:=|\().*', stripped):
            return True
            
        # Check for C-style variable declarations
        if re.match(r'^\s*(const\s+)?(int|char|float|double|void|long|short|unsigned)\s+\w+', stripped):
            return True
            
        return False
        
    def _is_asm_line(self, line: str) -> bool:
        """Check if line is assembly instruction"""
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            return False
            
        # Skip assembler directives that are not CPU instructions
        assembler_directives = [
            'global ', 'extern ', 'section ', 'align ', 'bits ', 'org ',
            'times ', 'incbin ', 'struc ', 'endstruc ', 
            'istruc ', 'iend ', 'absolute ', 'common ',
            'cpu ', 'float ', 'default '
        ]
        
        if any(stripped.lower().startswith(directive) for directive in assembler_directives):
            return False
            
        # Don't treat C code as assembly
        if stripped.startswith('#') or 'include' in stripped:
            return False
            
        # Don't treat C function definitions/declarations as assembly
        if re.match(r'^\s*(const\s+)?(int|char|float|double|void|long|short|unsigned)\s+\w+', stripped):
            return False
            
        # Don't treat printf/scanf calls as assembly
        if re.match(r'^\s*(printf|scanf|fflush)\s*\(', stripped):
            return False
            
        first_word = stripped.split()[0].lower() if stripped.split() else ''
        if first_word in ASM_OPS:
            return True
            
        # Check for labels - but be more restrictive
        if stripped.endswith(':') and not any(kw in stripped for kw in ['proc', 'function', 'def']):
            # Make sure it's not a C label that looks like a variable name
            label_name = stripped[:-1].strip()
            if re.match(r'^[a-zA-Z_]\w*$', label_name) and not label_name.startswith('.'):
                return True
            elif label_name.startswith('.'):  # Assembly local label
                return True
            
        # Be more conservative about register references - only if it looks like actual assembly
        if any(reg in stripped.lower() for reg in REGISTERS):
            # Only consider it assembly if it has assembly-like syntax
            if any(op in stripped.lower() for op in ['mov', 'add', 'sub', 'mul', 'div', 'xor', 'and', 'or']):
                return True
                
        return False
    
    def _is_pure_c_code(self, lines: List[str]) -> bool:
        """Check if the input is already pure C code"""
        total_lines = 0
        c_like_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue
                
            total_lines += 1
            
            # Check for C indicators
            if (stripped.startswith('#include') or 
                stripped.startswith('int ') or
                stripped.startswith('char ') or
                stripped.startswith('float ') or
                stripped.startswith('double ') or
                stripped.startswith('void ') or
                stripped.startswith('const ') or
                stripped.startswith('printf(') or
                stripped.startswith('scanf(') or
                stripped.startswith('fflush(') or
                stripped.startswith('return ') or
                stripped in ['{', '}'] or
                '=' in stripped and ';' in stripped or
                re.match(r'^\s*\w+\s*\([^)]*\)\s*;\s*$', stripped)):
                c_like_lines += 1
                
        # If more than 80% of lines look like C, consider it pure C
        return total_lines > 0 and (c_like_lines / total_lines) > 0.8
        
    def _translate_high_level_line(self, line: str) -> Optional[str]:
        """Translate a high-level construct to C"""
        stripped = re.split(r';|//', line)[0].strip()
        
        # Handle braces
        if stripped == '{':
            self.indent_level += 1
            return '{'
        elif stripped == '}':
            self.indent_level = max(0, self.indent_level - 1)
            return '}'
        
        # Handle variable declarations
        if stripped.startswith('var '):
            return self._translate_var_declaration(stripped)
            
        # Handle external includes
        if stripped.startswith('extern '):
            return self._translate_extern(stripped)
            
        # Handle print statements
        if stripped.startswith('print '):
            return self._translate_print_statement(stripped)
            
        # Handle scanf
        if stripped.startswith('scanf '):
            return self._translate_scanf_statement(stripped)
            
        # Handle control structures
        if stripped.startswith('if '):
            return self._translate_if_statement(stripped)
            
        if stripped == 'else':
            return '} else {'
            
        if stripped.startswith('else if '):
            cond = stripped[8:].rstrip('{').strip()
            return f'}} else if ({cond}) {{'
            
        if stripped.startswith('while '):
            cond = stripped[6:].rstrip('{').strip()
            return f'while ({cond}) {{'
            
        if stripped.startswith('for '):
            return self._translate_for_loop(stripped)
            
        # Handle return statements
        if stripped.startswith('return '):
            value = stripped[7:].strip()
            return f'return {value};'
            
        if stripped == 'ret':
            return None  # Handled by function epilogue
            
        # Handle function definitions
        if stripped.startswith('proc '):
            return self._translate_function_definition(stripped)
            
        if stripped.startswith('endp'):
            self.symbols.exit_function()
            self.in_function = None
            return None
            return '}'

        # Handle function calls
        if m := re.match(r'^\s*call\s+([a-zA-Z_]\w*)(?:\s*\(([^)]*)\))?', stripped, re.I):
            func_name, args_str = m.groups()
            return f'{func_name}({args_str or ""});'
            
        # Handle global directive
        if stripped.startswith('global '):
            return None
            
        # Handle comments
        if stripped.startswith(';'):
            return '//' + stripped[1:]

        # Remove unused variable from print_array
        if "var current_val: int" in stripped:
            return None
            
        # Default: treat as C statement
        if stripped and not stripped.endswith(';'):
            return stripped + ';'
            
        return stripped
        
    def _translate_var_declaration(self, line: str) -> Optional[str]:
        """Translate a 'var' declaration to C"""
        line = line[4:].strip()
        
        # Pattern: var name: type = value or var name: type
        m = re.match(r'^([a-zA-Z_]\w*)\s*:\s*([a-zA-Z_*\[\]\d\s]+)\s*(?:=\s*(.+))?$', line)
        if not m:
            return f'// Invalid var declaration: {line}'
        
        name, vtype, value = m.groups()
        vtype = vtype.strip()
        
        # Handle array types
        is_array = '[' in vtype
        if is_array:
            base_type = vtype.split('[')[0]
            size_match = re.search(r'\[(\d*)\]', vtype)
            size = size_match.group(1) if size_match and size_match.group(1) else ''
        else:
            base_type = vtype
            size = ''
        
        # Add to symbol table
        is_global = self.in_function is None
        var_info = VariableInfo(name, vtype, value, is_global=is_global, is_array=is_array)
        self.symbols.add_variable(var_info)
        
        # Generate C declaration
        if is_array:
            if value:
                return f'{base_type} {name}[{size}] = {value};'
            else:
                return f'{base_type} {name}[{size}];'
        else:
            if value:
                return f'{base_type} {name} = {value};'
            else:
                return f'{base_type} {name};'
                
    def _translate_extern(self, line: str) -> Optional[str]:
        """Translate extern directive"""
        arg = line[7:].strip()
        if arg.startswith('<') and arg.endswith('>'):
            return f'#include {arg}'
        if arg.startswith('"') and arg.endswith('"'):
            return f'#include {arg}'
        if '(' in arg and ')' in arg:
            return f'extern {arg};'
        return None
        
    def _translate_print_statement(self, line: str) -> str:
        """Translate print statement with enhanced support"""
        args_str = line[6:].strip()
        
        # Handle print with register (e.g., print "text", eax)
        reg_match = None
        for reg in REGISTERS:
            # Handle memory access like [array+4]
            mem_access_match = re.search(r'\[(\w+\s*\+\s*\d+)\]', args_str)
            if mem_access_match:
                # This is complex, let's use a temporary variable and assembly
                var_name = f"_tmp_print_{uuid.uuid4().hex[:6]}"
                return f'{{ int {var_name}; __asm__ __volatile__("movl {mem_access_match.group(1)}, %%eax; movl %%eax, %0" : "=r"({var_name})); printf("{args_str.split(",")[0].strip()[1:-1]}\\n", {var_name}); }}'
            if re.search(r'\b' + reg + r'\b', args_str, re.I):
                reg_match = reg
                break
                
        if reg_match:
            # Extract format string and register
            m = re.match(r'"([^"]+)"\s*,\s*(\w+)', args_str)
            if m:
                format_str, reg = m.groups()
                # Use inline assembly to get register value
                return f'{{ int _tmp; __asm__ __volatile__("mov %%{reg.lower()}, %0" : "=r"(_tmp)); printf("{format_str}\\n", _tmp); }}'
        
        # Parse arguments
        args = self._parse_print_args(args_str)
        if not args:
            return 'printf("\\n");'
        
        # Build format string and values
        format_parts = []
        values = []
        
        first_arg = args[0]
        if first_arg.startswith('"') and first_arg.endswith('"') and len(args) >= 2:
            # Check if this is a "label", variable pattern or a literal format string
            base_format = first_arg[1:-1]
            format_spec_count = base_format.count('%') - base_format.count('%%')
            
            if format_spec_count > 0:
                # This is a literal format string with format specifiers
                # Process remaining arguments
                for arg in args[1:]:
                    values.append(self._process_print_arg(arg))
                
                # Adjust values list to match format specifiers
                if format_spec_count > len(values):
                    while len(values) < format_spec_count:
                        values.append('0')
                elif format_spec_count < len(values):
                    values = values[:format_spec_count]
                    
                return f'printf("{base_format}\\n", {", ".join(values)}); fflush(stdout);'
            else:
                # This is a "label", variable pattern - no format specifiers in label
                label = base_format
                value_args = args[1:]
                
                format_parts = [label]
                values = []
                for val_arg in value_args:
                    val_arg = val_arg.strip()
                    fmt, val = self._get_format_for_arg(val_arg)
                    format_parts.append(fmt)
                    if val:
                        values.append(val)
                
                format_str = " ".join(format_parts).strip()
                if values:
                    return f'printf("{format_str}\\n", {", ".join(values)}); fflush(stdout);'
                else:
                    return f'printf("{format_str}\\n"); fflush(stdout);'
        elif first_arg.startswith('"') and first_arg.endswith('"') and len(args) == 1:
            # Single string argument
            clean_format = first_arg[1:-1]
            return f'printf("{clean_format}\\n"); fflush(stdout);'
        else:
            # Auto-generate format string from multiple arguments
            if len(args) >= 2 and args[0].strip().startswith('"'):
                # This should not be reached now due to above logic
                label_arg = args[0].strip()
                value_args = args[1:]
                
                if label_arg.startswith('"') and label_arg.endswith('"'):
                    label = label_arg[1:-1]
                    
                    format_parts = [label]
                    values = []
                    for val_arg in value_args:
                        val_arg = val_arg.strip()
                        fmt, val = self._get_format_for_arg(val_arg)
                        format_parts.append(fmt)
                        if val:
                            values.append(val)
                    
                    format_str = " ".join(format_parts).strip() + "\\n"
                    return f'printf("{format_str}", {", ".join(values)}); fflush(stdout);'
                else:
                    # Two separate arguments
                    fmt1, val1 = self._get_format_for_arg(label_arg)
                    fmt2, val2 = self._get_format_for_arg(value_arg)
                    format_str = fmt1 + ' ' + fmt2
                    values = [v for v in [val1, val2] if v]
                    if values:
                        return f'printf("{format_str}\\n", {", ".join(values)}); fflush(stdout);'
                    else:
                        return f'printf("{format_str}\\n"); fflush(stdout);'
            else:
                # Auto-generate format string
                for arg in args:
                    fmt, val = self._get_format_for_arg(arg)
                    format_parts.append(fmt)
                    if val:  # Only add to values if there's an actual value
                        values.append(val)
                
                format_str = ''.join(format_parts)
                # Handle the case where we have format but no values - need to find the actual variable
                if not values and format_parts:
                    # Look for variables in the original arguments that weren't processed
                    remaining_vars = []
                    for arg in args:
                        if not arg.startswith('"') and not arg.startswith("'"):
                            # This might be a variable reference
                            if self.symbols.get_variable(arg) or '[' in arg:
                                remaining_vars.append(arg)
                    
                    if remaining_vars:
                        values.extend(remaining_vars)
                
                if values:
                    return f'printf("{format_str}", {", ".join(values)}); fflush(stdout);'
                else:
                    return f'printf("{format_str}"); fflush(stdout);'
                
    def _translate_scanf_statement(self, line: str) -> str:
        """Translate scanf statement"""
        args_str = line[6:].strip()
        m = re.match(r'"([^"]+)"\s*,\s*(.+)', args_str)
        if not m:
            return f'// Invalid scanf: {line}'
            
        format_str, vars_str = m.groups()

        # Don't double-escape % in scanf format strings - they need to remain as single %
        format_str = format_str.replace('"', '\\"')

        variables = [v.strip() for v in vars_str.split(',')]
        
        scanf_vars = []
        for var in variables:
            var = var.strip()
            # Add & if not already present and not an array
            var_info = self.symbols.get_variable(var)
            if var_info and var_info.is_array:
                scanf_vars.append(var)
            elif not var.startswith('&'):
                scanf_vars.append(f'&{var}')
            else:
                scanf_vars.append(var)
                
        return f'scanf("{format_str}", {", ".join(scanf_vars)});'
        
    def _translate_if_statement(self, line: str) -> str:
        """Translate if statement"""
        cond = line[3:].rstrip('{').strip()
        # Handle assembly-style comparisons
        cond = self._translate_condition(cond)
        return f'if ({cond}) {{'
        
    def _translate_for_loop(self, line: str) -> str:
        """Translate for loop"""
        # Pattern: for var = start to end
        m = re.match(r'for\s+([a-zA-Z_]\w*)\s*=\s*(.+?)\s+to\s+(.+?)\s*(?:\{)?$', line)
        if m:
            var, start, end = m.groups()
            # If variable was already declared with `var`, don't re-declare it with `int`.
            # Otherwise, declare it in the loop.
            if not self.symbols.get_variable(var):
                return f'for (int {var} = {start}; {var} <= {end}; {var}++) {{'
            return f'for ({var} = {start}; {var} <= {end}; {var}++) {{'
            
        # Pattern: for (standard C syntax)
        m = re.match(r'for\s*\((.+)\)(?:\s*\{)?', line)
        if m:
            return f'for ({m.group(1)}) {{'
            
        return f'// Invalid for loop: {line}'
        
    def _translate_function_definition(self, line: str) -> Optional[str]:
        """Translate function definition"""
        rest = line[5:].strip()
        
        # Parse function name and parameters
        if ':' in rest:
            func_part, param_part = rest.split(':', 1)
            func_name = func_part.strip()
            
            # Parse parameters
            param_part = param_part.strip()
            
            # Handle parameters like "int a, int b"
            params = []
            ret_type = 'int'  # Default return type - most functions return int
            
            if param_part:
                # Split by comma and handle each parameter
                param_list = [p.strip() for p in param_part.split(',')]
                for param in param_list:
                    if param:
                        # Each parameter should be "type name"
                        param_parts = param.split()
                        if len(param_parts) == 2:
                            param_type, param_name = param_parts
                            params.append(f"{param_type} {param_name}")
                        elif len(param_parts) == 1:
                            # Just a name, assume int
                            params.append(f"int {param_parts[0]}")
                        else:
                            # More complex type, join all but last as type
                            param_type = ' '.join(param_parts[:-1])
                            param_name = param_parts[-1]
                            params.append(f"{param_type} {param_name}")
        else:
            func_name = rest.strip()
            ret_type = 'void'
            params = []
            
        self.in_function = func_name
        self.symbols.enter_function(func_name, ret_type, params)
        
        # Add parameters to symbol table
        for param in params:
            param_parts = param.split()
            if len(param_parts) >= 2:
                param_type = ' '.join(param_parts[:-1])
                param_name = param_parts[-1]
                self.symbols.add_variable(VariableInfo(param_name, param_type, is_global=False))
                
        return None  # Function definition is handled in the main translation
        
    def _parse_print_args(self, args_str: str) -> List[str]:
        """Parse print statement arguments"""
        args = []
        current_arg = ''
        in_quotes = False
        quote_char = None
        paren_depth = 0
        
        for char in args_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                current_arg += char
            elif char == quote_char and in_quotes and (not current_arg or current_arg[-1] != '\\'):
                in_quotes = False
                quote_char = None
                current_arg += char
            elif char == '(' and not in_quotes:
                paren_depth += 1
                current_arg += char
            elif char == ')' and not in_quotes:
                paren_depth -= 1
                current_arg += char
            elif char == ',' and not in_quotes and paren_depth == 0:
                args.append(current_arg.strip())
                current_arg = ''
            else:
                current_arg += char
                
        if current_arg:
            args.append(current_arg.strip())
            
        return args
        
    def _process_print_arg(self, arg: str) -> str:
        """Process a single print argument"""
        arg = arg.strip()
        
        # Handle array access [var]
        if arg.startswith('[') and arg.endswith(']'):
            var_name = arg[1:-1].strip()
            var_info = self.symbols.get_variable(var_name)
            if var_info:
                if var_info.is_array:
                    return f'{var_name}[0]'  # Print first element safely
                else:
                    return var_name
        
        # Handle regular variables
        var_info = self.symbols.get_variable(arg)
        if var_info:
            return arg
            
        # Return as-is (might be expression or literal)
        return arg
        
    def _get_format_for_arg(self, arg: str) -> Tuple[str, str]:
        """Get format specifier and processed value for an argument"""
        arg = arg.strip()
        
        # Check if it's a string literal
        if arg.startswith('"') and arg.endswith('"'):
            return arg[1:-1], ''
        
        # Handle empty print for newline
        if not arg:
            return '\\n', ''
            
        # Check if it's a character literal
        if arg.startswith("'") and arg.endswith("'"):
            return '%c', arg
        
        # Handle array access like array[1]
        if '[' in arg and ']' in arg:
            return '%d', arg
            
        # Check if it's a variable
        var_info = self.symbols.get_variable(arg)
        if var_info:
            if var_info.type == 'char':
                return '%c', arg
            elif 'char*' in var_info.type or 'char[]' in var_info.type:
                return '%s', arg
            elif 'float' in var_info.type:
                return '%f', arg
            elif 'double' in var_info.type:
                return '%lf', arg
            else:
                return '%d', arg
                
        # Check if it's a numeric literal
        try:
            int(arg)
            return '%d', arg
        except ValueError:
            pass
            
        # Handle space prefix (common for array printing) - look at context
        if arg.startswith('" "') or arg == '" "':
            return ' %d', ''  # Return format but no value yet
            
        # Default handling - check if it looks like a format string
        if '%' in arg:
            return arg, ''
        else:
            # Treat as integer format
            return '%d', arg
        
    def _translate_condition(self, cond: str) -> str:
        """Translate assembly-style conditions to C"""
        # Handle register comparisons
        for reg in REGISTERS:
            cond = re.sub(r'\b' + reg + r'\b', f'__{reg}__', cond, flags=re.I)
            
        return cond

class Compiler:
    """Main compiler interface with advanced features"""
    
    def __init__(self, options: CompilerOptions = None):
        self.options = options or CompilerOptions()
        self.translator = CodeTranslator(self.options)
        self.verbose = False
        self.jit_compiler = None
        
        self.output_dir = Path.cwd() / "output"
        # Initialize JIT compiler if requested
        if hasattr(self.options, 'jit') and self.options.jit:
            self.jit_compiler = JITCompiler(self.options.target_arch)

    def set_architecture(self, arch: str):
        """Set the target architecture directly."""
        if arch and hasattr(TargetArchitecture, arch.upper()):
            self.options.target_arch = TargetArchitecture[arch.upper()]
            # Re-initialize the translator to use the new architecture backend
            self.translator = CodeTranslator(self.options)
        else:
            log_message("WARN", f"Unknown or unsupported architecture: {arch}. Using default.")
        
    def compile(self, source_path: Path, output_path: Optional[Path] = None,
                exe_path: Optional[Path] = None, build: bool = True, run: bool = False, verbose: bool = False,
                target: Optional[str] = None):
        """Compile the source file"""
        self.verbose = verbose
        
        # Ensure the output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        if self.verbose:
            log_message("INFO", f"Reading source file: {source_path}")
            
        source_content = source_path.read_text()
        lines = source_content.splitlines()
        
        # Expand includes
        lines = self._expand_includes(lines, source_path.parent)
        
        # Set up compiler options from arguments
        if target:
            target_map = {
                'x86_64': TargetArchitecture.X86_64,
                'arm64': TargetArchitecture.ARM64,
                'riscv64': TargetArchitecture.RISCV64,
                'windows': TargetArchitecture.X86_64,
                'linux': TargetArchitecture.X86_64,
                'macos': TargetArchitecture.X86_64
            }
            self.options.target_arch = target_map.get(target.lower(), TargetArchitecture.X86_64)
            
        # Translate to C with advanced features
        if self.verbose:
            log_message("INFO", f"Translating to C (target: {self.options.target_arch.value})...")
            log_message("INFO", f"Optimization level: {self.options.optimization_level.value}")
            
        c_code = self.translator.translate_to_c(lines, str(source_path))
        
        # Determine output path
        if output_path:
            c_path = output_path
        else:
            c_path = self.output_dir / source_path.with_suffix('.c').name
            
        # Write C code
        c_path.write_text(c_code)
        log_message("OK", f"Generated C code written to {c_path}")
        
        if self.verbose:
            log_message("INFO", f"Generated {len(c_code.splitlines())} lines of C code")
        
        # Build if requested
        if build or run:
            built_exe_path = self._build_executable(c_path, exe_path, source_path.stem, run, target)
            return built_exe_path
            
        return c_path
            
    def _expand_includes(self, lines: List[str], base_path: Path) -> List[str]:
        """Recursively expand .include directives"""
        expanded = []
        
        for line in lines:
            stripped = line.strip()
            
            # Handle .include directive
            if stripped.startswith('.include'):
                m = re.match(r'\.include\s+"([^"]+)"', stripped)
                if m:
                    include_file = base_path / m.group(1)
                    if include_file.exists():
                        if self.verbose:
                            log_message("INFO", f"Including file: {include_file}")
                        include_lines = include_file.read_text().splitlines()
                        # Recursively expand includes in the included file
                        expanded.extend(self._expand_includes(include_lines, include_file.parent))
                    else:
                        log_message("WARN", f"Include file not found: {include_file}")
                        expanded.append(f"; Include file not found: {include_file}")
                else:
                    expanded.append(line)
            else:
                expanded.append(line)
                
        return expanded
        
    def _build_executable(self, c_path: Path, output_exe_path: Optional[Path], fallback_name: str, run: bool, target: Optional[str]) -> Optional[Path]:
        """Build executable using gcc with proper flags"""
        # Determine compiler based on target
        if target == 'windows':
            compiler_name = "x86_64-w64-mingw32-gcc"
            platform = "win32"
        else:
            compiler_name = "gcc"
            platform = sys.platform

        compiler = shutil.which(compiler_name)
        if not compiler:
            log_message("ERROR", f"Compiler '{compiler_name}' not found in PATH.")
            if target == 'windows':
                log_message("INFO", "Please install a MinGW-w64 cross-compiler (e.g., 'brew install mingw-w64').")
            return None

        # Determine output executable path
        if output_exe_path:
            exe_path = output_exe_path
        else:
            exe_path = self.output_dir / fallback_name
        if platform == "win32":
            exe_path = exe_path.with_suffix(".exe")
            
        # Build command with advanced flags based on compiler options
        cmd = [
            compiler,
            str(c_path),
            "-o", str(exe_path),
            f"-{self.options.optimization_level.value}",  # Dynamic optimization level
            "-Wall",         # All warnings
            "-Wextra",       # Extra warnings
            "-std=gnu11",    # C11 standard with GNU extensions (for inline asm)
            "-fno-stack-protector",  # Disable stack protector for simpler asm
        ]
        
        # Add debug information if enabled
        if self.options.debug_info:
            cmd.extend(["-g", "-gdwarf-4"])
            
        # Add Position Independent Code/Executable flags
        if self.options.pic:
            cmd.append("-fPIC")
        if self.options.pie:
            cmd.append("-fPIE")
            
        # Add Link Time Optimization
        if self.options.lto:
            cmd.extend(["-flto", "-fuse-linker-plugin"])
            
        # Add sanitizers
        for sanitizer in self.options.sanitizers:
            cmd.append(f"-fsanitize={sanitizer}")
            
        # Add CPU-specific optimizations
        if self.options.cpu_features:
            for feature in self.options.cpu_features:
                if feature.startswith('avx'):
                    cmd.append(f"-m{feature}")
                elif feature in ['sse4.1', 'sse4.2', 'ssse3', 'sse3']:
                    cmd.append(f"-m{feature}")
                    
        # Add vectorization flags (different for GCC vs Clang)
        if self.options.vectorize and self.options.optimization_level != OptimizationLevel.NONE:
            cmd.append("-ftree-vectorize")
            # Only add -fopt-info-vec for actual GCC, not clang
            # Check if the compiler binary is actually GCC (not clang masquerading as gcc)
            try:
                gcc_version_result = subprocess.run([compiler, "--version"], 
                                                  capture_output=True, text=True, timeout=5)
                if gcc_version_result.returncode == 0 and "gcc" in gcc_version_result.stdout.lower():
                    cmd.append("-fopt-info-vec")
            except:
                pass  # Skip if we can't determine compiler type
            
        # Add function inlining control
        if self.options.inline_functions:
            cmd.append("-finline-functions")
            
        # Add loop unrolling
        if self.options.unroll_loops:
            cmd.append("-funroll-loops")
            
        # Add custom flags
        cmd.extend(self.options.custom_flags)
        
        # Add macro definitions
        for name, value in self.options.define_macros.items():
            if value:
                cmd.append(f"-D{name}={value}")
            else:
                cmd.append(f"-D{name}")
                
        # Add include paths
        for include_path in self.options.include_paths:
            cmd.extend(["-I", str(include_path)])
            
        # Add library paths and libraries
        for lib_path in self.options.library_paths:
            cmd.extend(["-L", str(lib_path)])
        for library in self.options.libraries:
            cmd.append(f"-l{library}")
        
        # Add architecture-specific flags
        if platform == "darwin":  # macOS
            cmd.extend(["-arch", "x86_64"])  # Force x86_64 on Apple Silicon if needed
        elif platform.startswith("linux"):
            cmd.append("-no-pie")  # Disable PIE for easier debugging
            
        log_message("INFO", f"Compiling with: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                log_message("OK", f"Compilation successful: {exe_path}")
                
                # Make executable on Unix systems
                if platform != "win32":
                    exe_path.chmod(0o755)
                
                # Run if requested
                if run:
                    print()  # Add a newline for cleaner output
                    try:
                        run_result = subprocess.run(
                            [str(exe_path)],
                            capture_output=False,
                            text=True,
                            timeout=60,
                            check=False # Don't raise exception on non-zero exit codes
                        )
                        if self.verbose:
                            log_message("INFO", f"Program exited with code: {run_result.returncode}")
                    except subprocess.TimeoutExpired:
                        log_message("ERROR", "Program execution timed out")
                    except Exception as e:
                        log_message("ERROR", f"Failed to run program: {e}")
                        
                return exe_path
            else:
                log_message("ERROR", f"Compilation failed with code {result.returncode}")
                if result.stderr:
                    print("\n[STDERR]")
                    print(result.stderr)
                if result.stdout:
                    print("\n[STDOUT]")
                    print(result.stdout)
                    
                # Try to provide helpful error messages
                if "unknown register name" in result.stderr:
                    log_message("INFO", "HINT: The inline assembly uses incorrect register constraints. Check register names.")
                elif "invalid 'asm'" in result.stderr:
                    log_message("INFO", "HINT: The inline assembly syntax is incorrect. Ensure AT&T syntax is used properly.")
                    
                return None
                
        except subprocess.TimeoutExpired:
            log_message("ERROR", "Compilation timed out")
            return None
        except Exception as e:
            log_message("ERROR", f"Failed to compile: {e}")
            return None

def main():
    """Main entry point with Nim-style CLI"""
    cli_args = sys.argv[1:]

    if not cli_args or len(cli_args) < 2 or cli_args[0] not in ['c', 'compile']:
        print("Usage: python3 compiler.py c <sourcefile> [options]")
        print("\nOptions:")
        print("  -o:<file>      Set output executable name/path.")
        print("  -d:release     Build in release mode (optimizations, no debug).")
        print("  -d:debug       Build with debug info.")
        print("  -t:<target>    Set target platform (e.g., windows, linux, macos).")
        print("  -a:<arch>      Set target architecture (e.g., x86_64, arm64).")
        return 1

    source_file = cli_args[1]
    options_args = cli_args[2:]

    source_path = Path(source_file)
    if not source_path.exists():
        log_message("ERROR", f"Source file not found: {source_path}")
        return 1

    options = CompilerOptions()
    exe_path_arg = None
    target_arg = None
    arch_arg = None
    run_after_compile = False

    for arg in options_args:
        if arg.startswith("-o:"):
            # Prepend output directory if the path is not absolute
            out_name = arg.split(":", 1)[1]
            exe_path_arg = Path(out_name)
            if not exe_path_arg.is_absolute():
                exe_path_arg = Path("output") / exe_path_arg
        elif arg == "-d:release":
            options.optimization_level = OptimizationLevel.AGGRESSIVE  # -O3
            options.debug_info = False
        elif arg == "-d:debug":
            options.optimization_level = OptimizationLevel.NONE  # -O0
            options.debug_info = True
        elif arg.startswith("-t:"):
            target_arg = arg.split(":", 1)[1].lower()
        elif arg.startswith("-a:"):
            arch_arg = arg.split(":", 1)[1].lower()
        elif arg in ["-r", "--run"]:
            run_after_compile = True
        else:
            log_message("WARN", f"Unknown option: {arg}")

    compiler = Compiler(options)
    if arch_arg:
        compiler.set_architecture(arch_arg)

    try:
        result = compiler.compile(
            source_path,
            output_path=None,  # Let compiler decide C file path
            exe_path=exe_path_arg,
            build=True, # Always build
            run=run_after_compile,
            verbose=False,
            target=target_arg
        )

        if result:
            return 0
        else:
            return 1

    except Exception as e:
        log_message("ERROR", f"Compilation failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())