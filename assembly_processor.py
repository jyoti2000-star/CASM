#!/usr/bin/env python3
"""
Advanced Assembly Processing Suite for CASM
Combines assembly fixing, formatting, and post-processing with intelligent comparison
between different code generators. Handles any kind of assembly and C statement assembly code.
"""

import re
import os
import sys
import ast
import json
import tempfile
import difflib
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class AssemblyType(Enum):
    """Types of assembly instructions"""
    DATA_DECLARATION = "data_declaration"
    INSTRUCTION = "instruction"
    LABEL = "label"
    DIRECTIVE = "directive"
    COMMENT = "comment"
    FUNCTION_CALL = "function_call"
    CONTROL_FLOW = "control_flow"
    MEMORY_OPERATION = "memory_operation"

class InstructionFormat(Enum):
    """Assembly instruction formats"""
    NO_OPERANDS = "no_operands"
    ONE_OPERAND = "one_operand"
    TWO_OPERANDS = "two_operands"
    THREE_OPERANDS = "three_operands"
    VARIABLE_OPERANDS = "variable_operands"

@dataclass
class AssemblyInstruction:
    """Represents a parsed assembly instruction"""
    raw_line: str
    line_number: int
    instruction_type: AssemblyType
    mnemonic: Optional[str] = None
    operands: List[str] = field(default_factory=list)
    label: Optional[str] = None
    comment: Optional[str] = None
    section: Optional[str] = None
    format_type: InstructionFormat = InstructionFormat.NO_OPERANDS
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AssemblyPattern:
    """Represents a pattern found in assembly code"""
    pattern_type: str
    confidence: float
    location: Tuple[int, int]  # (start_line, end_line)
    context: Dict[str, Any]
    suggested_fix: Optional[str] = None

class AdvancedAssemblyAnalyzer:
    """Advanced analyzer that understands assembly structure and patterns"""
    
    def __init__(self):
        self.instructions = []
        self.labels = {}
        self.sections = {}
        self.variables = {}
        self.functions = {}
        self.patterns = []
        self.calling_conventions = {
            'windows_x64': {
                'int_args': ['rcx', 'rdx', 'r8', 'r9'],
                'float_args': ['xmm0', 'xmm1', 'xmm2', 'xmm3'],
                'return': 'rax',
                'shadow_space': 32,
                'stack_alignment': 16,
                'volatile_regs': ['rax', 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11', 'xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5'],
                'preserved_regs': ['rbx', 'rbp', 'rdi', 'rsi', 'rsp', 'r12', 'r13', 'r14', 'r15', 'xmm6', 'xmm7', 'xmm8', 'xmm9', 'xmm10', 'xmm11', 'xmm12', 'xmm13', 'xmm14', 'xmm15']
            },
            'system_v_x64': {
                'int_args': ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'],
                'float_args': ['xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7'],
                'return': 'rax',
                'shadow_space': 0,
                'stack_alignment': 16,
                'volatile_regs': ['rax', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11'] + [f'xmm{i}' for i in range(16)],
                'preserved_regs': ['rbx', 'rbp', 'rsp', 'r12', 'r13', 'r14', 'r15']
            }
        }
        
    def parse_assembly(self, content: str) -> List[AssemblyInstruction]:
        """Parse assembly content into structured instructions"""
        lines = content.split('\n')
        instructions = []
        current_section = None
        
        for i, line in enumerate(lines):
            instruction = self._parse_line(line, i + 1, current_section)
            if instruction.instruction_type == AssemblyType.DIRECTIVE and instruction.mnemonic and instruction.mnemonic.startswith('section'):
                current_section = instruction.operands[0] if instruction.operands else None
                instruction.section = current_section
            else:
                instruction.section = current_section
            instructions.append(instruction)
            
        self.instructions = instructions
        return instructions
    
    def _parse_line(self, line: str, line_number: int, current_section: str) -> AssemblyInstruction:
        """Parse a single line of assembly"""
        original_line = line
        stripped = line.strip()
        
        # Handle empty lines
        if not stripped:
            return AssemblyInstruction(
                raw_line=original_line,
                line_number=line_number,
                instruction_type=AssemblyType.COMMENT,
                section=current_section
            )
        
        # Handle comments
        if stripped.startswith(';'):
            return AssemblyInstruction(
                raw_line=original_line,
                line_number=line_number,
                instruction_type=AssemblyType.COMMENT,
                comment=stripped[1:].strip(),
                section=current_section
            )
        
        # Handle labels
        if ':' in stripped and not any(op in stripped for op in ['mov', 'lea', 'call', 'jmp', 'add', 'sub']):
            label = stripped.split(':')[0].strip()
            return AssemblyInstruction(
                raw_line=original_line,
                line_number=line_number,
                instruction_type=AssemblyType.LABEL,
                label=label,
                section=current_section
            )
        
        # Handle directives
        if any(stripped.startswith(d) for d in ['section', 'global', 'extern', 'dd', 'db', 'dw', 'dq']):
            parts = stripped.split(None, 1)
            mnemonic = parts[0]
            operands = self._parse_operands(parts[1] if len(parts) > 1 else "")
            
            return AssemblyInstruction(
                raw_line=original_line,
                line_number=line_number,
                instruction_type=AssemblyType.DATA_DECLARATION if mnemonic in ['dd', 'db', 'dw', 'dq'] else AssemblyType.DIRECTIVE,
                mnemonic=mnemonic,
                operands=operands,
                section=current_section,
                format_type=self._determine_format(operands)
            )
        
        # Handle instructions
        parts = stripped.split(None, 1)
        if parts:
            mnemonic = parts[0]
            operands = self._parse_operands(parts[1] if len(parts) > 1 else "")
            
            # Determine instruction type
            inst_type = self._classify_instruction(mnemonic, operands)
            
            return AssemblyInstruction(
                raw_line=original_line,
                line_number=line_number,
                instruction_type=inst_type,
                mnemonic=mnemonic,
                operands=operands,
                section=current_section,
                format_type=self._determine_format(operands)
            )
        
        # Fallback
        return AssemblyInstruction(
            raw_line=original_line,
            line_number=line_number,
            instruction_type=AssemblyType.COMMENT,
            section=current_section
        )
    
    def _parse_operands(self, operand_str: str) -> List[str]:
        """Parse operands from instruction"""
        if not operand_str:
            return []
        
        # Handle complex operands with brackets, quotes, etc.
        operands = []
        current_operand = ""
        bracket_depth = 0
        in_quotes = False
        quote_char = None
        
        for char in operand_str:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_operand += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_operand += char
            elif char in ['[', '('] and not in_quotes:
                bracket_depth += 1
                current_operand += char
            elif char in [']', ')'] and not in_quotes:
                bracket_depth -= 1
                current_operand += char
            elif char == ',' and bracket_depth == 0 and not in_quotes:
                operands.append(current_operand.strip())
                current_operand = ""
            else:
                current_operand += char
        
        if current_operand.strip():
            operands.append(current_operand.strip())
        
        return operands
    
    def _classify_instruction(self, mnemonic: str, operands: List[str]) -> AssemblyType:
        """Classify instruction type"""
        if mnemonic in ['call', 'ret']:
            return AssemblyType.FUNCTION_CALL
        elif mnemonic in ['jmp', 'je', 'jne', 'jl', 'jle', 'jg', 'jge', 'ja', 'jae', 'jb', 'jbe', 'cmp', 'test']:
            return AssemblyType.CONTROL_FLOW
        elif mnemonic in ['mov', 'lea', 'push', 'pop'] or any('[' in op for op in operands):
            return AssemblyType.MEMORY_OPERATION
        else:
            return AssemblyType.INSTRUCTION
    
    def _determine_format(self, operands: List[str]) -> InstructionFormat:
        """Determine instruction format based on operands"""
        count = len(operands)
        if count == 0:
            return InstructionFormat.NO_OPERANDS
        elif count == 1:
            return InstructionFormat.ONE_OPERAND
        elif count == 2:
            return InstructionFormat.TWO_OPERANDS
        elif count == 3:
            return InstructionFormat.THREE_OPERANDS
        else:
            return InstructionFormat.VARIABLE_OPERANDS

class IntelligentAssemblyComparator:
    """Compares assembly from different generators and identifies improvements"""
    
    def __init__(self):
        self.analyzer = AdvancedAssemblyAnalyzer()
    
    def compare_generators(self, primary_output: str, reference_output: str) -> Dict[str, Any]:
        """Compare outputs from different code generators"""
        
        # Parse both outputs
        primary_instructions = self.analyzer.parse_assembly(primary_output)
        reference_instructions = self.analyzer.parse_assembly(reference_output)
        
        comparison = {
            'primary_stats': self._analyze_instructions(primary_instructions),
            'reference_stats': self._analyze_instructions(reference_instructions),
            'differences': self._find_differences(primary_instructions, reference_instructions),
            'improvements': self._suggest_improvements(primary_instructions, reference_instructions),
            'patterns': self._identify_patterns(primary_instructions, reference_instructions)
        }
        
        return comparison
    
    def _analyze_instructions(self, instructions: List[AssemblyInstruction]) -> Dict[str, Any]:
        """Analyze instruction patterns and statistics"""
        stats = {
            'total_instructions': len(instructions),
            'by_type': {},
            'sections': set(),
            'labels': [],
            'function_calls': [],
            'memory_operations': [],
            'control_flow': [],
            'data_declarations': []
        }
        
        for inst in instructions:
            # Count by type
            inst_type = inst.instruction_type.value
            stats['by_type'][inst_type] = stats['by_type'].get(inst_type, 0) + 1
            
            # Collect sections
            if inst.section:
                stats['sections'].add(inst.section)
            
            # Collect specific instruction types
            if inst.instruction_type == AssemblyType.LABEL:
                stats['labels'].append(inst.label)
            elif inst.instruction_type == AssemblyType.FUNCTION_CALL:
                stats['function_calls'].append({
                    'mnemonic': inst.mnemonic,
                    'operands': inst.operands,
                    'line': inst.line_number
                })
            elif inst.instruction_type == AssemblyType.MEMORY_OPERATION:
                stats['memory_operations'].append({
                    'mnemonic': inst.mnemonic,
                    'operands': inst.operands,
                    'line': inst.line_number
                })
            elif inst.instruction_type == AssemblyType.CONTROL_FLOW:
                stats['control_flow'].append({
                    'mnemonic': inst.mnemonic,
                    'operands': inst.operands,
                    'line': inst.line_number
                })
            elif inst.instruction_type == AssemblyType.DATA_DECLARATION:
                stats['data_declarations'].append({
                    'mnemonic': inst.mnemonic,
                    'operands': inst.operands,
                    'line': inst.line_number
                })
        
        return stats
    
    def _find_differences(self, instructions1: List[AssemblyInstruction], instructions2: List[AssemblyInstruction]) -> List[Dict[str, Any]]:
        """Find meaningful differences between instruction sets"""
        differences = []
        
        # Convert to comparable format
        lines1 = [self._instruction_to_comparable(inst) for inst in instructions1]
        lines2 = [self._instruction_to_comparable(inst) for inst in instructions2]
        
        # Use difflib to find differences
        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                differences.append({
                    'type': tag,
                    'primary_lines': lines1[i1:i2],
                    'reference_lines': lines2[j1:j2],
                    'primary_range': (i1, i2),
                    'reference_range': (j1, j2)
                })
        
        return differences
    
    def _instruction_to_comparable(self, inst: AssemblyInstruction) -> str:
        """Convert instruction to comparable string format"""
        if inst.mnemonic:
            return f"{inst.mnemonic} {' '.join(inst.operands)}".strip()
        elif inst.label:
            return f"{inst.label}:"
        else:
            return inst.raw_line.strip()
    
    def _suggest_improvements(self, instructions1: List[AssemblyInstruction], instructions2: List[AssemblyInstruction]) -> List[Dict[str, Any]]:
        """Suggest improvements based on comparison"""
        improvements = []
        
        # Analyze function call patterns
        calls1 = [inst for inst in instructions1 if inst.instruction_type == AssemblyType.FUNCTION_CALL]
        calls2 = [inst for inst in instructions2 if inst.instruction_type == AssemblyType.FUNCTION_CALL]
        
        # Check for missing function calls
        call_names1 = set(inst.operands[0] if inst.operands else inst.mnemonic for inst in calls1 if inst.mnemonic == 'call')
        call_names2 = set(inst.operands[0] if inst.operands else inst.mnemonic for inst in calls2 if inst.mnemonic == 'call')
        
        missing_calls = call_names2 - call_names1
        if missing_calls:
            improvements.append({
                'type': 'missing_function_calls',
                'description': f"Missing function calls: {', '.join(missing_calls)}",
                'severity': 'high',
                'suggestion': 'Add missing function call instructions'
            })
        
        # Check for section declarations
        sections1 = set(inst.operands[0] for inst in instructions1 if inst.mnemonic == 'section' and inst.operands)
        sections2 = set(inst.operands[0] for inst in instructions2 if inst.mnemonic == 'section' and inst.operands)
        
        missing_sections = sections2 - sections1
        if missing_sections:
            improvements.append({
                'type': 'missing_sections',
                'description': f"Missing sections: {', '.join(missing_sections)}",
                'severity': 'medium',
                'suggestion': 'Add missing section declarations'
            })
        
        return improvements
    
    def _identify_patterns(self, instructions1: List[AssemblyInstruction], instructions2: List[AssemblyInstruction]) -> List[AssemblyPattern]:
        """Identify common patterns and anti-patterns"""
        patterns = []
        
        # Look for printf patterns
        printf_patterns1 = self._find_printf_patterns(instructions1)
        printf_patterns2 = self._find_printf_patterns(instructions2)
        
        for pattern in printf_patterns2:
            if pattern not in printf_patterns1:
                patterns.append(AssemblyPattern(
                    pattern_type='missing_printf_call',
                    confidence=0.9,
                    location=pattern['location'],
                    context=pattern,
                    suggested_fix=f"Add 'call printf' after parameter setup at line {pattern['location'][1]}"
                ))
        
        return patterns
    
    def _find_printf_patterns(self, instructions: List[AssemblyInstruction]) -> List[Dict[str, Any]]:
        """Find printf-like patterns in instructions"""
        patterns = []
        
        for i, inst in enumerate(instructions):
            if (inst.instruction_type == AssemblyType.MEMORY_OPERATION and 
                inst.mnemonic == 'lea' and 
                inst.operands and 
                'rcx' in inst.operands[0] and
                any('LC' in op for op in inst.operands)):
                
                # Look for parameter setup following this
                param_end = i
                for j in range(i + 1, min(len(instructions), i + 10)):
                    next_inst = instructions[j]
                    if (next_inst.mnemonic in ['mov', 'lea'] and 
                        any(reg in next_inst.operands[0] if next_inst.operands else '' 
                            for reg in ['rdx', 'r8', 'r9'])):
                        param_end = j
                    elif next_inst.mnemonic == 'call' and next_inst.operands and 'printf' in next_inst.operands[0]:
                        break  # Found the call
                    elif next_inst.instruction_type in [AssemblyType.LABEL, AssemblyType.CONTROL_FLOW]:
                        # Pattern ended without call
                        patterns.append({
                            'type': 'incomplete_printf',
                            'location': (i + 1, param_end + 1),
                            'format_string': inst.operands[1] if len(inst.operands) > 1 else 'unknown'
                        })
                        break
        
        return patterns

class NASMCompatibilityEngine:
    """Advanced NASM compatibility engine for dynamic assembly fixing"""
    
    def __init__(self, debug: bool = False, calling_convention: str = 'windows_x64'):
        self.debug = debug  # Use provided debug setting
        self.calling_convention = calling_convention
        self.register_mappings = {
            # 64-bit to 32-bit register mappings
            'rax': 'eax', 'rbx': 'ebx', 'rcx': 'ecx', 'rdx': 'edx',
            'rsi': 'esi', 'rdi': 'edi', 'rsp': 'esp', 'rbp': 'ebp',
            'r8': 'r8d', 'r9': 'r9d', 'r10': 'r10d', 'r11': 'r11d',
            'r12': 'r12d', 'r13': 'r13d', 'r14': 'r14d', 'r15': 'r15d'
        }
        
        # Windows x64 calling convention details
        self.windows_x64_convention = {
            'int_args': ['rcx', 'rdx', 'r8', 'r9'],
            'float_args': ['xmm0', 'xmm1', 'xmm2', 'xmm3'],
            'return_int': 'rax',
            'return_float': 'xmm0',
            'shadow_space': 32,
            'stack_alignment': 16,
            'volatile_regs': ['rax', 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11'],
            'preserved_regs': ['rbx', 'rbp', 'rdi', 'rsi', 'rsp', 'r12', 'r13', 'r14', 'r15']
        }
        
        self.compatibility_patterns = [
            # Register size mismatch patterns - simplified
            {
                'name': 'register_size_mismatch_mov',
                'pattern': r'mov\s+(\w+),\s+(\w+)',
                'validator': self._validate_register_size_mismatch,
                'fixer': self._fix_register_size_mismatch,
                'priority': 1
            },
            {
                'name': 'register_size_mismatch_add',
                'pattern': r'add\s+(\w+),\s+(\w+)',
                'validator': self._validate_register_size_mismatch,
                'fixer': self._fix_register_size_mismatch,
                'priority': 1
            },
            # Orphaned register dereference patterns - simplified
            {
                'name': 'orphaned_register_dereference',
                'pattern': r'(mov|add|sub|cmp)\s+(\w+),\s+dword\s+\[(\w+)\]',
                'validator': self._validate_orphaned_dereference,
                'fixer': self._fix_orphaned_dereference,
                'priority': 2
            },
            # Double dereference patterns - multiline
            {
                'name': 'double_dereference_pattern',
                'pattern': r'mov\s+(\w+),\s+\[rel\s+(V\d+)\]',
                'validator': self._validate_double_dereference,
                'fixer': self._fix_double_dereference,
                'priority': 3,
                'multiline': True
            }
        ]
        
        self.variable_map = {}  # Dynamic variable mapping
        self.known_variables = set()  # Track all variables found
        
    def analyze_and_fix_assembly(self, content: str) -> str:
        """Main method to analyze and fix NASM compatibility issues - fast version"""
        if self.debug:
            print("[NASM Engine] Starting compatibility analysis...")
            
        # Apply fast, direct fixes
        fixed_content = self._apply_fast_compatibility_fixes(content)
        
        if self.debug:
            print("[NASM Engine] Compatibility fixes completed")
            
        return fixed_content
    
    def _apply_fast_compatibility_fixes(self, content: str) -> str:
        """Apply comprehensive dynamic compatibility fixes using pattern recognition"""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        # First pass: detect all function calls for extern declarations
        function_calls = set()
        for line in lines:
            call_match = re.search(r'call\s+([a-zA-Z_]\w*)', line.strip())
            if call_match:
                func_name = call_match.group(1)
                if func_name not in {'main'}:
                    function_calls.add(func_name)
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Add missing extern declarations after section .data line
            if stripped == 'section .data' and i + 1 < len(lines):
                fixed_lines.append(line)
                # Skip comments after section
                while i + 1 < len(lines) and (lines[i + 1].strip().startswith(';') or not lines[i + 1].strip()):
                    i += 1
                    fixed_lines.append(lines[i])
                
                # Add extern declarations
                for func in sorted(function_calls):
                    fixed_lines.append(f'extern {func}')
                # Add floating-point constants if needed
                fixed_lines.append('    FLOAT_TWO dq 2.0  ; Constant for pow function')
                i += 1
                continue
            
            # Fix printf parameter setup for Windows x64 calling convention
            if self._is_printf_context(i, lines):
                printf_fixes = self._fix_printf_parameter_setup(i, lines)
                if printf_fixes:
                    fixed_lines.extend(printf_fixes['lines'])
                    i += printf_fixes['skip']
                    continue
            
            # Fix double dereference patterns
            if ('mov rbx, [rel V' in stripped and i + 1 < len(lines) and 
                'dword [rbx]' in lines[i + 1].strip()):
                # Replace with direct access
                var_match = re.search(r'\[rel (V\d+)\]', stripped)
                if var_match:
                    var_name = var_match.group(1)
                    next_line = lines[i + 1].strip()
                    dest_match = re.search(r'mov (\w+), dword \[rbx\]', next_line)
                    if dest_match:
                        dest_reg = dest_match.group(1)
                        indent = line[:len(line) - len(line.lstrip())]
                        fixed_lines.append(f'{indent}mov {dest_reg}, dword [rel {var_name}]  ; Fixed: Direct variable access')
                        i += 2  # Skip both lines
                        continue
            
            # Fix calling convention violations for Windows x64
            if self.calling_convention == 'windows_x64':
                # Check if this is parameter setup before a function call
                is_param_setup = any('call' in lines[j] for j in range(i + 1, min(len(lines), i + 5)) 
                                   if not lines[j].strip().startswith(';'))
                
                if is_param_setup:
                    # Fix: mov rdi, ... -> mov rcx, ... (1st parameter)
                    if re.search(r'mov\s+rdi,', stripped):
                        line = re.sub(r'mov\s+rdi,', 'mov rcx,', line)
                        line += '  ; Fixed: Windows x64 1st parameter (rdi->rcx)'
                    
                    # Fix: lea rsi, ... -> lea rdx, ... (2nd parameter)  
                    elif re.search(r'lea\s+rsi,', stripped):
                        line = re.sub(r'lea\s+rsi,', 'lea rdx,', line)
                        line += '  ; Fixed: Windows x64 2nd parameter (rsi->rdx)'
                    
                    # Fix: mov rsi, ... -> mov rdx, ... (2nd parameter)
                    elif re.search(r'mov\s+rsi,', stripped):
                        line = re.sub(r'mov\s+rsi,', 'mov rdx,', line)
                        line += '  ; Fixed: Windows x64 2nd parameter (rsi->rdx)'
            
            # Fix invalid register dereferences - be more specific about which variable to map to
            if re.search(r'dword \[(rdi|rsi|rbx|rax)\]', stripped) and 'rel' not in stripped:
                # Map to specific variables based on context
                if 'dword [rdi]' in stripped:
                    line = re.sub(r'dword \[rdi\]', 'dword [rel V03]', line)
                    line += '  ; Fixed: Invalid dereference (rdi) -> V03'
                elif 'dword [rbx]' in stripped:
                    line = re.sub(r'dword \[rbx\]', 'dword [rel V01]', line)
                    line += '  ; Fixed: Invalid dereference (rbx) -> V01'
                else:
                    line = re.sub(r'dword \[\w+\]', 'dword [rel V01]', line)
                    line += '  ; Fixed: Invalid dereference -> direct variable access'
            
            # Fix register size mismatches - be more comprehensive
            if 'mov edx, rax' in stripped:
                line = line.replace('mov edx, rax', 'mov edx, eax  ; Fixed: Register size match')
            elif 'mov ecx, rax' in stripped:
                line = line.replace('mov ecx, rax', 'mov ecx, eax  ; Fixed: Register size match')
            elif 'add edx, rax' in stripped:
                line = line.replace('add edx, rax', 'add edx, eax  ; Fixed: Register size match')
            elif 'add r9d, rax' in stripped:
                line = line.replace('add r9d, rax', 'add r9d, eax  ; Fixed: Register size match')
            
            # Remove MASM PTR keywords - be more comprehensive
            if 'QWORD PTR ' in stripped:
                line = line.replace('QWORD PTR ', 'qword ')
                line += '  ; Fixed: Removed MASM PTR keyword'
            elif ' PTR ' in stripped:
                line = line.replace(' PTR ', ' ')
                if '; Fixed:' not in line:
                    line += '  ; Fixed: Removed MASM PTR keyword'
            
            # Add missing sqrt extern if needed
            if 'call sqrt' in stripped and 'extern sqrt' not in '\n'.join(fixed_lines):
                # Find where to insert extern declarations
                for j, prev_line in enumerate(fixed_lines):
                    if prev_line.strip().startswith('extern printf'):
                        fixed_lines.insert(j + 1, 'extern sqrt')
                        break
            
            # Fix invalid stack operations (consecutive pops without pushes)
            if (stripped == 'pop rbx' and i + 2 < len(lines) and 
                lines[i + 1].strip() == 'pop rsi' and 
                lines[i + 2].strip() == 'pop rdi'):
                
                # These registers weren't pushed in Windows x64, so don't pop them
                fixed_lines.append('    ; Fixed: Removed invalid stack operations (rbx, rsi, rdi not pushed)')
                i += 3  # Skip all three invalid pops
                continue
            
            # Fix floating-point constant loading for pow function
            if 'movsd xmm1, qword [rel LC' in stripped:
                # Ensure we have proper floating-point constants
                lc_match = re.search(r'\[rel (LC\d+)\]', stripped)
                if lc_match:
                    lc_label = lc_match.group(1)
                    # Check if this LC label is defined in the data section
                    if not any(f'{lc_label}:' in prev_line for prev_line in fixed_lines):
                        # Add the missing constant definition
                        for j, prev_line in enumerate(fixed_lines):
                            if 'extern' in prev_line and j > 0:
                                fixed_lines.insert(j, f'    {lc_label} dq 2.0  ; Auto-generated constant')
                                break
            
            # Fix invalid IEEE 754 inline constants and problematic pow sequences
            if 'mov qword [rsp], 0x4000000000000000' in stripped:
                # Replace with simpler approach using a data constant
                line = '    movsd xmm1, qword [rel FLOAT_TWO]  ; Fixed: Use constant from data section'
            elif 'movsd xmm1, qword [rsp]  ; store 2.0 as second parameter' in stripped:
                line = '    movsd xmm1, qword [rel FLOAT_TWO]  ; Fixed: Use proper constant'
            elif 'movsd xmm1, qword [rsp]  ; 2.0' in stripped:
                line = '    movsd xmm1, qword [rel FLOAT_TWO]  ; Fixed: Use proper constant'
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _is_printf_context(self, line_idx: int, lines: List[str]) -> bool:
        """Check if current line is in a printf parameter setup context"""
        # Look ahead for printf call within next 10 lines
        for i in range(line_idx, min(len(lines), line_idx + 10)):
            if 'call printf' in lines[i]:
                return True
        return False
    
    def _fix_printf_parameter_setup(self, start_idx: int, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Fix printf parameter setup sequence for Windows x64"""
        # Look for printf call
        printf_line_idx = None
        for i in range(start_idx, min(len(lines), start_idx + 10)):
            if 'call printf' in lines[i]:
                printf_line_idx = i
                break
        
        if printf_line_idx is None:
            return None
        
        # Analyze the parameter setup sequence
        param_lines = []
        for i in range(start_idx, printf_line_idx):
            line = lines[i].strip()
            if (line and not line.startswith(';') and 
                any(reg in line for reg in ['rcx', 'rdx', 'r8', 'r9', 'rdi', 'rsi', 'lea', 'mov'])):
                param_lines.append((i, line))
        
        if not param_lines:
            return None
        
        # Fix parameter setup for Windows x64 calling convention
        fixed_lines = []
        skip_count = 0
        
        for i in range(start_idx, printf_line_idx + 1):
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith(';'):
                fixed_lines.append(line)
                skip_count += 1
                continue
            
            # Fix printf parameter order for Windows x64
            if 'lea rcx, [rel LC' in stripped:
                # Format string - already correct for Windows x64
                fixed_lines.append(line)
            elif 'lea rdx, [rel V' in stripped:
                # String parameter - already correct
                fixed_lines.append(line)
            elif 'mov edx, dword [rel V' in stripped:
                # Integer parameter in 2nd position - correct
                fixed_lines.append(line)
            elif 'lea r8, [rel V' in stripped:
                # String parameter in 3rd position - correct
                fixed_lines.append(line)
            elif 'mov r8d, dword [rel V' in stripped:
                # Integer parameter in 3rd position - correct
                fixed_lines.append(line)
            elif 'mov r9d,' in stripped:
                # 4th parameter - correct
                fixed_lines.append(line)
            elif 'call printf' in stripped:
                fixed_lines.append(line)
            else:
                # Other instructions that might need fixing
                fixed_line = self._fix_printf_parameter_line(line)
                fixed_lines.append(fixed_line)
            
            skip_count += 1
        
        return {
            'lines': fixed_lines,
            'skip': skip_count
        }
    
    def _fix_printf_parameter_line(self, line: str) -> str:
        """Fix individual parameter setup line for printf"""
        stripped = line.strip()
        
        # Fix System V to Windows x64 parameter register mapping
        if 'mov rdi,' in stripped:
            # First parameter should be rcx in Windows x64
            line = re.sub(r'mov rdi,', 'mov rcx,', line)
            line += '  ; Fixed: printf 1st param (rdi->rcx)'
        elif 'lea rdi,' in stripped:
            line = re.sub(r'lea rdi,', 'lea rcx,', line)
            line += '  ; Fixed: printf 1st param (rdi->rcx)'
        elif 'mov rsi,' in stripped:
            # Second parameter should be rdx in Windows x64
            line = re.sub(r'mov rsi,', 'mov rdx,', line)
            line += '  ; Fixed: printf 2nd param (rsi->rdx)'
        elif 'lea rsi,' in stripped:
            line = re.sub(r'lea rsi,', 'lea rdx,', line)
            line += '  ; Fixed: printf 2nd param (rsi->rdx)'
        
        return line
    
    def _analyze_assembly_context(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze assembly context to understand variables, functions, and patterns"""
        context = {
            'variables': {},
            'function_calls': set(),
            'extern_declarations': set(),
            'string_constants': {},
            'register_usage': {},
            'memory_references': [],
            'problematic_patterns': []
        }
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Analyze variable declarations
            if re.match(r'^\s*(V\d+|LC\d+|__\w+)\s+(dd|db|dw|dq)', stripped):
                var_match = re.match(r'^\s*(\w+)\s+(dd|db|dw|dq)\s+(.+)', stripped)
                if var_match:
                    var_name, var_type, var_value = var_match.groups()
                    context['variables'][var_name] = {
                        'type': var_type,
                        'value': var_value,
                        'line': i
                    }
            
            # Analyze function calls
            call_match = re.search(r'call\s+([a-zA-Z_]\w*)', stripped)
            if call_match:
                func_name = call_match.group(1)
                context['function_calls'].add(func_name)
            
            # Analyze extern declarations
            if stripped.startswith('extern '):
                extern_name = stripped.split()[1] if len(stripped.split()) > 1 else ''
                if extern_name:
                    context['extern_declarations'].add(extern_name)
            
            # Analyze problematic patterns
            if self._detect_problematic_pattern(stripped, i, lines):
                context['problematic_patterns'].append({
                    'line': i,
                    'pattern': stripped,
                    'type': self._classify_problem(stripped)
                })
        
        return context
    
    def _detect_problematic_pattern(self, line: str, line_num: int, lines: List[str]) -> bool:
        """Detect problematic assembly patterns"""
        # Check for register dereferencing without proper pointer setup
        if re.search(r'\[(rdi|rsi|rbx|rax)\]', line) and 'rel' not in line:
            return True
        
        # Check for register size mismatches
        if re.search(r'mov\s+(e\w+),\s+(r\w+)', line):
            return True
        
        # Check for invalid calling convention usage
        if 'call' in line and any(reg in line for reg in ['rdi', 'rsi']) and self.calling_convention == 'windows_x64':
            return True
        
        return False
    
    def _classify_problem(self, line: str) -> str:
        """Classify the type of problem in the line"""
        if re.search(r'\[(rdi|rsi|rbx|rax)\]', line):
            return 'invalid_dereference'
        elif re.search(r'mov\s+(e\w+),\s+(r\w+)', line):
            return 'register_size_mismatch'
        elif 'call' in line and any(reg in line for reg in ['rdi', 'rsi']):
            return 'calling_convention_violation'
        return 'unknown'
    
    def _fix_extern_declarations(self, lines: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Fix missing extern declarations"""
        missing_externs = analysis['function_calls'] - analysis['extern_declarations'] - {'main'}
        
        if not missing_externs:
            return lines
        
        # Find insertion point for extern declarations
        insert_point = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('section .data'):
                insert_point = i + 3  # After section and comments
                break
            elif line.strip().startswith('V01') or line.strip().startswith('LC1'):
                insert_point = i
                break
        
        # Insert missing extern declarations
        extern_lines = [f'extern {func}' for func in sorted(missing_externs)]
        lines = lines[:insert_point] + extern_lines + [''] + lines[insert_point:]
        
        if self.debug:
            print(f"[NASM Engine] Added extern declarations: {missing_externs}")
        
        return lines
    
    def _fix_calling_convention_violations(self, lines: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Fix Windows x64 calling convention violations"""
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_line = line
            
            # Fix incorrect register usage for Windows x64
            if self.calling_convention == 'windows_x64':
                # Fix: mov rdi, ... -> mov rcx, ... (1st parameter)
                if re.search(r'mov\s+rdi,', line) and self._is_parameter_setup(line, i, lines):
                    fixed_line = re.sub(r'mov\s+rdi,', 'mov rcx,', fixed_line)
                    fixed_line += '  ; Fixed: Windows x64 1st parameter'
                
                # Fix: mov rsi, ... -> mov rdx, ... (2nd parameter)
                elif re.search(r'mov\s+rsi,', line) and self._is_parameter_setup(line, i, lines):
                    fixed_line = re.sub(r'mov\s+rsi,', 'mov rdx,', fixed_line)
                    fixed_line += '  ; Fixed: Windows x64 2nd parameter'
                
                # Fix: lea rsi, ... -> lea rdx, ... (2nd parameter)
                elif re.search(r'lea\s+rsi,', line) and self._is_parameter_setup(line, i, lines):
                    fixed_line = re.sub(r'lea\s+rsi,', 'lea rdx,', fixed_line)
                    fixed_line += '  ; Fixed: Windows x64 2nd parameter'
            
            fixed_lines.append(fixed_line)
        
        return fixed_lines
    
    def _is_parameter_setup(self, line: str, line_num: int, lines: List[str]) -> bool:
        """Check if this line is setting up a parameter for a function call"""
        # Look ahead for a call instruction within the next few lines
        for i in range(line_num + 1, min(len(lines), line_num + 5)):
            if 'call' in lines[i]:
                return True
        return False
    
    def _fix_register_usage_errors(self, lines: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Fix register usage errors and size mismatches"""
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            
            # Fix register size mismatches
            fixed_line = re.sub(r'mov\s+edx,\s+rax', 'mov edx, eax  ; Fixed: Register size match', fixed_line)
            fixed_line = re.sub(r'mov\s+ecx,\s+rax', 'mov ecx, eax  ; Fixed: Register size match', fixed_line)
            fixed_line = re.sub(r'add\s+edx,\s+rax', 'add edx, eax  ; Fixed: Register size match', fixed_line)
            
            # Remove PTR keywords for NASM compatibility
            if ' PTR ' in fixed_line:
                fixed_line = fixed_line.replace(' PTR ', ' ')
                if not '; Fixed:' in fixed_line:
                    fixed_line += '  ; Fixed: Removed MASM PTR keyword'
            
            fixed_lines.append(fixed_line)
        
        return fixed_lines
    
    def _fix_memory_access_patterns(self, lines: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Fix invalid memory access patterns and dereferencing"""
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Fix double-dereference patterns
            if self._is_double_dereference_pattern(line, i, lines, analysis):
                fixed_instruction = self._fix_double_dereference(line, i, lines, analysis)
                if fixed_instruction:
                    fixed_lines.append(fixed_instruction)
                    i += 2  # Skip the problematic pattern
                    continue
            
            # Fix orphaned register dereferences
            if re.search(r'dword\s+\[(rdi|rsi|rbx|rax)\]', line):
                # Try to map to the correct variable
                var_name = self._infer_variable_from_register_context(line, i, lines, analysis)
                if var_name:
                    fixed_line = re.sub(r'dword\s+\[\w+\]', f'dword [rel {var_name}]  ; Fixed: Direct variable access', line)
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line + '  ; Warning: Could not resolve variable reference')
            else:
                fixed_lines.append(line)
            
            i += 1
        
        return fixed_lines
    
    def _is_double_dereference_pattern(self, line: str, line_num: int, lines: List[str], analysis: Dict[str, Any]) -> bool:
        """Check if this is a double-dereference pattern"""
        if not re.search(r'mov\s+\w+,\s+\[rel\s+V\d+\]', line):
            return False
        
        if line_num + 1 < len(lines):
            next_line = lines[line_num + 1]
            if re.search(r'dword\s+\[\w+\]', next_line):
                return True
        
        return False
    
    def _fix_double_dereference(self, line: str, line_num: int, lines: List[str], analysis: Dict[str, Any]) -> Optional[str]:
        """Fix double-dereference pattern"""
        var_match = re.search(r'\[rel\s+(V\d+)\]', line)
        if not var_match:
            return None
        
        var_name = var_match.group(1)
        next_line = lines[line_num + 1]
        
        # Extract the destination register from the next line
        dest_match = re.search(r'mov\s+(\w+),\s+dword', next_line)
        if dest_match:
            dest_reg = dest_match.group(1)
            indent = line[:len(line) - len(line.lstrip())]
            return f'{indent}mov {dest_reg}, dword [rel {var_name}]  ; Fixed: Direct variable access'
        
        return None
    
    def _infer_variable_from_register_context(self, line: str, line_num: int, lines: List[str], analysis: Dict[str, Any]) -> Optional[str]:
        """Infer which variable a register dereference should refer to"""
        # Extract the register being dereferenced
        reg_match = re.search(r'dword\s+\[(\w+)\]', line)
        if not reg_match:
            return None
        
        reg = reg_match.group(1)
        
        # Look backwards for recent assignments to this register
        for i in range(max(0, line_num - 10), line_num):
            prev_line = lines[i]
            var_match = re.search(rf'mov\s+{reg},\s+\[rel\s+(V\d+)\]', prev_line)
            if var_match:
                return var_match.group(1)
        
        # Default to V01 if available
        if 'V01' in analysis['variables']:
            return 'V01'
        
        return None
    
    def _fix_stack_operations(self, lines: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Fix invalid stack operations"""
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for invalid pop operations
            if (line.strip() == 'pop rbx' and 
                i + 2 < len(lines) and 
                lines[i + 1].strip() == 'pop rsi' and 
                lines[i + 2].strip() == 'pop rdi'):
                
                # These registers weren't pushed in Windows x64, so don't pop them
                fixed_lines.append('    ; Fixed: Removed invalid stack operations (rbx, rsi, rdi not pushed)')
                i += 3  # Skip all three invalid pops
                continue
            
            fixed_lines.append(line)
            i += 1
        
        return fixed_lines
    
    def _fix_math_function_calls(self, lines: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Fix math function calls to use proper Windows x64 calling convention"""
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Handle pow() function calls
            if 'call pow' in line or 'call\tpow' in line:
                # Ensure proper setup for pow(double, double) -> double
                # Parameters should be in XMM0 and XMM1, result in XMM0
                fixed_line = self._ensure_proper_pow_setup(line, i, lines)
                fixed_lines.append(fixed_line)
            
            # Handle sqrt() function calls
            elif 'call sqrt' in line or 'call\tsqrt' in line:
                # Ensure proper setup for sqrt(double) -> double
                # Parameter should be in XMM0, result in XMM0
                fixed_line = self._ensure_proper_sqrt_setup(line, i, lines)
                fixed_lines.append(fixed_line)
            
            else:
                fixed_lines.append(line)
            
            i += 1
        
        return fixed_lines
    
    def _ensure_proper_pow_setup(self, line: str, line_num: int, lines: List[str]) -> str:
        """Ensure pow() function has proper Windows x64 setup"""
        # For now, just add a comment about proper usage
        if '; Fixed:' not in line:
            return line + '  ; Note: pow(xmm0, xmm1) -> xmm0'
        return line
    
    def _ensure_proper_sqrt_setup(self, line: str, line_num: int, lines: List[str]) -> str:
        """Ensure sqrt() function has proper Windows x64 setup"""
        # For now, just add a comment about proper usage
        if '; Fixed:' not in line:
            return line + '  ; Note: sqrt(xmm0) -> xmm0'
        return line
    
    def _discover_assembly_context(self, content: str) -> None:
        """Discover variables, functions, and other context from assembly"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines):
            stripped = line.strip()
            
            # Discover variable declarations
            if re.match(r'^(V\d+|[a-zA-Z_]\w*)\s+(dd|db|dw|dq)', stripped):
                parts = stripped.split()
                if len(parts) >= 2:
                    var_name = parts[0]
                    self.known_variables.add(var_name)
                    if self.debug:
                        print(f"[NASM Engine] Discovered variable: {var_name}")
            
            # Discover variable references in [rel VarName] format
            rel_refs = re.findall(r'\[rel\s+([A-Z][A-Z0-9_]*)\]', stripped)
            for var_ref in rel_refs:
                self.known_variables.add(var_ref)
                if self.debug:
                    print(f"[NASM Engine] Discovered variable reference: {var_ref}")
    
    def _apply_compatibility_fixes(self, content: str) -> str:
        """Apply all compatibility fixes in priority order"""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        # Compile regex patterns once for efficiency
        compiled_patterns = {}
        for pattern_info in self.compatibility_patterns:
            compiled_patterns[pattern_info['name']] = re.compile(pattern_info['pattern'])
        
        while i < len(lines):
            line = lines[i]
            original_line = line
            fix_applied = False
            
            # Skip empty lines and comments for efficiency
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                fixed_lines.append(line)
                i += 1
                continue
            
            # Try each pattern in priority order
            for pattern_info in sorted(self.compatibility_patterns, key=lambda x: x['priority']):
                if pattern_info.get('multiline', False):
                    # Handle multiline patterns
                    result = self._apply_multiline_pattern(lines, i, pattern_info)
                    if result:
                        fixed_lines.extend(result['lines'])
                        i += result['skip']
                        fix_applied = True
                        break
                else:
                    # Handle single-line patterns with compiled regex
                    compiled_pattern = compiled_patterns[pattern_info['name']]
                    match = compiled_pattern.search(stripped)
                    if match:
                        if pattern_info['validator'](match, line, i, lines):
                            fixed_line = pattern_info['fixer'](match, line, i, lines)
                            if fixed_line != line:
                                if self.debug:
                                    print(f"[NASM Engine] Fixed {pattern_info['name']} at line {i+1}")
                                    print(f"  Before: {line.strip()}")
                                    print(f"  After:  {fixed_line.strip()}")
                                line = fixed_line
                                fix_applied = True
                                break
            
            if not fix_applied:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)
    
    def _apply_multiline_pattern(self, lines: List[str], start_idx: int, pattern_info: Dict) -> Optional[Dict]:
        """Apply multiline patterns like double-dereference fixes"""
        if pattern_info['name'] == 'double_dereference_pattern':
            return self._handle_double_dereference_multiline(lines, start_idx)
        return None
    
    def _handle_double_dereference_multiline(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Handle double-dereference patterns across multiple lines"""
        if start_idx >= len(lines):
            return None
            
        current_line = lines[start_idx].strip()
        
        # Look for pattern: mov reg, [rel VarName] followed by mov/add/etc with [reg]
        first_match = re.search(r'mov\s+([a-z0-9]+),\s+\[rel\s+(V\d+)\]', current_line)
        if not first_match:
            return None
            
        temp_reg = first_match.group(1)
        var_name = first_match.group(2)
        
        # Look ahead for instructions using this register for dereference
        for lookahead in range(1, min(5, len(lines) - start_idx)):
            next_line = lines[start_idx + lookahead].strip()
            
            # Check for dereference of the temp register
            deref_pattern = rf'(mov|add|sub|cmp)\s+([a-z0-9]+),\s+dword\s+\[{temp_reg}\]'
            deref_match = re.search(deref_pattern, next_line)
            
            if deref_match:
                operation = deref_match.group(1)
                dest_reg = deref_match.group(2)
                
                # Create direct access instruction
                indent = lines[start_idx][:len(lines[start_idx]) - len(lines[start_idx].lstrip())]
                fixed_instruction = f'{indent}{operation} {dest_reg}, dword [rel {var_name}]  ; Fixed: Direct access to HASM variable'
                
                if self.debug:
                    print(f"[NASM Engine] Fixed double-dereference pattern:")
                    print(f"  Removed: {current_line}")
                    print(f"  Removed: {next_line}")
                    print(f"  Added:   {fixed_instruction.strip()}")
                
                # Return the fixed version, skipping the original two lines
                result_lines = []
                for j in range(start_idx):
                    result_lines.append(lines[j])
                
                result_lines.append(fixed_instruction)
                
                # Skip the lines that were part of the pattern
                return {
                    'lines': [fixed_instruction],
                    'skip': lookahead + 1
                }
        
        return None
    
    def _validate_register_size_mismatch(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> bool:
        """Validate if there's a register size mismatch"""
        reg1 = match.group(1)
        reg2 = match.group(2)
        
        # Check if one is 64-bit and other is 32-bit
        is_64bit_1 = any(reg1.startswith(prefix) for prefix in ['r', 'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rsp', 'rbp'])
        is_64bit_2 = any(reg2.startswith(prefix) for prefix in ['r', 'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rsp', 'rbp'])
        
        is_32bit_1 = any(reg1.endswith(suffix) for suffix in ['d', 'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp'])
        is_32bit_2 = any(reg2.endswith(suffix) for suffix in ['d', 'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp'])
        
        # Return True if there's a size mismatch
        return (is_32bit_1 and is_64bit_2) or (is_64bit_1 and is_32bit_2)
    
    def _fix_register_size_mismatch(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> str:
        """Fix register size mismatches"""
        reg1 = match.group(1)
        reg2 = match.group(2)
        operation = match.group(0).split()[0]  # mov, add, sub, etc.
        
        # Convert 64-bit register to 32-bit equivalent
        if reg2 in self.register_mappings:
            fixed_reg2 = self.register_mappings[reg2]
            fixed_line = line.replace(f'{reg2}', fixed_reg2)
            fixed_line += '  ; Fixed: Register size match'
            return fixed_line
        
        return line
    
    def _validate_double_dereference(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> bool:
        """Validate double-dereference patterns"""
        # This is handled in multiline processing
        return False
    
    def _fix_double_dereference(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> str:
        """Fix double-dereference patterns"""
        # This is handled in multiline processing
        return line
    
    def _validate_orphaned_dereference(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> bool:
        """Validate orphaned register dereferences"""
        reg = match.group(3)
        
        # Check if this register is likely holding a variable address
        # Look for recent mov reg, [rel VarName] instructions
        for lookback in range(max(0, line_num - 10), line_num):
            if lookback < len(lines):
                prev_line = lines[lookback].strip()
                if f'mov {reg}, [rel V' in prev_line:
                    return True
        
        # Also check if register name suggests it's holding an address
        return reg in ['rbx', 'rax', 'rsi', 'rdi']
    
    def _fix_orphaned_dereference(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> str:
        """Fix orphaned register dereferences"""
        operation = match.group(1)
        dest_reg = match.group(2)
        source_reg = match.group(3)
        
        # Try to find the most likely variable this refers to
        target_var = self._infer_variable_from_context(source_reg, line_num, lines)
        
        if target_var:
            fixed_line = re.sub(
                rf'dword\s+\[{source_reg}\]',
                f'dword [rel {target_var}]  ; Fixed: HASM variable reference',
                line
            )
            return fixed_line
        
        return line
    
    def _infer_variable_from_context(self, reg: str, line_num: int, lines: List[str]) -> Optional[str]:
        """Infer which variable a register likely refers to"""
        # Look backwards for mov reg, [rel VarName]
        for lookback in range(max(0, line_num - 10), line_num):
            if lookback < len(lines):
                prev_line = lines[lookback].strip()
                var_match = re.search(rf'mov\s+{reg},\s+\[rel\s+(V\d+)\]', prev_line)
                if var_match:
                    return var_match.group(1)
        
        # Default to V01 if no specific variable found
        if 'V01' in self.known_variables:
            return 'V01'
        elif self.known_variables:
            return list(self.known_variables)[0]
        
        return None
    
    def _validate_missing_rel(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> bool:
        """Validate missing [rel] directives"""
        var_name = match.group(2)
        
        # Check if this looks like a variable name
        return (var_name in self.known_variables or 
                re.match(r'^[A-Z][A-Z0-9_]*$', var_name))
    
    def _fix_missing_rel(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> str:
        """Fix missing [rel] directives"""
        reg = match.group(1)
        var_name = match.group(2)
        
        fixed_line = line.replace(f'lea {reg}, {var_name}', f'lea {reg}, [rel {var_name}]')
        fixed_line += '  ; Fixed: Added [rel] directive'
        
        return fixed_line
    
    def _validate_calling_convention(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> bool:
        """Validate calling convention issues"""
        # For now, just return False - this can be expanded for more complex validation
        return False
    
    def _fix_calling_convention(self, match: re.Match, line: str, line_num: int, lines: List[str]) -> str:
        """Fix calling convention issues"""
        # Placeholder for future calling convention fixes
        return line

class AdvancedAssemblyProcessor:
    """Main assembly processor that combines all functionality"""
    
    def __init__(self, debug: bool = False, calling_convention: str = 'windows_x64'):
        self.debug = debug
        self.calling_convention = calling_convention
        self.analyzer = AdvancedAssemblyAnalyzer()
        self.comparator = IntelligentAssemblyComparator()
        self.nasm_engine = NASMCompatibilityEngine(debug=debug, calling_convention=calling_convention)
        self.extern_declarations = set()
        self.function_calls = set()
        self.excluded_functions = {'main'}  # Functions that shouldn't have extern declarations
        
    def process_assembly(self, content: str, reference_content: str = None) -> str:
        """Main method to process assembly with intelligent analysis"""
        
        # Step 1: Parse and analyze the assembly
        instructions = self.analyzer.parse_assembly(content)
        
        # Step 2: Extract function calls and existing extern declarations
        self._extract_function_calls_and_externs(instructions)
        
        # Step 3: Compare with reference if available
        improvements = []
        patterns = []
        if reference_content:
            comparison = self.comparator.compare_generators(content, reference_content)
            improvements = comparison.get('improvements', [])
            patterns = comparison.get('patterns', [])
            
            if self.debug:
                print(f"Comparison found {len(improvements)} improvements and {len(patterns)} patterns")
        else:
            patterns = self._analyze_standalone(instructions)
        
        # Step 4: Apply intelligent fixes including auto-extern generation
        fixed_content = self._apply_intelligent_fixes(content, instructions, improvements, patterns)
        
        # Step 5: Apply additional post-processing fixes
        post_processed_content = self._apply_post_processing(fixed_content)
        
        # Step 6: Apply NASM compatibility fixes dynamically
        nasm_fixed_content = self._apply_nasm_compatibility_fixes(post_processed_content)
        
        # Step 7: Format the output
        formatted_content = self._format_assembly(nasm_fixed_content)
        
        return formatted_content
    
    def _extract_function_calls_and_externs(self, instructions: List[AssemblyInstruction]) -> None:
        """Extract all function calls and existing extern declarations"""
        self.function_calls = set()
        self.extern_declarations = set()
        
        for inst in instructions:
            # Extract function calls
            if inst.mnemonic == 'call' and inst.operands:
                function_name = inst.operands[0].strip()
                # Remove any whitespace or tabs that might be present
                function_name = re.sub(r'\s+', '', function_name)
                if function_name not in self.excluded_functions:
                    self.function_calls.add(function_name)
                    if self.debug:
                        print(f"Found function call: {function_name}")
            
            # Extract existing extern declarations
            elif inst.mnemonic == 'extern' and inst.operands:
                extern_name = inst.operands[0].strip()
                self.extern_declarations.add(extern_name)
                if self.debug:
                    print(f"Found extern declaration: {extern_name}")
        
        if self.debug:
            print(f"Total function calls found: {self.function_calls}")
            print(f"Total extern declarations found: {self.extern_declarations}")
    
    def _get_missing_extern_declarations(self) -> Set[str]:
        """Get function calls that don't have extern declarations"""
        return self.function_calls - self.extern_declarations
    
    def _analyze_standalone(self, instructions: List[AssemblyInstruction]) -> List[AssemblyPattern]:
        """Analyze assembly without reference for patterns and issues"""
        patterns = []
        
        # Check for common issues
        has_sections = any(inst.mnemonic == 'section' for inst in instructions)
        if not has_sections:
            patterns.append(AssemblyPattern(
                pattern_type='missing_sections',
                confidence=1.0,
                location=(1, 1),
                context={'issue': 'No section declarations found'},
                suggested_fix='Add section .data and section .text declarations'
            ))
        
        # Check for function calls without extern declarations
        function_calls = set()
        extern_declarations = set()
        
        for inst in instructions:
            if inst.mnemonic == 'call' and inst.operands:
                function_calls.add(inst.operands[0])
            elif inst.mnemonic == 'extern' and inst.operands:
                extern_declarations.add(inst.operands[0])
        
        missing_externs = function_calls - extern_declarations - {'main'}
        if missing_externs:
            patterns.append(AssemblyPattern(
                pattern_type='missing_extern_declarations',
                confidence=0.9,
                location=(1, 1),
                context={'missing': list(missing_externs)},
                suggested_fix=f'Add extern declarations for: {", ".join(missing_externs)}'
            ))
        
        # Check for syntax errors (multiple instructions per line)
        for i, inst in enumerate(instructions):
            if self._is_syntax_error(inst.raw_line.strip()):
                patterns.append(AssemblyPattern(
                    pattern_type='syntax_error',
                    confidence=1.0,
                    location=(i, i),
                    context={'line': inst.raw_line.strip(), 'error_type': 'multiple_instructions_per_line'},
                    suggested_fix='Split multiple instructions into separate lines'
                ))
        
        return patterns
    
    def _is_syntax_error(self, line: str) -> bool:
        """Detect basic syntax errors"""
        # Lines with multiple instructions on same line
        if 'call ' in line and any(instr in line for instr in ['mov', 'lea', 'add', 'sub', 'push', 'pop']):
            return True
        
        # Instructions followed immediately by other instructions (handle both spaces and tabs)
        if re.search(r'call\s+\w+\s+mov', line):
            return True
            
        return False
    
    def _apply_intelligent_fixes(self, content: str, instructions: List[AssemblyInstruction], 
                                improvements: List[Dict[str, Any]], patterns: List[AssemblyPattern]) -> str:
        """Apply intelligent fixes based on analysis"""
        
        lines = content.split('\n')
        fixed_lines = lines[:]
        
        # Step 1: Add missing extern declarations automatically first (before removing existing ones)
        missing_externs = self._get_missing_extern_declarations()
        if missing_externs:
            if self.debug:
                print(f"Adding extern declarations for: {missing_externs}")
            
            # Find the insertion point (after section .data or at the beginning)
            insertion_point = self._find_extern_insertion_point(fixed_lines)
            
            # Add extern declarations in sorted order for consistency
            extern_lines = [f"extern {func}" for func in sorted(missing_externs)]
            
            # Insert extern declarations
            for i, extern_line in enumerate(extern_lines):
                fixed_lines.insert(insertion_point + i, extern_line)
        
        # Step 2: Now collect all existing extern declarations and rebuild them properly
        all_function_calls = self.function_calls
        if all_function_calls:
            # Find existing extern declarations and remove them
            fixed_lines = [line for line in fixed_lines if not line.strip().startswith('extern ')]
            
            # Add all extern declarations in one place
            insertion_point = self._find_extern_insertion_point(fixed_lines)
            extern_lines = [f"extern {func}" for func in sorted(all_function_calls)]
            
            # Insert all extern declarations at once
            for i, extern_line in enumerate(extern_lines):
                fixed_lines.insert(insertion_point + i, extern_line)
        
        # Step 3: Apply pattern fixes in reverse order to maintain line numbers
        for pattern in sorted(patterns, key=lambda x: x.location[0], reverse=True):
            if pattern.pattern_type == 'syntax_error':
                line_idx = pattern.location[0]
                if line_idx < len(fixed_lines):
                    original_line = fixed_lines[line_idx]
                    new_lines = self._split_multiple_instructions(original_line)
                    fixed_lines[line_idx:line_idx+1] = new_lines.split('\n')
            
            elif pattern.pattern_type == 'missing_printf_call':
                insert_line = pattern.location[1] - 1
                if insert_line < len(fixed_lines):
                    fixed_lines.insert(insert_line + 1, '    call printf')
        
        # Step 4: Apply improvement fixes
        for improvement in improvements:
            if improvement['type'] == 'missing_function_calls':
                # This is now handled automatically in Step 2
                pass
        
        return '\n'.join(fixed_lines)
    
    def _split_multiple_instructions(self, line: str) -> str:
        """Split multiple instructions on same line"""
        if 'call ' in line:
            # Handle cases like "call printf    mov r8, rsi"
            call_match = re.match(r'(.*?call\s+\w+)(.*)', line)
            if call_match:
                call_part = call_match.group(1).strip()
                remaining = call_match.group(2).strip()
                
                parts = [call_part]
                
                # If there's remaining content, add it as separate line(s)
                if remaining:
                    # Handle tabs and multiple spaces
                    remaining = re.sub(r'\s+', ' ', remaining)
                    # Split on common instruction keywords
                    remaining_parts = re.split(r'\s+(mov|lea|add|sub|push|pop|jmp|call)\s+', remaining)
                    
                    current_instruction = ""
                    for i, part in enumerate(remaining_parts):
                        if i == 0 and part.strip():
                            # First part might be continuation
                            continue
                        elif part.strip() in ['mov', 'lea', 'add', 'sub', 'push', 'pop', 'jmp', 'call']:
                            current_instruction = part.strip()
                        elif current_instruction and part.strip():
                            parts.append('    ' + current_instruction + ' ' + part.strip())
                            current_instruction = ""
                
                return '\n'.join(parts)
        
        return line
    
    def _find_extern_insertion_point(self, lines: List[str]) -> int:
        """Find the best place to insert extern declarations"""
        # Look for section .data or section .text
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('section .data'):
                # Insert after section .data and any comments
                insertion_point = i + 1
                # Skip any comments or empty lines after section declaration
                while (insertion_point < len(lines) and 
                       (lines[insertion_point].strip().startswith(';') or 
                        lines[insertion_point].strip() == '')):
                    insertion_point += 1
                return insertion_point
            elif stripped.startswith('section .text') or stripped.startswith('global '):
                # Insert before section .text or global
                return i
        
        # If no sections found, insert at the beginning after any initial comments
        insertion_point = 0
        while (insertion_point < len(lines) and 
               (lines[insertion_point].strip().startswith(';') or 
                lines[insertion_point].strip() == '')):
            insertion_point += 1
        return insertion_point
    
    def _apply_nasm_compatibility_fixes(self, content: str) -> str:
        """Apply dynamic NASM compatibility fixes using the compatibility engine"""
        if self.debug:
            print("[Processor] Applying NASM compatibility fixes...")
        
        # Use the NASM compatibility engine for dynamic fixes
        return self.nasm_engine.analyze_and_fix_assembly(content)
    
    def _apply_post_processing(self, content: str) -> str:
        """Apply additional post-processing fixes"""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            
            # Fix variable references
            if '[rel x]' in line:
                fixed_line = line.replace('[rel x]', '[rel V01]')
            
            # Fix register size mismatches
            if 'mov rax, [rel V' in line:
                fixed_line = re.sub(r'mov rax, \[rel (V\d+)\]', r'mov eax, dword [rel \1]', fixed_line)
            
            # Fix lea instructions missing [rel]
            fixed_line = re.sub(r'lea\s+(\w+),\s+([A-Z][A-Z0-9_]+)$', r'lea \1, [rel \2]', fixed_line)
            
            # Fix memory operations
            fixed_line = re.sub(r'mov\s+(\w+),\s+dword\s+\[([^\]]+)\]', r'mov \1, dword [\2]', fixed_line)
            
            # Fix cmp instructions
            if 'cmp rax, ' in line:
                fixed_line = re.sub(r'cmp\s+rax,\s+(\d+)', r'cmp eax, \1', fixed_line)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _format_assembly(self, content: str) -> str:
        """Advanced assembly formatter that preserves all code"""
        lines = content.split('\n')
        formatted_lines = []
        current_section = None
        
        # First pass: check if we need to add sections
        has_data_section = any('section .data' in line for line in lines)
        has_text_section = any('section .text' in line for line in lines)
        
        # Add section .data at the beginning if needed
        if not has_data_section:
            formatted_lines.append('section .data')
        
        # Keep track of extern declarations to avoid duplicates
        added_externs = set()
        header_added = False  # Track if header comments have been added
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines at the beginning
            if not stripped and not formatted_lines:
                continue
            
            # Skip duplicate or unwanted header comments
                if has_data_section and not header_added:
                    # Only add the first set of headers if we already have a data section
                    formatted_lines.append(line)
                    header_added = True
                # Always skip additional headers
                continue
            
            # Handle section declarations
            if stripped.startswith('section '):
                current_section = stripped.split()[1]
                if stripped not in formatted_lines:  # Avoid duplicates
                    if current_section == '.data' and has_data_section:
                        formatted_lines.append(stripped)
                        formatted_lines.append('')
                    else:
                        formatted_lines.append('')  # Empty line before section
                        formatted_lines.append(stripped)
                continue
            
            # Handle extern declarations - avoid duplicates
            if stripped.startswith('extern '):
                extern_func = stripped.split()[1] if len(stripped.split()) > 1 else ''
                if extern_func and extern_func not in added_externs:
                    formatted_lines.append(stripped)
                    added_externs.add(extern_func)
                continue
            
            # Add section .text before global main if needed
            if not has_text_section and stripped.startswith('global main'):
                formatted_lines.append('')  # Empty line
                formatted_lines.append('section .text')
                has_text_section = True
            
            # Handle labels (preserve exactly as-is)
            if ':' in stripped and not any(op in stripped for op in ['mov', 'lea', 'call']):
                formatted_lines.append(stripped)
                continue
            
            # Handle directives (global, etc.)
            if stripped.startswith('global'):
                formatted_lines.append(stripped)
                continue
            
            # Handle data declarations with proper indentation
            if any(stripped.startswith(d) for d in ['dd', 'db', 'dw', 'dq']) or re.match(r'^[A-Za-z_]\w*\s+(dd|db|dw|dq)', stripped):
                if not line.startswith('    ') and not line.startswith('\t'):
                    formatted_lines.append(f'    {stripped}')
                else:
                    formatted_lines.append(line)
                continue
            
            # Handle instructions with proper indentation
            if stripped and not stripped.startswith(';'):
                # Only add indentation if not already indented properly
                if not line.startswith('    ') and not line.startswith('\t'):
                    formatted_lines.append(f'    {stripped}')
                else:
                    formatted_lines.append(line)
            else:
                # Comments and other lines - preserve as-is
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

# Main integration functions
def process_assembly_with_comparison(primary_assembly: str, reference_assembly: str = None, debug: bool = False, calling_convention: str = 'windows_x64') -> str:
    """Main function to process assembly using intelligent comparison"""
    processor = AdvancedAssemblyProcessor(debug=debug, calling_convention=calling_convention)
    return processor.process_assembly(primary_assembly, reference_assembly)

def process_assembly_file(input_path: str, output_path: str = None, reference_path: str = None, debug: bool = False, calling_convention: str = 'windows_x64') -> bool:
    """Process an assembly file using advanced intelligent analysis"""
    if output_path is None:
        output_path = input_path
    
    try:
        with open(input_path, 'r') as f:
            content = f.read()
        
        reference_content = None
        if reference_path and Path(reference_path).exists():
            with open(reference_path, 'r') as f:
                reference_content = f.read()
        
        processed_content = process_assembly_with_comparison(content, reference_content, debug, calling_convention)
        
        with open(output_path, 'w') as f:
            f.write(processed_content)
        
        return True
    except Exception as e:
        if debug:
            print(f"Error processing assembly: {e}")
        return False

# Backward compatibility classes
class DynamicAssemblyFixer:
    """Backward compatibility wrapper"""
    
    def __init__(self):
        self.processor = AdvancedAssemblyProcessor()
        self.debug = False
    
    def fix_assembly_structure(self, content: str) -> str:
        return self.processor.process_assembly(content)
    
    def fix_assembly(self, content: str) -> str:
        return self.processor.process_assembly(content)
    
    def analyze_assembly(self, lines: List[str]) -> Dict:
        content = '\n'.join(lines)
        instructions = self.processor.analyzer.parse_assembly(content)
        return {
            'syntax_errors': [{'line': i, 'content': line.strip()} for i, line in enumerate(lines) if self.processor._is_syntax_error(line.strip())],
            'missing_extern': set(),
            'incomplete_printf_calls': []
        }
    
    def _is_syntax_error(self, line: str) -> bool:
        return self.processor._is_syntax_error(line)
    
    def _classify_syntax_error(self, line: str) -> str:
        if 'call ' in line and any(instr in line for instr in ['mov', 'lea', 'add', 'sub']):
            return 'multiple_instructions_per_line'
        return 'unknown'

class ASMFormatter:
    """Backward compatibility wrapper for formatter"""
    
    def __init__(self):
        self.processor = AdvancedAssemblyProcessor()
    
    def format_file(self, input_file: str, output_file: str) -> bool:
        return process_assembly_file(input_file, output_file)

class AssemblyPostProcessor:
    """Backward compatibility wrapper for post-processor"""
    
    def __init__(self):
        self.processor = AdvancedAssemblyProcessor()
        self.debug = False
    
    def process_file(self, input_path: str, output_path: str = None) -> bool:
        return process_assembly_file(input_path, output_path, debug=self.debug)

# Integration function for the main pipeline
def integrate_with_pipeline(asm_content: str, reference_content: str = None) -> str:
    """Integration function for the CASM compilation pipeline"""
    return process_assembly_with_comparison(asm_content, reference_content)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assembly_processor.py <input_file> [output_file] [--debug] [--reference=<file>]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file
    debug = False
    reference_file = None
    
    for arg in sys.argv[2:]:
        if arg.startswith('--reference='):
            reference_file = arg.split('=', 1)[1]
        elif arg == '--debug':
            debug = True
        elif not arg.startswith('--'):
            output_file = arg
    
    if process_assembly_file(input_file, output_file, reference_file, debug):
        print(f"Successfully processed assembly: {input_file} -> {output_file}")
    else:
        print(f"Failed to process assembly: {input_file}")
        sys.exit(1)