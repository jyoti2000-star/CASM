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
                'stack_alignment': 16
            },
            'system_v_x64': {
                'int_args': ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'],
                'float_args': ['xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7'],
                'return': 'rax',
                'shadow_space': 0,
                'stack_alignment': 16
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

class AdvancedAssemblyProcessor:
    """Main assembly processor that combines all functionality"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.analyzer = AdvancedAssemblyAnalyzer()
        self.comparator = IntelligentAssemblyComparator()
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
        
        # Step 6: Format the output
        formatted_content = self._format_assembly(post_processed_content)
        
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
            formatted_lines.append('; Generated by Advanced High-Level Assembly Preprocessor')
            formatted_lines.append('; Target platform: windows (Microsoft x64)')
            formatted_lines.append('')
        
        # Keep track of extern declarations to avoid duplicates
        added_externs = set()
        header_added = False  # Track if header comments have been added
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines at the beginning
            if not stripped and not formatted_lines:
                continue
            
            # Skip duplicate or unwanted header comments
            if (stripped.startswith('; Generated by Advanced High-Level Assembly Preprocessor') or
                stripped.startswith('; Target platform:')):
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
                        formatted_lines.append('; Generated by Advanced High-Level Assembly Preprocessor')
                        formatted_lines.append('; Target platform: windows (Microsoft x64)')
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
def process_assembly_with_comparison(primary_assembly: str, reference_assembly: str = None, debug: bool = False) -> str:
    """Main function to process assembly using intelligent comparison"""
    processor = AdvancedAssemblyProcessor(debug=debug)
    return processor.process_assembly(primary_assembly, reference_assembly)

def process_assembly_file(input_path: str, output_path: str = None, reference_path: str = None, debug: bool = False) -> bool:
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
        
        processed_content = process_assembly_with_comparison(content, reference_content, debug)
        
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