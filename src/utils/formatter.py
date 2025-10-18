#!/usr/bin/env python3
import re
from typing import List, Dict, Set
from ..utils.colors import print_info, print_success, print_warning

class AssemblyFormatter:
    """Format assembly code for better readability and NASM compatibility"""
    
    def __init__(self):
        self.indent_size = 4
        self.label_column = 0
        self.instruction_column = 4
        self.comment_column = 40
        
    def format_assembly(self, assembly_code: str) -> str:
        """Format assembly code"""
        lines = assembly_code.split('\n')
        
        # First pass: clean excessive comments and group instructions
        cleaned_lines = self._clean_excessive_comments(lines)
        
        formatted_lines = []
        current_section = None
        
        for line in cleaned_lines:
            formatted_line = self._format_line(line, current_section)
            if formatted_line is not None:  # Skip None lines (filtered out)
                formatted_lines.append(formatted_line)
            
            # Track current section
            stripped = line.strip().lower()
            if stripped.startswith('section'):
                current_section = stripped.split()[1] if len(stripped.split()) > 1 else None
        
        # Add proper section spacing
        final_lines = self._add_section_spacing(formatted_lines)
        
        return '\n'.join(final_lines)
    
    def _clean_excessive_comments(self, lines: List[str]) -> List[str]:
        """Remove excessive comments but keep original C statement comments"""
        cleaned = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip excessive block comments
            if any(pattern in line for pattern in [
                '; === C CODE BLOCK ===',
                '; === END C CODE BLOCK ===',
                '; Compiled with combined GCC compilation',
                '; Generated assembly:',
                '; Variable declaration:',
                '; Complex expression:',
                '/ /',  # Remove // style comments
            ]):
                i += 1
                continue
            
            # Keep original C statement comments (they contain function calls, etc.)
            if line.startswith('; Original C:'):
                # Extract the actual C statement and create a clean comment
                c_statement = line.replace('; Original C:', '').strip()
                cleaned.append(f"    ; {c_statement}")
                
                # Skip the next few lines until we hit actual assembly instructions
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if (next_line and 
                        not next_line.startswith(';') and 
                        not next_line.startswith('/') and
                        not any(skip in next_line for skip in ['===', 'Compiled', 'Generated'])):
                        break
                    i += 1
                continue
            
            # Keep assembly instructions, labels, and directives
            if (line and 
                not line.startswith('/') and
                not any(skip in line for skip in ['===', 'Compiled', 'Generated', 'Variable declaration', 'Complex expression'])):
                cleaned.append(lines[i])
            
            i += 1
        
        return cleaned
    
    def _format_line(self, line: str, current_section: str) -> str:
        """Format a single line"""
        stripped = line.strip()
        
        # Empty lines
        if not stripped:
            return ''
        
        # Skip certain unwanted patterns
        if any(pattern in stripped for pattern in [
            '/ /',       # Remove // style comments
        ]):
            return None  # Signal to skip this line
        
        # Comments - keep them simple
        if stripped.startswith(';'):
            return f"    {stripped}"
        
        # Labels
        if stripped.endswith(':') and not any(op in stripped for op in ['mov', 'add', 'sub', 'call']):
            return stripped  # Labels start at column 0
        
        # Directives (section, extern, global, etc.)
        if any(stripped.startswith(word) for word in ['section', 'extern', 'global', 'db', 'dd', 'dq', 'dw']):
            return self._format_directive(stripped, current_section)
        
        # Instructions
        return self._format_instruction(stripped)
    
    def _format_comment(self, comment: str) -> str:
        """Format comment line"""
        if comment.startswith(';;;'):
            # Major section comment
            return f"\n{comment}\n"
        elif comment.startswith(';;'):
            # Subsection comment
            return f"\n{comment}"
        else:
            # Regular comment
            return comment
    
    def _format_label(self, label: str) -> str:
        """Format label"""
        return label  # Labels start at column 0
    
    def _format_directive(self, directive: str, current_section: str) -> str:
        """Format assembler directive"""
        parts = directive.split(None, 1)
        directive_name = parts[0].lower()
        
        if directive_name == 'section':
            return f"\n{directive}"
        elif directive_name in ['extern', 'global']:
            return directive
        elif directive_name in ['db', 'dd', 'dq', 'dw'] and current_section == '.data':
            # Data declarations in data section
            return f"    {directive}"
        else:
            return f"    {directive}"
    
    def _format_instruction(self, instruction: str) -> str:
        """Format assembly instruction"""
        # Split instruction and comment
        if ';' in instruction:
            instr_part, comment_part = instruction.split(';', 1)
            instr_part = instr_part.strip()
            comment_part = comment_part.strip()
            
            # Format with aligned comment
            formatted_instr = f"    {instr_part}"
            if len(formatted_instr) < self.comment_column:
                padding = self.comment_column - len(formatted_instr)
                return f"{formatted_instr}{' ' * padding}; {comment_part}"
            else:
                return f"{formatted_instr}  ; {comment_part}"
        else:
            return f"    {instruction}"
    
    def _add_section_spacing(self, lines: List[str]) -> List[str]:
        """Add proper spacing between sections"""
        result = []
        prev_line_type = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            current_line_type = self._get_line_type(stripped)
            
            # Add spacing before sections
            if current_line_type == 'section' and prev_line_type and prev_line_type != 'empty':
                result.append('')
            
            # Add spacing after sections
            if (prev_line_type == 'section' and 
                current_line_type not in ['empty', 'section'] and 
                stripped):
                result.append('')
            
            result.append(line)
            prev_line_type = current_line_type if stripped else 'empty'
        
        return result
    
    def _get_line_type(self, line: str) -> str:
        """Determine the type of assembly line"""
        if not line:
            return 'empty'
        elif line.startswith(';'):
            return 'comment'
        elif line.startswith('section'):
            return 'section'
        elif line.startswith('extern') or line.startswith('global'):
            return 'directive'
        elif line.endswith(':'):
            return 'label'
        else:
            return 'instruction'
    
    def optimize_assembly(self, assembly_code: str) -> str:
        """Apply basic assembly optimizations"""
        lines = assembly_code.split('\n')
        optimized_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Remove redundant moves
            if line.startswith('mov') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if self._is_redundant_move(line, next_line):
                    print_info(f"Removed redundant instruction: {line}")
                    i += 2  # Skip both lines
                    continue
            
            # Remove unnecessary push/pop pairs
            if line.startswith('push') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if self._is_unnecessary_push_pop(line, next_line):
                    print_info(f"Removed unnecessary push/pop: {line}, {next_line}")
                    i += 2  # Skip both lines
                    continue
            
            optimized_lines.append(lines[i])
            i += 1
        
        return '\n'.join(optimized_lines)
    
    def _is_redundant_move(self, line1: str, line2: str) -> bool:
        """Check if two consecutive moves are redundant"""
        if not line1.startswith('mov') or not line2.startswith('mov'):
            return False
        
        # Extract operands
        try:
            parts1 = line1.split()[1].split(',')
            parts2 = line2.split()[1].split(',')
            
            src1, dst1 = parts1[1].strip(), parts1[0].strip()
            src2, dst2 = parts2[1].strip(), parts2[0].strip()
            
            # mov a, b followed by mov b, a
            return src1 == dst2 and dst1 == src2
        except:
            return False
    
    def _is_unnecessary_push_pop(self, line1: str, line2: str) -> bool:
        """Check if push/pop pair is unnecessary"""
        if not line1.startswith('push') or not line2.startswith('pop'):
            return False
        
        try:
            reg1 = line1.split()[1].strip()
            reg2 = line2.split()[1].strip()
            return reg1 == reg2
        except:
            return False
    
    def add_debug_info(self, assembly_code: str, source_file: str) -> str:
        """Add debug information to assembly"""
        lines = assembly_code.split('\n')

        # Try to detect an existing target header emitted by the transpiler
        detected_target_line = None
        for l in lines[:10]:
            if l.strip().lower().startswith('; target:'):
                detected_target_line = l.strip()
                break

        # If the assembly begins with the transpiler header, strip that block
        # to avoid duplicate headers. The transpiler header usually starts with
        # '; Generated by Advanced Assembly Transpiler' and is followed by a
        # few comment lines and a blank line.
        stripped_lines = lines
        if stripped_lines and stripped_lines[0].strip().startswith('; Generated by Advanced Assembly Transpiler'):
            # Find the first blank line after the initial header block
            cut = 0
            for i, l in enumerate(stripped_lines):
                if l.strip() == '':
                    cut = i + 1
                    break
            if cut > 0:
                stripped_lines = stripped_lines[cut:]

        # Build CASM header using detected target when available
        header = [f"; Generated by CASM from {source_file}"]
        if detected_target_line:
            header.append(detected_target_line)
        else:
            # Fallback to a generic target description
            import platform as _platform
            host = _platform.system()
            arch = _platform.machine()
            header.append(f"; Target: {host} {arch}")

        header.append("; Assembler: NASM")
        header.append("")

        return '\n'.join(header + stripped_lines)
    
    def validate_assembly(self, assembly_code: str) -> List[str]:
        """Validate assembly code for common issues"""
        issues = []
        lines = assembly_code.split('\n')
        
        defined_labels = set()
        referenced_labels = set()
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for labels
            if stripped.endswith(':') and not any(op in stripped for op in ['mov', 'add', 'sub']):
                label = stripped[:-1]
                if label in defined_labels:
                    issues.append(f"Line {i}: Duplicate label '{label}'")
                defined_labels.add(label)
            
            # Check for label references
            for word in stripped.split():
                if ':' not in word and word.isalpha() and not word.startswith(';'):
                    # Might be a label reference
                    if any(instr in stripped for instr in ['jmp', 'je', 'jne', 'call']):
                        referenced_labels.add(word)
        
        # Check for undefined labels
        undefined = referenced_labels - defined_labels
        for label in undefined:
            if not label.startswith(('printf', 'scanf', 'main')):  # Skip external functions
                issues.append(f"Undefined label reference: '{label}'")
        
        return issues

# Global instance
formatter = AssemblyFormatter()