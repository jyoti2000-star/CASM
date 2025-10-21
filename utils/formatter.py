#!/usr/bin/env python3
import re
from typing import List, Dict, Set
from casm.utils.colors import print_info, print_success, print_warning

class AssemblyFormatter:
    def __init__(self):
        self.indent_size = 4
        self.label_column = 0
        self.instruction_column = 4
        self.comment_column = 40

    def format_assembly(self, assembly_code: str) -> str:
        lines = assembly_code.split('\n')
        cleaned_lines = self._clean_excessive_comments(lines)
        formatted_lines = []
        current_section = None
        for line in cleaned_lines:
            formatted_line = self._format_line(line, current_section)
            if formatted_line is not None:
                formatted_lines.append(formatted_line)
            stripped = line.strip().lower()
            if stripped.startswith('section'):
                current_section = stripped.split()[1] if len(stripped.split()) > 1 else None
        final_lines = self._add_section_spacing(formatted_lines)
        return '\n'.join(final_lines)

    def _clean_excessive_comments(self, lines: List[str]) -> List[str]:
        cleaned = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if any(pattern in line for pattern in [
                '; === C CODE BLOCK ===',
                '; === END C CODE BLOCK ===',
                '; Compiled with combined GCC compilation',
                '; Generated assembly:',
                '; Variable declaration:',
                '; Complex expression:',
                '/ /',
            ]):
                i += 1
                continue
            if line.startswith('; Original C:'):
                c_statement = line.replace('; Original C:', '').strip()
                cleaned.append(f"    ; {c_statement}")
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if (next_line and not next_line.startswith(';') and not next_line.startswith('/') and not any(skip in next_line for skip in ['===', 'Compiled', 'Generated'])):
                        break
                    i += 1
                continue
            if (line and not line.startswith('/') and not any(skip in line for skip in ['===', 'Compiled', 'Generated', 'Variable declaration', 'Complex expression'])):
                cleaned.append(lines[i])
            i += 1
        return cleaned

    def _format_line(self, line: str, current_section: str) -> str:
        stripped = line.strip()
        if not stripped:
            return ''
        if any(pattern in stripped for pattern in ['/ /']):
            return None
        if stripped.startswith(';'):
            return f"    {stripped}"
        if stripped.endswith(':') and not any(op in stripped for op in ['mov', 'add', 'sub', 'call']):
            return stripped
        if any(stripped.startswith(word) for word in ['section', 'extern', 'global', 'db', 'dd', 'dq', 'dw']):
            return self._format_directive(stripped, current_section)
        return self._format_instruction(stripped)

    def _format_directive(self, directive: str, current_section: str) -> str:
        parts = directive.split(None, 1)
        directive_name = parts[0].lower()
        if directive_name == 'section':
            return f"\n{directive}"
        elif directive_name in ['extern', 'global']:
            return directive
        elif directive_name in ['db', 'dd', 'dq', 'dw'] and current_section == '.data':
            return f"    {directive}"
        else:
            return f"    {directive}"

    def _format_instruction(self, instruction: str) -> str:
        if ';' in instruction:
            instr_part, comment_part = instruction.split(';', 1)
            instr_part = instr_part.strip()
            comment_part = comment_part.strip()
            formatted_instr = f"    {instr_part}"
            if len(formatted_instr) < self.comment_column:
                padding = self.comment_column - len(formatted_instr)
                return f"{formatted_instr}{' ' * padding}; {comment_part}"
            else:
                return f"{formatted_instr}  ; {comment_part}"
        else:
            return f"    {instruction}"

    def _add_section_spacing(self, lines: List[str]) -> List[str]:
        result = []
        prev_line_type = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            current_line_type = self._get_line_type(stripped)
            if current_line_type == 'section' and prev_line_type and prev_line_type != 'empty':
                result.append('')
            if (prev_line_type == 'section' and current_line_type not in ['empty', 'section'] and stripped):
                result.append('')
            result.append(line)
            prev_line_type = current_line_type if stripped else 'empty'
        return result

    def _get_line_type(self, line: str) -> str:
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
        lines = assembly_code.split('\n')
        optimized_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('mov') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if self._is_redundant_move(line, next_line):
                    print_info(f"Removed redundant instruction: {line}")
                    i += 2
                    continue
            if line.startswith('push') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if self._is_unnecessary_push_pop(line, next_line):
                    print_info(f"Removed unnecessary push/pop: {line}, {next_line}")
                    i += 2
                    continue
            optimized_lines.append(lines[i])
            i += 1
        return '\n'.join(optimized_lines)

    def _is_redundant_move(self, line1: str, line2: str) -> bool:
        if not line1.startswith('mov') or not line2.startswith('mov'):
            return False
        try:
            parts1 = line1.split()[1].split(',')
            parts2 = line2.split()[1].split(',')
            src1, dst1 = parts1[1].strip(), parts1[0].strip()
            src2, dst2 = parts2[1].strip(), parts2[0].strip()
            return src1 == dst2 and dst1 == src2
        except:
            return False

    def _is_unnecessary_push_pop(self, line1: str, line2: str) -> bool:
        if not line1.startswith('push') or not line2.startswith('pop'):
            return False
        try:
            reg1 = line1.split()[1].strip()
            reg2 = line2.split()[1].strip()
            return reg1 == reg2
        except:
            return False

    def add_debug_info(self, assembly_code: str, source_file: str) -> str:
        lines = assembly_code.split('\n')
        detected_target_line = None
        for l in lines[:10]:
            if l.strip().lower().startswith('; target:'):
                detected_target_line = l.strip()
                break
        stripped_lines = lines
        if stripped_lines and stripped_lines[0].strip().startswith('; Generated by Advanced Assembly Transpiler'):
            cut = 0
            for i, l in enumerate(stripped_lines):
                if l.strip() == '':
                    cut = i + 1
                    break
            if cut > 0:
                stripped_lines = stripped_lines[cut:]
        header = [f"; Generated by CASM from {source_file}"]
        if detected_target_line:
            header.append(detected_target_line)
        else:
            import platform as _platform
            host = _platform.system()
            arch = _platform.machine()
            header.append(f"; Target: {host} {arch}")
        header.append("; Assembler: NASM")
        header.append("")
        return '\n'.join(header + stripped_lines)

    def validate_assembly(self, assembly_code: str) -> List[str]:
        issues = []
        lines = assembly_code.split('\n')
        defined_labels = set()
        referenced_labels = set()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.endswith(':') and not any(op in stripped for op in ['mov', 'add', 'sub']):
                label = stripped[:-1]
                if label in defined_labels:
                    issues.append(f"Line {i}: Duplicate label '{label}'")
                defined_labels.add(label)
            for word in stripped.split():
                if ':' not in word and word.isalpha() and not word.startswith(';'):
                    if any(instr in stripped for instr in ['jmp', 'je', 'jne', 'call']):
                        referenced_labels.add(word)
        undefined = referenced_labels - defined_labels
        for label in undefined:
            if not label.startswith(('printf', 'scanf', 'main')):
                issues.append(f"Undefined label reference: '{label}'")
        return issues

# Global instance
formatter = AssemblyFormatter()
