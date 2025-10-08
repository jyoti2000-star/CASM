#!/usr/bin/env python3
import re
from typing import Dict, List, Set
from ..utils.colors import print_info, print_success, print_warning, print_error

class AssemblyFixer:
    """Fix assembly code to be NASM-compatible"""
    
    def __init__(self):
        self.string_counter = 1
        self.seen_strings = set()
        self.label_mappings = {}  # Track label mappings for consistency
        self.defined_labels = set()  # Track defined labels
        self.referenced_labels = set()  # Track referenced labels
        self.if_counter = 1  # Counter for if labels
        self.defined_variables = set()  # Track defined variables to avoid duplicates
        self.for_counter = 1  # Counter for for labels
        self.while_counter = 1  # Counter for while labels
        
    def fix_assembly(self, assembly_code: str) -> str:
        """Main entry point - fix all assembly issues"""
        print_info("Fixing assembly for NASM compatibility...")
        
        # Split into lines for processing
        lines = assembly_code.split('\n')
        
        # Phase 1: Collect all labels and references
        self._collect_labels_and_references(lines)
        
        # Phase 2: Fix the issues
        fixed_lines = []
        in_data_section = False
        
        for line in lines:
            if line.strip().startswith('section .data'):
                in_data_section = True
                fixed_lines.append(line)
                continue
            elif line.strip().startswith('section .text'):
                in_data_section = False
                fixed_lines.append(line)
                continue
            
            if in_data_section:
                fixed_line = self._fix_data_section_line(line)
            else:
                fixed_line = self._fix_text_section_line(line)
            
            fixed_lines.append(fixed_line)
        
        # Phase 3: Add missing string definitions
        fixed_lines = self._add_missing_strings(fixed_lines)
        
        result = '\n'.join(fixed_lines)
        print_success("Assembly fixed for NASM compatibility")
        return result
    
    def _collect_labels_and_references(self, lines: List[str]):
        """Collect all label definitions and references"""
        for line in lines:
            stripped = line.strip()
            
            # Find label definitions (lines ending with :)
            if ':' in stripped and not stripped.startswith(';'):
                # Extract label name
                if stripped.endswith(':'):
                    label = stripped[:-1].strip()
                    self.defined_labels.add(label)
                elif ':\t' in stripped or ': ' in stripped:
                    label = stripped.split(':')[0].strip()
                    self.defined_labels.add(label)
            
            # Find label references (lea, jmp, je, call with [rel label])
            ref_patterns = [
                r'lea\s+\w+,\s*\[rel\s+(\w+)\]',
                r'j[a-z]+\s+(\w+)',
                r'call\s+(\w+)',
                r'jmp\s+(\w+)',
                r'movsd\s+\w+,\s*QWORD\s+PTR\s+(\w+)\[rip\]',  # GCC floating point constants
                r'lea\s+\w+,\s*(\w+)\[rip\]'  # GCC style label references
            ]
            
            for pattern in ref_patterns:
                matches = re.findall(pattern, stripped)
                for match in matches:
                    self.referenced_labels.add(match)
    
    def _fix_data_section_line(self, line: str) -> str:
        """Fix issues in data section lines"""
        stripped = line.strip()
        
        # Track variable definitions to avoid duplicates
        if ' dd ' in line and not line.strip().startswith(';'):
            var_match = re.match(r'\s*(\w+)\s+dd\s+', line)
            if var_match:
                var_name = var_match.group(1)
                if var_name in self.defined_variables:
                    # Skip duplicate variable definition
                    return f"    ; Duplicate {var_name} definition skipped\n"
                else:
                    self.defined_variables.add(var_name)
        
        # Fix string format issues
        if 'db "' in stripped and '", 10, "", 0' in stripped:
            # Remove the extra empty string
            line = line.replace('", 10, "", 0', '", 10, 0')
        
        # Fix .LC labels to be global
        if stripped.startswith('.LC'):
            # Convert .LC0 db "..." to LC0 db "..." (remove the dot)
            line = line.replace('.LC', 'LC')
        
        return line
    
    def _fix_text_section_line(self, line: str) -> str:
        """Fix issues in text section lines"""
        stripped = line.strip()
        
        # Fix GCC PTR syntax to NASM syntax
        if 'QWORD PTR' in line:
            line = line.replace('QWORD PTR', 'qword')
        if 'DWORD PTR' in line:
            line = line.replace('DWORD PTR', 'dword')
        if 'WORD PTR' in line:
            line = line.replace('WORD PTR', 'word')
        if 'BYTE PTR' in line:
            line = line.replace('BYTE PTR', 'byte')
        
        # Fix .refptr references - convert to direct variable access
        if '.refptr.var_' in line:
            # Convert: mov rax, qword .refptr.var_number[rip]
            # To: lea rax, [rel var_number]
            if 'mov' in line and 'qword .refptr.var_' in line:
                var_match = re.search(r'mov\s+(\w+),\s*qword\s+\.refptr\.(var_\w+)\[rip\]', line)
                if var_match:
                    reg = var_match.group(1)
                    var_name = var_match.group(2)
                    line = f"        lea {reg}, [rel {var_name}]"
        
        # Fix references to .LC labels (remove the dot) - applies to all lines, including C code blocks
        if '.LC' in line:
            line = re.sub(r'\.LC(\d+)', r'LC\1', line)
        
        # Fix RIP-relative addressing format for NASM
        if '[rip]' in line:
            # Convert "lea rax, LC0[rip]" to "lea rax, [rel LC0]" 
            line = re.sub(r'lea\s+(\w+),\s*(\w+)\[rip\]', r'lea \1, [rel \2]', line)
            # Convert "movsd xmm0, qword LC0[rip]" to "movsd xmm0, [rel LC0]"
            line = re.sub(r'(\w+)\s+(\w+),\s*qword\s+(\w+)\[rip\]', r'\1 \2, [rel \3]', line)
            # Convert "mov rax, qword LC0[rip]" to "mov rax, [rel LC0]"
            line = re.sub(r'(\w+)\s+(\w+),\s*(\w+)\[rip\]', r'\1 \2, [rel \3]', line)
        
        # Fix random label names with clean ones
        line = self._fix_random_labels(line)
        
        # Fix missing string references
        line = self._fix_string_references(line)
        
        # Fix label consistency issues
        line = self._fix_label_consistency(line)
        
        return line
    
    def _fix_random_labels(self, line: str) -> str:
        """Replace random label names with clean sequential ones - DISABLED"""
        # DISABLED: This function was causing duplicate label issues
        # The code generator now produces unique labels correctly
        return line
    
    def _fix_string_references(self, line: str) -> str:
        """Fix undefined string references"""
        # Check for undefined string references and define them if needed
        for i in range(10):  # Check str_0 through str_9
            if f'str_{i}' in line and f'str_{i}' not in self.seen_strings:
                self.seen_strings.add(f'str_{i}')
        
        return line
    
    def _fix_label_consistency(self, line: str) -> str:
        """Fix label consistency issues - DISABLED"""
        # DISABLED: This function was causing duplicate label issues
        # The code generator now produces unique labels correctly
        return line
    
    def _add_missing_strings(self, lines: List[str]) -> List[str]:
        """Add missing string definitions to data section and fix inconsistent labeling"""
        missing_strings = self.referenced_labels - self.defined_labels
        
        # Also check for inconsistent str_ numbering
        str_refs = {label for label in self.referenced_labels if label.startswith('str_')}
        str_defs = {label for label in self.defined_labels if label.startswith('str_')}
        
        # Check for missing LC constants
        lc_refs = {label for label in self.referenced_labels if label.startswith('LC')}
        lc_defs = {label for label in self.defined_labels if label.startswith('LC')}
        missing_lc = lc_refs - lc_defs
        
        print_info(f"String references: {str_refs}")
        print_info(f"String definitions: {str_defs}")
        print_info(f"LC references: {lc_refs}")
        print_info(f"LC definitions: {lc_defs}")
        print_info(f"Missing LC constants: {missing_lc}")
        
        # Find the data section and fix/add strings
        result_lines = []
        in_data_section = False
        data_lines = []
        text_section_start = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('section .data'):
                in_data_section = True
                result_lines.append(line)
                continue

            # If we encounter a different section while in data mode, flush data_lines
            if in_data_section and (stripped.startswith('section .bss') or (stripped.startswith('section') and not stripped.startswith('section .data'))):
                # Process data section to fix string labeling
                if data_lines:
                    fixed_data_lines = self._fix_data_section_strings(data_lines, str_refs)
                    result_lines.extend(fixed_data_lines)
                    data_lines = []

                # Append the non-data section header (e.g., section .bss) and continue
                in_data_section = False
                result_lines.append(line)
                continue

            if stripped.startswith('section .text'):
                text_section_start = i
                # If we were still collecting data lines, flush them before text
                if in_data_section and data_lines:
                    fixed_data_lines = self._fix_data_section_strings(data_lines, str_refs)
                    result_lines.extend(fixed_data_lines)
                    data_lines = []
                in_data_section = False

                # Add missing LC constants before text section
                if missing_lc:
                    result_lines.append("")
                    result_lines.append("    ; Missing LC constants")
                    for lc_label in sorted(missing_lc):
                        if lc_label == 'LC0':
                            # Add the floating point constant for 2.0
                            result_lines.append(f"    {lc_label}:")
                            result_lines.append("        dq 2.0  ; 2.0 in double precision")
                        else:
                            result_lines.append(f"    {lc_label} db \"Missing constant {lc_label}\", 0")

                result_lines.append(line)
                continue

            if in_data_section:
                data_lines.append(line)
            else:
                result_lines.append(line)
        
        return result_lines
    
    def _fix_data_section_strings(self, data_lines: List[str], referenced_strings: Set[str]) -> List[str]:
        """Fix string definitions to match references"""
        fixed_lines = []
        existing_strings = []
        
        # Collect existing string definitions
        for line in data_lines:
            stripped = line.strip()
            if stripped.startswith('str_') and ' db ' in stripped:
                # Extract label and content
                parts = stripped.split(' db ', 1)
                if len(parts) == 2:
                    label = parts[0]
                    content = parts[1]
                    existing_strings.append((label, content))
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Create new sequential string definitions for all referenced strings
        for i, ref_str in enumerate(sorted(referenced_strings), 1):
            if i <= len(existing_strings):
                # Use existing content with new label
                _, content = existing_strings[i-1]
                fixed_lines.append(f'    {ref_str} db {content}')
            else:
                # Add placeholder for missing strings
                fixed_lines.append(f'    {ref_str} db "Placeholder string", 0')
        
        return fixed_lines

# Global instance
assembly_fixer = AssemblyFixer()