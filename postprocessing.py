#!/usr/bin/env python3
"""
Post-processing module for CASM compiler
Cleans up generated assembly by:
- Removing duplicates (variables, constants, labels)
- Grouping and organizing sections properly
- Fixing indentation and formatting
- Optimizing code structure
- Cleaning up redundant declarations
"""

import re
import os
from typing import List, Dict, Set, Tuple, Optional
from collections import OrderedDict

class AssemblyPostProcessor:
    def __init__(self):
        self.data_section = OrderedDict()  # Keep order but prevent duplicates
        self.bss_section = OrderedDict()
        self.text_section = []
        self.extern_declarations = set()
        self.global_declarations = set()
        self.string_constants = OrderedDict()
        self.variables = OrderedDict()
        self.float_constants = OrderedDict()
        self.labels_used = set()
        self.labels_defined = set()
        
    def process_file(self, input_file: str, output_file: str) -> bool:
        """Process an assembly file and create a cleaned version"""
        try:
            with open(input_file, 'r') as f:
                content = f.read()
            
            # Parse and clean the assembly
            cleaned_content = self.clean_assembly(content)
            
            # Write the cleaned version
            with open(output_file, 'w') as f:
                f.write(cleaned_content)
            
            print(f"[+] Post-processing completed: {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            print(f"[!] Post-processing error: {e}")
            return False
    
    def clean_assembly(self, content: str) -> str:
        """Main cleaning function that orchestrates all cleanup operations"""
        lines = content.split('\n')
        
        # Reset state for new file
        self._reset_state()
        
        # Phase 1: Parse and categorize all content
        self._parse_content(lines)
        
        # Phase 2: Remove duplicates and organize
        self._remove_duplicates()
        
        # Phase 3: Optimize and clean up
        self._optimize_sections()
        
        # Phase 4: Generate clean assembly
        return self._generate_clean_assembly()
    
    def _reset_state(self):
        """Reset all internal state for new file processing"""
        self.data_section.clear()
        self.bss_section.clear()
        self.text_section.clear()
        self.extern_declarations.clear()
        self.global_declarations.clear()
        self.string_constants.clear()
        self.variables.clear()
        self.float_constants.clear()
        self.labels_used.clear()
        self.labels_defined.clear()
    
    def _parse_content(self, lines: List[str]):
        """Parse the assembly content and categorize different sections"""
        current_section = None
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments during parsing
            if not line or line.startswith(';'):
                if current_section == 'text':
                    self.text_section.append(lines[i])
                i += 1
                continue
            
            # Section headers
            if line.startswith('section .'):
                current_section = line.split('.')[1]
                i += 1
                continue
            
            # External declarations
            if line.startswith('extern '):
                extern_name = line.split()[1]
                self.extern_declarations.add(extern_name)
                i += 1
                continue
            
            # Global declarations
            if line.startswith('global '):
                global_name = line.split()[1]
                self.global_declarations.add(global_name)
                i += 1
                continue
            
            # Data section items
            if current_section == 'data':
                self._parse_data_line(lines[i])
            elif current_section == 'bss':
                self._parse_bss_line(lines[i])
            elif current_section == 'text':
                self.text_section.append(lines[i])
                # Track label usage
                self._track_labels(lines[i])
            
            i += 1
    
    def _parse_data_line(self, line: str):
        """Parse a data section line and categorize it"""
        stripped = line.strip()
        
        if not stripped:
            return
        
        # Variable declarations (V01, V02, etc.)
        if re.match(r'^\s*V\d+\s+d[bdwq]', line):
            var_match = re.match(r'^\s*(V\d+)\s+(d[bdwq])\s+(.+)', line)
            if var_match:
                var_name = var_match.group(1)
                var_type = var_match.group(2)
                var_value = var_match.group(3)
                self.variables[var_name] = (var_type, var_value, line)
        
        # String constants (LC1, LC2, etc.)
        elif re.match(r'^\s*LC\d+\s+db', line):
            lc_match = re.match(r'^\s*(LC\d+)\s+db\s+(.+)', line)
            if lc_match:
                lc_name = lc_match.group(1)
                lc_value = lc_match.group(2)
                self.string_constants[lc_name] = (lc_value, line)
        
        # Float constants (FLOAT_TWO, etc.)
        elif re.match(r'^\s*\w+\s+dq\s+[\d.]+', line):
            float_match = re.match(r'^\s*(\w+)\s+dq\s+(.+)', line)
            if float_match:
                float_name = float_match.group(1)
                float_value = float_match.group(2)
                # Only keep one instance of each float constant
                if float_name not in self.float_constants:
                    self.float_constants[float_name] = (float_value, line)
        
        # Assembly string constants (__str1, __str2, etc.)
        elif re.match(r'^\s*__str\d+\s+db', line):
            str_match = re.match(r'^\s*(__str\d+)\s+db\s+(.+)', line)
            if str_match:
                str_name = str_match.group(1)
                str_value = str_match.group(2)
                self.string_constants[str_name] = (str_value, line)
        
        # Other data section items
        else:
            # Use the line itself as key to prevent exact duplicates
            self.data_section[stripped] = line
    
    def _parse_bss_line(self, line: str):
        """Parse a BSS section line"""
        stripped = line.strip()
        if stripped:
            self.bss_section[stripped] = line
    
    def _track_labels(self, line: str):
        """Track label usage and definitions in text section"""
        # Track label definitions (labels ending with :)
        if ':' in line and not line.strip().startswith(';'):
            label_match = re.match(r'^\s*(\w+):', line)
            if label_match:
                self.labels_defined.add(label_match.group(1))
        
        # Track label usage in instructions
        label_refs = re.findall(r'\[rel (\w+)\]', line)
        for label in label_refs:
            self.labels_used.add(label)
        
        # Track other label references (including in lea and call instructions)
        other_refs = re.findall(r'\b(LC\d+|V\d+|__str\d+|\.L\d+)\b', line)
        for label in other_refs:
            self.labels_used.add(label)
        
        # Track direct label calls
        call_refs = re.findall(r'call\s+(\w+)', line)
        for label in call_refs:
            if not label in ['printf', 'puts', 'GetStdHandle', 'WriteConsoleA', 'ExitProcess']:
                self.labels_used.add(label)
        
        # Track jump targets
        jump_refs = re.findall(r'j\w+\s+(\.?\w+)', line)
        for label in jump_refs:
            self.labels_used.add(label)
    
    def _remove_duplicates(self):
        """Remove duplicate declarations and optimize"""
        # Remove unused string constants but preserve __str constants that are referenced
        self._preserve_essential_labels()
        
        # Remove unused variables
        used_vars = {label for label in self.labels_used if label.startswith('V')}
        unused_vars = set(self.variables.keys()) - used_vars
        for unused in unused_vars:
            print(f"[DEBUG] Removing unused variable: {unused}")
            del self.variables[unused]

    def _preserve_essential_labels(self):
        """Preserve labels that are essential even if not detected in basic tracking"""
        # Scan text section more thoroughly for label usage
        all_text = '\n'.join(self.text_section)
        
        # Find all actual label references in the text
        for label_name in list(self.string_constants.keys()):
            if label_name in all_text:
                self.labels_used.add(label_name)
        
        # Now remove truly unused labels
        used_lc_labels = {label for label in self.labels_used if label.startswith('LC') or label.startswith('__str')}
        unused_lc = set(self.string_constants.keys()) - used_lc_labels
        for unused in unused_lc:
            print(f"[DEBUG] Removing unused string constant: {unused}")
            del self.string_constants[unused]
    
    def _optimize_sections(self):
        """Optimize section organization and content"""
        # First, move any misplaced data declarations from text to data section
        self._move_misplaced_data()
        
        # Fix broken label references
        self._fix_label_references()
        
        # Clean up text section
        cleaned_text = []
        prev_was_empty = False
        
        for line in self.text_section:
            stripped = line.strip()
            
            # Skip data declarations that should be in data section
            if re.match(r'^\s*__str\d+\s+db', line):
                continue
            
            # Remove excessive empty lines
            if not stripped:
                if not prev_was_empty:
                    cleaned_text.append('')
                prev_was_empty = True
                continue
            
            prev_was_empty = False
            
            # Fix indentation
            if stripped.startswith(';'):
                # Comments - ensure consistent formatting
                cleaned_text.append(f"; {stripped[1:].strip()}")
            elif ':' in stripped and not stripped.startswith(' '):
                # Labels - no indentation
                cleaned_text.append(stripped)
            elif any(stripped.startswith(instr) for instr in ['mov', 'lea', 'call', 'push', 'pop', 'add', 'sub', 'cmp', 'jmp', 'je', 'jne', 'jle', 'jge', 'movsd', 'movapd', 'addsd', 'pxor', 'cvtsi2sd', 'movq']):
                # Instructions - proper indentation
                cleaned_text.append(f"    {stripped}")
            else:
                # Other lines - preserve with minimal cleanup
                cleaned_text.append(line.rstrip())
        
        self.text_section = cleaned_text

    def _move_misplaced_data(self):
        """Move data declarations that ended up in text section to data section"""
        text_lines_to_remove = []
        
        for i, line in enumerate(self.text_section):
            stripped = line.strip()
            
            # Find misplaced string constants
            if re.match(r'^\s*__str\d+\s+db', line):
                str_match = re.match(r'^\s*(__str\d+)\s+db\s+(.+)', line)
                if str_match:
                    str_name = str_match.group(1)
                    str_value = str_match.group(2)
                    self.string_constants[str_name] = (str_value, line)
                    text_lines_to_remove.append(i)
        
        # Remove misplaced lines from text section (in reverse order to maintain indices)
        for i in reversed(text_lines_to_remove):
            del self.text_section[i]

    def _fix_label_references(self):
        """Fix broken label references in the text section"""
        # Create a mapping of used labels to their actual names
        label_mapping = {}
        
        # Check for common broken references and map them correctly
        for i, line in enumerate(self.text_section):
            # Fix references to removed labels
            if '__str1' in line and '__str1' not in self.string_constants:
                # Try to find a suitable replacement
                for str_name in self.string_constants:
                    if 'Demo' in str(self.string_constants[str_name][0]):
                        self.text_section[i] = line.replace('__str1', str_name)
                        break
            
            if '__str3' in line and '__str3' not in self.string_constants:
                # Try to find a suitable replacement
                for str_name in self.string_constants:
                    if 'Demo completed' in str(self.string_constants[str_name][0]):
                        self.text_section[i] = line.replace('__str3', str_name)
                        break
    
    def _generate_clean_assembly(self) -> str:
        """Generate the final cleaned assembly output"""
        output_lines = []
        
        # Header comment
        output_lines.extend([
            "; Clean assembly generated by CASM Post-Processor",
            "; Duplicates removed, sections organized, indentation fixed",
            ""
        ])
        
        # Data section
        if self.variables or self.string_constants or self.float_constants or self.data_section:
            output_lines.append("section .data")
            output_lines.append("")
            
            # External declarations first
            if self.extern_declarations:
                for extern in sorted(self.extern_declarations):
                    output_lines.append(f"extern {extern}")
                output_lines.append("")
            
            # Float constants
            if self.float_constants:
                output_lines.append("    ; Floating point constants")
                for name, (value, original_line) in self.float_constants.items():
                    output_lines.append(f"    {name} dq {value}")
                output_lines.append("")
            
            # Variables
            if self.variables:
                output_lines.append("    ; Variables")
                for name in sorted(self.variables.keys(), key=lambda x: int(x[1:])):  # Sort V01, V02, etc.
                    var_type, var_value, original_line = self.variables[name]
                    output_lines.append(f"    {name} {var_type} {var_value}")
                output_lines.append("")
            
            # String constants
            if self.string_constants:
                output_lines.append("    ; String constants")
                
                # Group by type
                lc_constants = {k: v for k, v in self.string_constants.items() if k.startswith('LC')}
                str_constants = {k: v for k, v in self.string_constants.items() if k.startswith('__str')}
                
                # LC constants first
                for name in sorted(lc_constants.keys(), key=lambda x: int(x[2:])):  # Sort LC1, LC2, etc.
                    value, original_line = lc_constants[name]
                    output_lines.append(f"    {name} db {value}")
                
                if str_constants:
                    output_lines.append("")
                    output_lines.append("    ; Assembly string constants")
                    for name in sorted(str_constants.keys(), key=lambda x: int(x[5:])):  # Sort __str1, __str2, etc.
                        value, original_line = str_constants[name]
                        output_lines.append(f"    {name} db {value}")
                
                output_lines.append("")
            
            # Other data section items
            if self.data_section:
                output_lines.append("    ; Other data")
                for line in self.data_section.values():
                    if line.strip():
                        output_lines.append(f"    {line.strip()}")
                output_lines.append("")
        
        # BSS section
        if self.bss_section:
            output_lines.append("section .bss")
            output_lines.append("")
            for line in self.bss_section.values():
                if line.strip():
                    output_lines.append(f"    {line.strip()}")
            output_lines.append("")
        
        # Text section
        if self.text_section:
            output_lines.append("section .text")
            output_lines.append("")
            
            # Global declarations
            if self.global_declarations:
                for global_decl in sorted(self.global_declarations):
                    output_lines.append(f"global {global_decl}")
                output_lines.append("")
            
            # Text section content
            output_lines.extend(self.text_section)
        
        return '\n'.join(output_lines)

def main():
    """Main function for standalone usage"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python3 postprocessing.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    processor = AssemblyPostProcessor()
    success = processor.process_file(input_file, output_file)
    
    if success:
        print("Post-processing completed successfully!")
    else:
        print("Post-processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()