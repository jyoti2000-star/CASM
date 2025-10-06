#!/usr/bin/env python3

import sys
import os
import re
from typing import List, Optional

class ASMFormatter:
    def __init__(self):
        self.data_section = []
        self.bss_section = []
        self.text_section = []
    
    def format_file(self, input_file: str, output_file: str) -> bool:
        """Format an assembly file with proper section headers"""
        try:
            with open(input_file, 'r') as f:
                content = f.read()
            
            # Reset sections for new file
            self.data_section = []
            self.bss_section = []
            self.text_section = []
            
            # Parse the content
            self._parse_content(content)
            
            # Generate formatted output
            formatted_content = self._generate_formatted_output()
            
            # Write to output file
            with open(output_file, 'w') as f:
                f.write(formatted_content)
            
            print(f"Successfully formatted {input_file} -> {output_file}")
            print(f"Data section: {len(self.data_section)} lines")
            print(f"BSS section: {len(self.bss_section)} lines")
            print(f"Text section: {len(self.text_section)} lines")
            return True
            
        except Exception as e:
            print(f"Error formatting file: {str(e)}")
            return False
    
    def _parse_content(self, content: str) -> None:
        """Parse content and separate into sections"""
        lines = content.split('\n')
        current_section = 'text'  # Default to text section
        in_data_section = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for custom data markers BEFORE removing comments
            if stripped == '; DATA START':
                in_data_section = True
                current_section = 'data'
                continue
            elif stripped == '; DATA END':
                in_data_section = False
                current_section = 'text'
                continue
            
            # Remove comments and strip whitespace
            cleaned_line = self._remove_comments(line)
            stripped_cleaned = cleaned_line.strip()
            
            # Skip empty lines
            if not stripped_cleaned:
                continue
            
            # Check for existing section headers (in case they're already there)
            if stripped_cleaned.startswith('section .'):
                if 'data' in stripped_cleaned:
                    current_section = 'data'
                elif 'bss' in stripped_cleaned:
                    current_section = 'bss'
                elif 'text' in stripped_cleaned:
                    current_section = 'text'
                continue
            
            # Force classification based on current data section markers
            if in_data_section:
                # Everything between ; DATA START and ; DATA END goes to data section
                self.data_section.append(cleaned_line)
            elif self._is_bss_line(cleaned_line):
                if current_section != 'bss':
                    current_section = 'bss'
                self.bss_section.append(cleaned_line)
            else:
                # Everything else goes to text section
                if current_section != 'text':
                    current_section = 'text'
                self.text_section.append(cleaned_line)
    
    def _remove_comments(self, line: str) -> str:
        """Remove comments from assembly line"""
        # Find semicolon that's not inside quotes
        in_quotes = False
        quote_char = None
        
        for i, char in enumerate(line):
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char == ';' and not in_quotes:
                return line[:i].rstrip()
        
        return line
    
    def _unindent_line(self, line: str) -> str:
        """Remove unnecessary indentation from assembly line"""
        stripped = line.strip()
        if not stripped:
            return ""
        
        # Labels should not be indented
        if stripped.endswith(':'):
            return stripped
        
        # Directives and declarations should not be indented
        if any(stripped.startswith(directive) for directive in ['section', 'global', 'extern', 'dd', 'db', 'dw', 'dq', 'dt', 'resb', 'resw', 'resd', 'resq']):
            return stripped
        
        # Instructions should have minimal indentation (4 spaces)
        if stripped:
            return f"    {stripped}"
        
        return stripped
    
    def _is_block_separator(self, line: str) -> bool:
        """Check if line represents a block separator (label or significant instruction)"""
        stripped = line.strip()
        
        # Labels are block separators
        if stripped.endswith(':'):
            return True
        
        # Function calls are block separators
        if 'call ' in stripped:
            return True
        
        # Jumps are block separators
        if any(stripped.startswith(jmp) for jmp in ['jmp ', 'je ', 'jne ', 'jl ', 'jle ', 'jg ', 'jge ', 'jz ', 'jnz ']):
            return True
        
        return False
    
    def _add_spacing_to_section(self, lines: List[str]) -> List[str]:
        """Add spacing between logical blocks in a section"""
        if not lines:
            return lines
        
        result = []
        prev_was_separator = False
        i = 0
        
        while i < len(lines):
            current_line = self._unindent_line(lines[i])
            
            # Skip empty lines at the beginning
            if not current_line and not result:
                i += 1
                continue
            
            # Check for V/LC label patterns that should be combined with next line
            if self._should_combine_with_next_line(current_line, lines, i):
                # Combine label with next line
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                combined = self._combine_label_with_data(current_line, next_line)
                
                is_separator = self._is_block_separator(combined)
                
                # Add spacing before block separators (except the first line)
                if is_separator and result and not prev_was_separator:
                    result.append("")
                
                result.append(combined)
                prev_was_separator = is_separator
                i += 2  # Skip both lines since we combined them
                continue
            
            is_separator = self._is_block_separator(current_line)
            
            # Add spacing before block separators (except the first line)
            if is_separator and result and not prev_was_separator:
                result.append("")
            
            result.append(current_line)
            prev_was_separator = is_separator
            i += 1
        
        return result
    
    def _should_combine_with_next_line(self, current_line: str, lines: List[str], index: int) -> bool:
        """Check if current line should be combined with the next line"""
        stripped = current_line.strip()
        
        # Check for V/LC labels
        if re.match(r'^(V|LC)\d+:$', stripped):
            # Check if next line contains data directive
            if index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if any(directive in next_line for directive in ['dd ', 'db ', 'dw ', 'dq ', 'dt ']):
                    return True
        
        return False
    
    def _combine_label_with_data(self, label_line: str, data_line: str) -> str:
        """Combine label with data directive on single line"""
        label = label_line.strip().rstrip(':')  # Remove colon
        data = data_line.strip()
        
        # Format with proper indentation like __str1
        return f"    {label} {data}"
    
    def _has_content_in_current_section(self, section: str) -> bool:
        """Check if current section has any content"""
        if section == 'data':
            return len(self.data_section) > 0
        elif section == 'bss':
            return len(self.bss_section) > 0
        else:
            return len(self.text_section) > 0
    
    def _is_data_line(self, line: str) -> bool:
        """Check if line belongs to data section"""
        stripped = line.strip()
        
        # Data directives
        if any(directive in stripped for directive in ['dd ', 'db ', 'dw ', 'dq ', 'dt ']):
            return True
        
        # String constants and variable labels (LC labels and V labels)
        if re.match(r'^(LC|V)\d+:', stripped):
            return True
        
        # String labels (starting with __ and ending with digits)
        if re.match(r'^__str\d+\s+db', stripped):
            return True
        
        # Check for data-like patterns (label followed by data directive on same line)
        if ':' in stripped:
            # Get the next non-empty part after the colon
            parts = stripped.split(':', 1)
            if len(parts) > 1:
                after_colon = parts[1].strip()
                # If there's data directive after colon, it's data
                if any(directive in after_colon for directive in ['dd ', 'db ', 'dw ', 'dq ', 'dt ']):
                    return True
                # If it's a label followed by an instruction (like add, mov, etc.), it's NOT data
                if any(instr in after_colon for instr in ['add ', 'mov ', 'sub ', 'mul ', 'div ', 'cmp ', 'jmp ', 'call ', 'push ', 'pop ', 'lea ', 'xor ', 'and ', 'or ']):
                    return False
                # If it's empty after colon but the label looks like data (V, LC, __str)
                label_name = parts[0].strip()
                if re.match(r'^(LC|V)\d+$', label_name) or re.match(r'^__str\d+$', label_name):
                    return True
        
        return False
    
    def _is_bss_line(self, line: str) -> bool:
        """Check if line belongs to BSS section (uninitialized data)"""
        stripped = line.strip()
        
        # BSS directives
        if any(directive in stripped for directive in ['resb ', 'resw ', 'resd ', 'resq ', 'rest ']):
            return True
        
        return False
    
    def _generate_formatted_output(self) -> str:
        """Generate properly formatted assembly with section headers and spacing"""
        output_lines = []
        
        # Data section
        if self.data_section:
            output_lines.append('section .data')
            formatted_data = self._add_spacing_to_section(self.data_section)
            output_lines.extend(formatted_data)
            output_lines.append('')  # Empty line after section
        
        # BSS section
        if self.bss_section:
            output_lines.append('section .bss')
            formatted_bss = self._add_spacing_to_section(self.bss_section)
            output_lines.extend(formatted_bss)
            output_lines.append('')  # Empty line after section
        
        # Text section
        if self.text_section:
            output_lines.append('section .text')
            formatted_text = self._add_spacing_to_section(self.text_section)
            output_lines.extend(formatted_text)
        
        return '\n'.join(output_lines)
    
    def format_directory(self, directory: str, pattern: str = "*.asm") -> None:
        """Format all assembly files in a directory"""
        import glob
        
        files = glob.glob(os.path.join(directory, pattern))
        
        for file_path in files:
            if file_path.endswith('.asm'):
                base_name = os.path.splitext(file_path)[0]
                output_file = f"{base_name}_formatted.asm"
                self.format_file(file_path, output_file)
                # Reset sections for next file
                self.data_section = []
                self.bss_section = []
                self.text_section = []

# Command line interface
if __name__ == "__main__":
    formatter = ASMFormatter()
    
    if len(sys.argv) < 2:
        print("Usage: python3 formatter.py <command> [args]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "format":
        if len(sys.argv) < 3:
            print("Usage: python3 formatter.py format <input_file> [output_file]")
            sys.exit(1)
        
        input_file = sys.argv[2]
        
        if len(sys.argv) >= 4:
            output_file = sys.argv[3]
        else:
            # Generate output filename by adding _formatted before extension
            base_name = os.path.splitext(input_file)[0]
            ext = os.path.splitext(input_file)[1]
            output_file = f"{base_name}_formatted{ext}"
        
        success = formatter.format_file(input_file, output_file)
        sys.exit(0 if success else 1)
    
    elif command == "dir":
        if len(sys.argv) < 3:
            print("Usage: python3 formatter.py dir <directory> [pattern]")
            sys.exit(1)
        
        directory = sys.argv[2]
        pattern = sys.argv[3] if len(sys.argv) >= 4 else "*.asm"
        
        formatter.format_directory(directory, pattern)
        sys.exit(0)
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: format, dir")
        sys.exit(1)