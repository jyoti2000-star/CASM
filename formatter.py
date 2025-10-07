#!/usr/bin/env python3
"""
Advanced Assembly Formatter for CASM
Properly organizes assembly code into correct sections:
- .data section: data declarations (dd, db, dw, dq), variables, constants
- .text section: extern declarations, assembly instructions, labels, code
"""

import re
import os
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SectionType(Enum):
    """Assembly section types"""
    DATA = "data"
    TEXT = "text"
    BSS = "bss"
    RODATA = "rodata"

@dataclass
class AssemblyLine:
    """Represents a single line of assembly code"""
    content: str
    line_number: int
    section_type: SectionType
    is_label: bool = False
    is_data_declaration: bool = False
    is_extern: bool = False
    is_global: bool = False
    is_instruction: bool = False
    is_comment: bool = False
    is_empty: bool = False
    label_name: Optional[str] = None
    data_name: Optional[str] = None

class AssemblyFormatter:
    """Advanced assembly formatter that organizes code into proper sections"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.data_lines = []
        self.text_lines = []
        self.bss_lines = []
        self.header_comments = []
        self.parsed_lines = []
        self.instruction_groups = []  # Track instruction groups for spacing
        
        # Data declaration patterns
        self.data_patterns = [
            r'^\s*\w+\s*:\s*dd\s+',    # label: dd value
            r'^\s*\w+\s*:\s*db\s+',    # label: db value
            r'^\s*\w+\s*:\s*dw\s+',    # label: dw value
            r'^\s*\w+\s*:\s*dq\s+',    # label: dq value
            r'^dd\s+',                 # dd value (no indentation)
            r'^db\s+',                 # db value (no indentation)
            r'^dw\s+',                 # dw value (no indentation) 
            r'^dq\s+',                 # dq value (no indentation)
            r'^\s*dd\s+',              # dd value (with indentation)
            r'^\s*db\s+',              # db value (with indentation)
            r'^\s*dw\s+',              # dw value (with indentation)
            r'^\s*dq\s+',              # dq value (with indentation)
            r'^\s*[A-Za-z_]\w*\s*db\s+',  # variable db
            r'^\s*[A-Za-z_]\w*\s*dd\s+',  # variable dd
            r'^\s*[A-Za-z_]\w*\s*dw\s+',  # variable dw
            r'^\s*[A-Za-z_]\w*\s*dq\s+',  # variable dq
        ]
        
        # Label patterns for data section
        self.data_label_patterns = [
            r'^\s*V\d+\s*:',           # Variable labels (V01:, V02:, etc.)
            r'^\s*LC\d+\s*:',          # Literal constant labels (LC1:, LC2:, etc.)
            r'^\s*__str\d+\s*:',       # String labels (__str1:, __str2:, etc.)
        ]
        
        # Instruction patterns
        self.instruction_patterns = [
            r'^\s*(mov|lea|add|sub|mul|div|push|pop|call|ret|jmp|je|jne|jl|jle|jg|jge|ja|jae|jb|jbe|cmp|test|and|or|xor|not|shl|shr|inc|dec|imul|idiv|neg|adc|sbb)\s+',
        ]
        
        # Patterns for instruction comments to keep (original ASM statements)
        self.keep_comment_patterns = [
            r';\s*%\w+',               # ;%println, ;%if, ;%exit, etc.
            r';\s*C\s+instruction',    # C instruction comments
        ]
        
    def _clean_inline_comments(self, line: str) -> str:
        """Remove inline comments from assembly instructions"""
        stripped = line.strip()
        
        # Don't remove comments from lines that are pure comments
        if stripped.startswith(';'):
            return line
            
        # Don't remove comments from labels
        if stripped.endswith(':'):
            return line
            
        # Remove inline comments from instructions
        if ';' in stripped:
            # Split on semicolon and take only the instruction part
            instruction_part = stripped.split(';')[0].strip()
            if instruction_part:
                # Preserve original indentation
                indent = len(line) - len(line.lstrip())
                return ' ' * indent + instruction_part
        
        return line
        
    def _clean_c_instruction_comment(self, line: str) -> str:
        """Clean up C instruction comments to show just the original statement"""
        stripped = line.strip()
        
        # Match C instruction pattern: '; C instruction N: %! statement'
        match = re.match(r';\s*C\s+instruction\s+\d+:\s*(%!.*)', stripped)
        if match:
            original_statement = match.group(1)
            # Preserve original indentation
            indent = len(line) - len(line.lstrip())
            return ' ' * indent + '; ' + original_statement
            
        return line
    
    def format_assembly(self, content: str) -> str:
        """Main method to format assembly content"""
        if self.debug:
            print("[Formatter] Starting assembly formatting...")
        
        # Parse the content
        self._parse_content(content)
        
        # Organize into sections
        self._organize_sections()
        
        # Generate formatted output
        formatted_content = self._generate_formatted_output()
        
        if self.debug:
            print(f"[Formatter] Generated {len(self.data_lines)} data lines and {len(self.text_lines)} text lines")
        
        return formatted_content
    
    def _parse_content(self, content: str) -> None:
        """Parse assembly content and classify each line"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            parsed_line = self._parse_line(line, i + 1)
            self.parsed_lines.append(parsed_line)
            
            if self.debug and parsed_line.section_type != SectionType.TEXT:
                print(f"[Formatter] Line {i+1} -> {parsed_line.section_type.value}: {line.strip()}")
    
    def _parse_line(self, line: str, line_number: int) -> AssemblyLine:
        """Parse a single line and determine its characteristics"""
        stripped = line.strip()
        
        # Empty lines
        if not stripped:
            return AssemblyLine(
                content=line,
                line_number=line_number,
                section_type=SectionType.TEXT,
                is_empty=True
            )
        
        # Comments
        if stripped.startswith(';'):
            # Keep only instruction comments (original ASM statements)
            should_keep = any(re.search(pattern, stripped) for pattern in self.keep_comment_patterns)
            
            if should_keep:
                return AssemblyLine(
                    content=line,
                    line_number=line_number,
                    section_type=SectionType.TEXT,
                    is_comment=True
                )
            else:
                # Skip all other comments
                return AssemblyLine(
                    content="",  # Empty content to skip
                    line_number=line_number,
                    section_type=SectionType.TEXT,
                    is_empty=True
                )
        
        # Section declarations (skip - we'll add our own)
        if stripped.startswith('section '):
            return AssemblyLine(
                content=line,
                line_number=line_number,
                section_type=SectionType.TEXT,
                is_comment=True  # Treat as comment to skip
            )
        
        # Global declarations
        if stripped.startswith('global '):
            return AssemblyLine(
                content=line,
                line_number=line_number,
                section_type=SectionType.TEXT,
                is_global=True
            )
        
        # Extern declarations
        if stripped.startswith('extern '):
            return AssemblyLine(
                content=line,
                line_number=line_number,
                section_type=SectionType.TEXT,
                is_extern=True
            )
        
        # Check if it's an instruction FIRST before checking data patterns
        for pattern in self.instruction_patterns:
            if re.match(pattern, stripped):
                return AssemblyLine(
                    content=line,
                    line_number=line_number,
                    section_type=SectionType.TEXT,
                    is_instruction=True
                )
        
        # Regular labels - prioritize code labels over data labels
        if ':' in stripped and not any(op in stripped for op in ['mov', 'lea', 'call', 'jmp', 'add', 'sub']):
            label_name = stripped.split(':')[0].strip()
            
            # Check if it's a data label
            is_data_label = any(re.match(pattern, stripped) for pattern in self.data_label_patterns)
            
            if is_data_label:
                return AssemblyLine(
                    content=line,
                    line_number=line_number,
                    section_type=SectionType.DATA,
                    is_label=True,
                    label_name=label_name
                )
            else:
                # All other labels (main, __else2, etc.) go in text section
                return AssemblyLine(
                    content=line,
                    line_number=line_number,
                    section_type=SectionType.TEXT,
                    is_label=True,
                    label_name=label_name
                )
        
        # Data declarations (only after checking instructions and labels)
        for pattern in self.data_patterns:
            if re.match(pattern, stripped):
                # Extract data name if present
                data_name = None
                if ':' in stripped:
                    data_name = stripped.split(':')[0].strip()
                elif len(stripped.split()) > 1 and stripped.split()[0] not in ['dd', 'db', 'dw', 'dq']:
                    data_name = stripped.split()[0]
                
                return AssemblyLine(
                    content=line,
                    line_number=line_number,
                    section_type=SectionType.DATA,
                    is_data_declaration=True,
                    data_name=data_name
                )
        
        # Default to text section for anything else
        if stripped and not stripped.startswith(';'):
            return AssemblyLine(
                content=line,
                line_number=line_number,
                section_type=SectionType.TEXT,
                is_instruction=True
            )
        
        return AssemblyLine(
            content=line,
            line_number=line_number,
            section_type=SectionType.TEXT,
            is_instruction=True
        )
    
    def _remove_inline_comments(self, line: str) -> str:
        """Remove inline comments from assembly instructions"""
        # Don't remove comments from data declarations
        stripped = line.strip()
        if any(re.match(pattern, stripped) for pattern in self.data_patterns):
            return line
        
        # For instructions, remove everything after ; except for specific patterns
        if ';' in line:
            # Split on semicolon
            parts = line.split(';', 1)
            instruction_part = parts[0].rstrip()
            comment_part = parts[1].strip() if len(parts) > 1 else ""
            
            # Keep only specific comment patterns
            if comment_part:
                # Keep important comments like "Reserve shadow space", "Load base address", etc.
                important_keywords = [
                    "reserve shadow space", "load base address", "add offset", 
                    "save handle", "console handle", "text buffer", "number of chars",
                    "written count", "5th parameter", "exit code", "std_output_handle"
                ]
                
                if any(keyword in comment_part.lower() for keyword in important_keywords):
                    return f"{instruction_part}  ; {comment_part}"
                else:
                    # Remove other inline comments
                    return instruction_part
            else:
                return instruction_part
        
        return line
    
    def _organize_sections(self) -> None:
        """Organize parsed lines into appropriate sections"""
        self.data_lines = []
        self.text_lines = []
        self.header_comments = []
        
        # Organize by section type
        for i, line in enumerate(self.parsed_lines):
            if line.section_type == SectionType.DATA:
                self.data_lines.append(line.content)
            elif line.section_type == SectionType.TEXT:
                # Skip section declarations and empty lines from removed comments
                if (not line.content.strip().startswith('section ') and 
                    line.content.strip() != ""):
                    self.text_lines.append(line.content)
        
        # Post-process data lines to combine labels with their declarations
        self.data_lines = self._combine_data_labels_and_declarations(self.data_lines)
        
        # Look for orphaned db declarations in text lines and move them to data
        self._move_orphaned_declarations()
    
    def _combine_data_labels_and_declarations(self, data_lines: List[str]) -> List[str]:
        """Combine data labels with their declarations on single lines"""
        combined_lines = []
        i = 0
        
        # First pass: collect all labels and their positions
        labels = {}
        declarations = {}
        
        for idx, line in enumerate(data_lines):
            stripped = line.strip()
            if stripped.endswith(':'):
                label_name = stripped[:-1]  # Remove the colon
                labels[label_name] = idx
            elif any(stripped.startswith(decl) for decl in ['dd', 'db', 'dw', 'dq']):
                declarations[idx] = stripped
        
        # Second pass: combine labels with declarations
        processed = set()
        
        for idx, line in enumerate(data_lines):
            if idx in processed:
                continue
                
            stripped = line.strip()
            
            # Check if this is a label line
            if stripped.endswith(':'):
                label_name = stripped[:-1]  # Remove the colon
                
                # Look for the corresponding declaration immediately after
                if idx + 1 < len(data_lines):
                    next_line = data_lines[idx + 1].strip()
                    if any(next_line.startswith(decl) for decl in ['dd', 'db', 'dw', 'dq']):
                        # Combine label and declaration (without colon)
                        combined_line = f"    {label_name} {next_line}"
                        combined_lines.append(combined_line)
                        processed.add(idx)
                        processed.add(idx + 1)
                        continue
                
                # If no immediate declaration, look for orphaned declarations in text section
                # For now, just add the label without colon
                combined_lines.append(f"    {label_name}")
                processed.add(idx)
                
            elif any(stripped.startswith(decl) for decl in ['dd', 'db', 'dw', 'dq']) and idx not in processed:
                # Standalone declaration
                combined_lines.append(f"    {stripped}")
                processed.add(idx)
            elif stripped and idx not in processed:
                # Other lines
                if not stripped.startswith('    ') and not stripped.startswith('\t'):
                    combined_lines.append(f"    {stripped}")
                else:
                    combined_lines.append(line)
                processed.add(idx)
        
        return combined_lines
    
    def _move_orphaned_declarations(self):
        """Move orphaned db declarations from text section to data section"""
        orphaned_declarations = []
        remaining_text_lines = []
        
        for line in self.text_lines:
            stripped = line.strip()
            # Check if this is a db declaration that should be in data section
            if stripped.startswith('db ') and ('"' in stripped or "'" in stripped):
                # This looks like a string declaration that belongs in data
                orphaned_declarations.append(f"    {stripped}")
            else:
                remaining_text_lines.append(line)
        
        # Update text lines without the orphaned declarations
        self.text_lines = remaining_text_lines
        
        # Try to match orphaned declarations with empty labels in data section
        if orphaned_declarations:
            updated_data_lines = []
            orphaned_iter = iter(orphaned_declarations)
            
            for line in self.data_lines:
                stripped = line.strip()
                # Check if this is an empty label (no declaration)
                if stripped and not any(decl in stripped for decl in [' dd ', ' db ', ' dw ', ' dq ']):
                    # Try to get the next orphaned declaration
                    try:
                        orphaned_decl = next(orphaned_iter).strip()
                        # Combine the label with the orphaned declaration
                        combined = f"    {stripped} {orphaned_decl.strip()}"
                        updated_data_lines.append(combined)
                    except StopIteration:
                        # No more orphaned declarations
                        updated_data_lines.append(line)
                else:
                    updated_data_lines.append(line)
            
            # Add any remaining orphaned declarations at the end
            try:
                while True:
                    remaining_orphan = next(orphaned_iter)
                    updated_data_lines.append(remaining_orphan)
            except StopIteration:
                pass
            
            self.data_lines = updated_data_lines
    
    def _generate_formatted_output(self) -> str:
        """Generate the final formatted assembly output"""
        output_lines = []
        
        # Add .data section
        output_lines.append('section .data')
        
        # Add data section content
        if self.data_lines:
            for line in self.data_lines:
                if line.strip():  # Skip empty lines in data section
                    output_lines.append(line)  # Lines are already properly formatted from _combine_data_labels_and_declarations
        else:
            output_lines.append('    ; No data declarations')
        
        output_lines.append('')
        
        # Add .text section
        output_lines.append('section .text')
        
        # Add text section content with instruction grouping
        if self.text_lines:
            # First add extern and global declarations
            extern_lines = []
            global_lines = []
            other_lines = []
            
            for line in self.text_lines:
                stripped = line.strip()
                if stripped.startswith('extern '):
                    extern_lines.append(line)
                elif stripped.startswith('global '):
                    global_lines.append(line)
                else:
                    other_lines.append(line)
            
            # Add extern declarations first
            if extern_lines:
                for line in extern_lines:
                    output_lines.append(line)
                output_lines.append('')
            
            # Add global declarations
            if global_lines:
                for line in global_lines:
                    output_lines.append(line)
                output_lines.append('')
            
            # Add the rest of the code with grouping
            output_lines.extend(self._group_instructions(other_lines))
        
        return '\n'.join(output_lines)
    
    def _group_instructions(self, lines: List[str]) -> List[str]:
        """Group instructions with proper spacing between groups"""
        grouped_lines = []
        current_group = []
        last_was_comment = False
        last_was_label = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                # Empty line - end current group if it has content
                if current_group:
                    grouped_lines.extend(current_group)
                    grouped_lines.append('')  # Add spacing between groups
                    current_group = []
                continue
            
            # Check if this is an instruction comment (original ASM statement)
            is_instruction_comment = any(re.search(pattern, stripped) for pattern in self.keep_comment_patterns)
            
            # Check if this is a label
            is_label = stripped.endswith(':') and not any(op in stripped for op in ['mov', 'lea', 'call', 'jmp'])
            
            # Start a new group on:
            # 1. Instruction comments (original ASM statements)
            # 2. Labels
            if (is_instruction_comment or is_label) and current_group:
                # End current group
                grouped_lines.extend(current_group)
                grouped_lines.append('')  # Add spacing between groups
                current_group = []
            
            # Add line to current group with proper indentation and comment cleaning
            if stripped:
                # Clean inline comments and C instruction comments
                cleaned_line = self._clean_inline_comments(line)
                cleaned_line = self._clean_c_instruction_comment(cleaned_line)
                
                stripped_cleaned = cleaned_line.strip()
                if (not cleaned_line.startswith('    ') and not cleaned_line.startswith('\t') and 
                    not stripped_cleaned.endswith(':') and not stripped_cleaned.startswith(';') and
                    not stripped_cleaned.startswith('global ') and not stripped_cleaned.startswith('extern ')):
                    current_group.append(f'    {stripped_cleaned}')
                else:
                    current_group.append(cleaned_line)
            
            last_was_comment = is_instruction_comment
            last_was_label = is_label
        
        # Add final group
        if current_group:
            grouped_lines.extend(current_group)
        
        return grouped_lines
    
    def format_file(self, input_file: str, output_file: str = None) -> bool:
        """Format an assembly file"""
        if output_file is None:
            output_file = input_file
        
        try:
            with open(input_file, 'r') as f:
                content = f.read()
            
            formatted_content = self.format_assembly(content)
            
            with open(output_file, 'w') as f:
                f.write(formatted_content)
            
            if self.debug:
                print(f"[Formatter] Successfully formatted: {input_file} -> {output_file}")
            
            return True
        except Exception as e:
            if self.debug:
                print(f"[Formatter] Error formatting file: {e}")
            return False

def format_assembly_content(content: str, debug: bool = False) -> str:
    """Utility function to format assembly content"""
    formatter = AssemblyFormatter(debug=debug)
    return formatter.format_assembly(content)

def format_assembly_file(input_file: str, output_file: str = None, debug: bool = False) -> bool:
    """Utility function to format an assembly file"""
    formatter = AssemblyFormatter(debug=debug)
    return formatter.format_file(input_file, output_file)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python formatter.py <input_file> [output_file] [--debug]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else input_file
    debug = '--debug' in sys.argv
    
    if format_assembly_file(input_file, output_file, debug):
        print(f"Successfully formatted assembly: {input_file} -> {output_file}")
    else:
        print(f"Failed to format assembly: {input_file}")
        sys.exit(1)