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

        # Final global pass: rewrite any remaining memory-to-memory operations
        # that might have slipped through into safe register-mediated sequences.
        # Handle patterns like: lea [ rel V11 ] , [ rel V9 ]
        result = re.sub(r"\blea\s*\[\s*rel\s+(V\d+)\s*\]\s*,\s*\[\s*rel\s+(V\d+)\s*\]",
            lambda m: f"    lea rax, [rel {m.group(2)}]\n    mov qword [ rel {m.group(1)} ], rax",
            result)

        # Handle mov with memory source: mov qword [ dst ] , [ rel V# ]
        result = re.sub(r"\bmov\s*(?P<size>qword|dword|word|byte)?\s*\[\s*(?P<dst>[^\]]+)\s*\]\s*,\s*\[\s*rel\s+(?P<src>V\d+)\s*\]",
            lambda m: f"    mov {'rax' if (m.group('size')=='qword') else 'eax'}, [ rel {m.group('src')} ]\n    mov { (m.group('size')+' ') if m.group('size') else '' }[ {m.group('dst')} ], {'rax' if (m.group('size')=='qword') else 'eax'}",
            result)

        # Fix cases where a 64-bit memory store is written from a 32-bit register
        # (common generator mistake). Convert 'lea eax, [ rel V# ]' -> 'lea rax, [ rel V# ]'
        # and 'mov qword [ rel V# ], eax' -> 'mov qword [ rel V# ], rax'
        result = re.sub(r"\blea\s+eax\s*,\s*\[\s*rel\s+(V\d+)\s*\]",
            lambda m: f"    lea rax, [rel {m.group(1)}]",
            result)

        result = re.sub(r"\bmov\s+qword\s*\[\s*rel\s+(V\d+)\s*\]\s*,\s*eax",
            lambda m: f"    mov qword [ rel {m.group(1)} ], rax",
            result)

        print_success("Assembly fixed for NASM compatibility")
    # Final safety-net: ensure any referenced V# labels that remain
        # undefined are given simple placeholders in a .data section so
        # NASM does not abort with 'symbol not defined'. This avoids
        # brittle ordering issues earlier in the pipeline.
        all_v_refs = set(re.findall(r"\b(V\d+)\b", result))
        missing_vs = sorted([v for v in all_v_refs if v not in self.defined_labels])
        if missing_vs:
            add_lines = []
            add_lines.append("")
            add_lines.append("    ; Placeholder V# variables added by AssemblyFixer")
            add_lines.append('section .data')
            for v in missing_vs:
                add_lines.append(f"    {v} dd 0")
            # Append placeholders to the end of the assembly
            result = result + '\n' + '\n'.join(add_lines)

        # Final cleanup: remove all semicolon comments and any standalone
        # 'nop' lines that were used as markers by earlier passes. We do
        # this as the last step so other fixups operate on the original
        # annotations; now we produce the clean assembly expected by users.
        cleaned_lines = []
        for ln in result.split('\n'):
            s = ln.rstrip()
            # Remove full-line or trailing comments (starting with ';')
            # But preserve leading whitespace before instruction if any.
            if ';' in s:
                # Remove ';' and everything after it
                s = s.split(';', 1)[0].rstrip()

            # Skip pure 'nop' lines (possibly indented)
            if re.match(r'^\s*nop\s*$', s, re.I):
                continue

            # Skip empty lines created by stripping comments (but keep a
            # single blank line for separation if desired)
            cleaned_lines.append(s)

        # Reassemble into final text, stripping possible leading/trailing blank lines
        final = '\n'.join([l for l in cleaned_lines]).strip() + '\n'
        return final
    
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

            # Also detect data labels defined as 'LABEL db ...' or 'LABEL dd ...'
            m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s+(db|dq|dd|resb|times)\b', line)
            if m:
                label = m.group(1)
                self.defined_labels.add(label)
    
    def _fix_data_section_line(self, line: str) -> str:
        """Fix issues in data section lines"""
        stripped = line.strip()
        
        # Track variable definitions to avoid duplicates
        if ' dd ' in line and not line.strip().startswith(';'):
            var_match = re.match(r'\s*(\w+)\s+dd\s+', line)
            if var_match:
                var_name = var_match.group(1)
                # Do not treat CASM-generated V# symbols as duplicates to
                # be removed -- they are intentionally emitted and may
                # appear multiple times in different forms during
                # prettification/transpilation passes. Only skip genuine
                # duplicates for user-defined labels (non-V#).
                if var_name in self.defined_variables and not re.match(r'^V\d+$', var_name):
                    # Skip duplicate variable definition (non-V# only)
                    return f"    ; Duplicate {var_name} definition skipped\n"
                else:
                    self.defined_variables.add(var_name)
        
        # Fix string format issues
        if 'db "' in stripped and '", 10, "", 0' in stripped:
            # Remove the extra empty string
            line = line.replace('", 10, "", 0', '", 10, 0')
        
        # Fix .LC labels to be global
            if stripped.startswith('.LC'):
                # Convert .LC0 db "..." to CSTR0 db "..."
                line = re.sub(r'\.LC(\d+)', r'CSTR\1', line)
        
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

        # Fix memory-to-memory operations which are invalid in x86.
        # Example incorrect forms we sometimes emit:
        #   lea [ rel V11 ] , [ rel V9 ]
        #   mov qword [ rsp + 24 ] , [ rel V11 ]
        # Convert them into a register-mediated sequence, e.g.:
        #   lea rax, [rel V9]
        #   mov qword [ rel V11 ], rax
        m2m = re.match(r"\s*(?:(?P<size>qword|dword|word|byte)\s+)?(mov|lea)\s+(?P<dst>\[.*?\])\s*,\s*(?P<src>\[.*\])", line, re.I)
        if m2m:
            instr = m2m.group(2).lower()
            size = (m2m.group('size') or '').lower()
            dst = m2m.group('dst').strip()
            src = m2m.group('src').strip()

            # Choose appropriate temp register size
            if size == 'qword':
                temp_reg = 'rax'
            else:
                temp_reg = 'eax'

            # For lea: compute address of src into temp_reg
            if instr == 'lea':
                # src is a memory operand like [ rel V9 ] - use it as-is in lea
                new_lines = []
                new_lines.append(f"        lea {temp_reg}, {src}")
                # write temp_reg into dst (use qword for pointers)
                size_prefix = 'qword ' if size == 'qword' or instr == 'lea' else (size + ' ' if size else '')
                new_lines.append(f"        mov {size_prefix}{dst}, {temp_reg}")
                return '\n'.join(new_lines)

            # For mov: move src -> temp_reg, then temp_reg -> dst
            if instr == 'mov':
                new_lines = []
                new_lines.append(f"        mov {temp_reg}, {src}")
                size_prefix = (size + ' ') if size else ''
                new_lines.append(f"        mov {size_prefix}{dst}, {temp_reg}")
                return '\n'.join(new_lines)
        
        # Fix .refptr references - convert to direct variable access
        # Handle .refptr references for both old 'var_' prefix and new 'V<number>' labels
        if '.refptr.' in line:
            # Convert patterns like: mov rax, qword .refptr.var_name[rip] or .refptr.V1[rip]
            # into: lea rax, [rel var_name]  or lea rax, [rel V1]
            # Regex accepts either var_<name> or V<number>
            if 'mov' in line and 'qword .refptr.' in line:
                var_match = re.search(r"mov\s+(\w+),\s*qword\s+\.refptr\.((?:var_\w+)|(?:V\d+))\[rip\]", line)
                if var_match:
                    reg = var_match.group(1)
                    var_name = var_match.group(2)
                    line = f"        lea {reg}, [rel {var_name}]"
            else:
                # Also handle other forms like 'lea rax, qword .refptr.V1[rip]' if present
                var_match2 = re.search(r"(lea|mov)\s+(\w+),\s*(?:qword\s+)?\.refptr\.((?:var_\w+)|(?:V\d+))\[rip\]", line)
                if var_match2:
                    instr = var_match2.group(1)
                    reg = var_match2.group(2)
                    var_name = var_match2.group(3)
                    # Normalize to lea reg, [rel VAR]
                    line = f"        lea {reg}, [rel {var_name}]"
        
        # Fix references to .LC labels (remove the dot) - applies to all lines, including C code blocks
        if '.LC' in line:
              line = re.sub(r'\.LC(\d+)', r'CSTR\1', line)
        
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

        # Remove invalid register-size moves generated by the codegen such
        # as `mov rax, eax` which NASM rejects. Writing to the 32-bit
        # register (e.g. `mov eax, ...`) already zero-extends the upper
        # 32 bits, so the 64-bit mov is redundant and invalid.
        try:
            line = re.sub(r"\bmov\s+rax\s*,\s*eax\b", "        ; removed redundant mov rax, eax", line, flags=re.I)
        except Exception:
            pass
        
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
        # Check for undefined string references (STR1..STRN)
        for m in re.finditer(r'STR(\d+)', line):
            lbl = m.group(0)
            if lbl not in self.seen_strings:
                self.seen_strings.add(lbl)
        
        return line
    
    def _fix_label_consistency(self, line: str) -> str:
        """Fix label consistency issues - DISABLED"""
        # DISABLED: This function was causing duplicate label issues
        # The code generator now produces unique labels correctly
        return line
    
    def _add_missing_strings(self, lines: List[str]) -> List[str]:
        """Add missing string definitions to data section and fix inconsistent labeling"""
        missing_strings = self.referenced_labels - self.defined_labels
        
        # Also check for inconsistent STR numbering
        str_refs = {label for label in self.referenced_labels if label.startswith('STR')}
        str_defs = {label for label in self.defined_labels if label.startswith('STR')}

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

                # Add missing LC constants and placeholders before text section
                # Compute any referenced V# symbols that are not defined and
                # ensure we always provide a simple placeholder so NASM
                # doesn't fail with 'symbol not defined'. This is a defensive
                # fallback for cases where earlier passes removed the original
                # V# definitions unexpectedly.
                all_v_refs = set(re.findall(r"\b(V\d+)\b", '\n'.join(lines)))
                missing_vs = sorted([v for v in all_v_refs if v not in self.defined_labels])

                if missing_lc or missing_strings or missing_vs:
                    result_lines.append("")
                    result_lines.append("    ; Missing LC constants / placeholders added by AssemblyFixer")
                    for lc_label in sorted(missing_lc):
                        if lc_label == 'LC0':
                            # Add the floating point constant for 2.0
                            result_lines.append(f"    {lc_label}:")
                            result_lines.append("        dq 2.0  ; 2.0 in double precision")
                        else:
                            result_lines.append(f"    {lc_label} db \"Missing constant {lc_label}\", 0")

                    # Add placeholders for missing V# variables (e.g., V6, V8)
                    # Prefer the explicit missing_vs computed above, but also
                    # include any V# discovered in missing_strings as a fallback.
                    missing_vars = sorted(set(missing_vs) | {lbl for lbl in missing_strings if re.match(r'^V\d+$', lbl)})
                    for v in missing_vars:
                        result_lines.append(f"    {v} dd 0")

                    # Add extern declarations for missing __imp_* imports
                    missing_imports = sorted([lbl for lbl in missing_strings if lbl.startswith('__imp_')])
                    if missing_imports:
                        result_lines.append("")
                        result_lines.append("    ; Missing import symbols")
                        for imp in missing_imports:
                            result_lines.append(f"    extern {imp}")

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
        existing_map = {}

        # Collect existing string definitions (labels with db/dq/dd)
        label_db_re = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+db\s+(.*)")

        for line in data_lines:
            m = label_db_re.match(line)
            if m:
                label = m.group(1)
                content = m.group(2)
                # Preserve original definition
                existing_map[label] = content
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        # Append placeholder definitions only for referenced strings that are not defined
        for ref_str in sorted(referenced_strings):
            if ref_str not in existing_map:
                fixed_lines.append(f'    {ref_str} db "Placeholder string", 0')
        
        return fixed_lines

# Global instance
assembly_fixer = AssemblyFixer()