#!/usr/bin/env python3

from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
import re
from datetime import datetime
from pathlib import Path

@dataclass
class MacroParameter:
    """Represents a macro parameter"""
    name: str
    default_value: Optional[str] = None
    is_variadic: bool = False

@dataclass 
class Macro:
    """Represents a preprocessor macro"""
    name: str
    parameters: List[MacroParameter] = field(default_factory=list)
    body: List[str] = field(default_factory=list)
    is_function_like: bool = False
    is_builtin: bool = False
    defined_at: Optional[str] = None
    usage_count: int = 0

class Preprocessor:
    """Advanced preprocessor with macro support"""
    
    def __init__(self):
        self.macros: Dict[str, Macro] = {}
        self.include_paths: List[str] = ['.', '/usr/local/include', '/usr/include']
        self.included_files: Set[str] = set()
        self.conditional_stack: List[bool] = []
        self.current_file = ""
        self.line_number = 0
        self.definitions: Dict[str, str] = {}
        
        # Register built-in macros
        self._register_builtin_macros()
    
    def _register_builtin_macros(self):
        """Register built-in preprocessor macros"""
        builtins = {
            '__DATE__': datetime.now().strftime('"%b %d %Y"'),
            '__TIME__': datetime.now().strftime('"%H:%M:%S"'),
            '__FILE__': '""',  # Will be set during processing
            '__LINE__': '0',   # Will be updated during processing
            '__VERSION__': '"HLASM 2.0"',
            '__ARCH__': '"x86_64"',
            '__OS__': '"unknown"',  # Will be detected
        }
        
        for name, value in builtins.items():
            self.macros[name] = Macro(
                name=name,
                body=[value],
                is_builtin=True
            )
    
    def add_include_path(self, path: str):
        """Add include search path"""
        if path not in self.include_paths:
            self.include_paths.append(path)
    
    def define_macro(self, name: str, value: str = "", 
                    parameters: List[str] = None, line_number: int = 0):
        """Define a new macro"""
        macro_params = []
        if parameters:
            for param in parameters:
                if param.endswith('...'):
                    # Variadic parameter
                    macro_params.append(MacroParameter(param[:-3], is_variadic=True))
                elif '=' in param:
                    # Parameter with default value
                    param_name, default = param.split('=', 1)
                    macro_params.append(MacroParameter(param_name.strip(), default.strip()))
                else:
                    macro_params.append(MacroParameter(param.strip()))
        
        self.macros[name] = Macro(
            name=name,
            parameters=macro_params,
            body=value.split('\n') if value else [],
            is_function_like=bool(parameters),
            defined_at=f"{self.current_file}:{line_number}"
        )
    
    def undefine_macro(self, name: str):
        """Remove a macro definition"""
        if name in self.macros and not self.macros[name].is_builtin:
            del self.macros[name]
    
    def is_defined(self, name: str) -> bool:
        """Check if macro is defined"""
        return name in self.macros
    
    def expand_macro(self, name: str, arguments: List[str] = None) -> str:
        """Expand a macro with given arguments"""
        if name not in self.macros:
            return name
        
        macro = self.macros[name]
        macro.usage_count += 1
        
        if not macro.is_function_like:
            # Simple text replacement
            return '\n'.join(macro.body)
        
        # Function-like macro expansion
        if arguments is None:
            arguments = []
        
        # Build parameter substitution map
        substitutions = {}
        
        for i, param in enumerate(macro.parameters):
            if i < len(arguments):
                substitutions[param.name] = arguments[i]
            elif param.default_value:
                substitutions[param.name] = param.default_value
            elif param.is_variadic:
                # Handle variadic parameters
                variadic_args = arguments[i:] if i < len(arguments) else []
                substitutions[param.name] = ', '.join(variadic_args)
            else:
                substitutions[param.name] = ""
        
        # Perform substitution
        expanded_lines = []
        for line in macro.body:
            expanded_line = line
            for param_name, value in substitutions.items():
                expanded_line = expanded_line.replace(f'#{param_name}#', value)
            expanded_lines.append(expanded_line)
        
        return '\n'.join(expanded_lines)
    
    def process_file(self, filename: str) -> List[str]:
        """Process a file with all preprocessor directives"""
        self.current_file = filename
        self.line_number = 0
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise PreprocessorError(f"File not found: {filename}")
        
        # Update built-in macros
        self.macros['__FILE__'].body = [f'"{filename}"']
        
        processed_lines = []
        i = 0
        
        while i < len(lines):
            self.line_number = i + 1
            line = lines[i].rstrip()
            
            # Update __LINE__ macro
            self.macros['__LINE__'].body = [str(self.line_number)]
            
            # Process preprocessor directives
            if line.strip().startswith('#'):
                directive_result = self._process_directive(line, lines, i)
                if directive_result['skip_lines'] > 0:
                    i += directive_result['skip_lines']
                if directive_result['insert_lines']:
                    processed_lines.extend(directive_result['insert_lines'])
            else:
                # Process macro expansions in regular lines
                if self._should_include_line():
                    expanded_line = self._expand_macros_in_line(line)
                    processed_lines.append(expanded_line)
            
            i += 1
        
        return processed_lines
    
    def _process_directive(self, line: str, all_lines: List[str], current_index: int) -> Dict[str, Any]:
        """Process a single preprocessor directive"""
        result = {'skip_lines': 0, 'insert_lines': []}
        
        directive_line = line.strip()[1:]  # Remove #
        parts = directive_line.split(None, 1)
        
        if not parts:
            return result
        
        directive = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if directive == 'define':
            # Skip define directives - they cause system issues
            result['insert_lines'] = [f"; Skipped directive: {line.strip()}"]
        elif directive == 'undef':
            self._handle_undef(args)
        elif directive == 'ifdef':
            # Skip ifdef directives
            result['insert_lines'] = [f"; Skipped directive: {line.strip()}"]
        elif directive == 'ifndef':
            # Skip ifndef directives  
            result['insert_lines'] = [f"; Skipped directive: {line.strip()}"]
        elif directive == 'if':
            # Skip if directives
            result['insert_lines'] = [f"; Skipped directive: {line.strip()}"]
        elif directive == 'elif':
            # Skip elif directives
            result['insert_lines'] = [f"; Skipped directive: {line.strip()}"]
        elif directive == 'else':
            # Skip else directives
            result['insert_lines'] = [f"; Skipped directive: {line.strip()}"]
        elif directive == 'endif':
            # Skip endif directives
            result['insert_lines'] = [f"; Skipped directive: {line.strip()}"]
        elif directive == 'include':
            included_lines = self._handle_include(args)
            result['insert_lines'] = included_lines
        elif directive == 'pragma':
            self._handle_pragma(args)
        elif directive == 'error':
            raise PreprocessorError(f"#error: {args}")
        elif directive == 'warning':
            print(f"Warning: {args}")
        
        return result
    
    def _handle_define(self, args: str):
        """Handle #define directive"""
        # Parse: #define NAME(params) body or #define NAME body
        match = re.match(r'(\w+)(?:\((.*?)\))?\s*(.*)', args)
        
        if not match:
            raise PreprocessorError(f"Invalid #define syntax: {args}")
        
        name, params_str, body = match.groups()
        
        parameters = None
        if params_str is not None:
            # Function-like macro
            parameters = [p.strip() for p in params_str.split(',')] if params_str else []
        
        self.define_macro(name, body, parameters, self.line_number)
    
    def _handle_undef(self, args: str):
        """Handle #undef directive"""
        name = args.strip()
        self.undefine_macro(name)
    
    def _handle_ifdef(self, args: str):
        """Handle #ifdef directive"""
        name = args.strip()
        condition = self.is_defined(name)
        self.conditional_stack.append(condition)
    
    def _handle_ifndef(self, args: str):
        """Handle #ifndef directive"""
        name = args.strip()
        condition = not self.is_defined(name)
        self.conditional_stack.append(condition)
    
    def _handle_if(self, args: str):
        """Handle #if directive"""
        # Simple expression evaluation
        condition = self._evaluate_condition(args)
        self.conditional_stack.append(condition)
    
    def _handle_elif(self, args: str):
        """Handle #elif directive"""
        if not self.conditional_stack:
            raise PreprocessorError("#elif without #if")
        
        # If previous condition was true, this elif is false
        if self.conditional_stack[-1]:
            self.conditional_stack[-1] = False
        else:
            # Evaluate this condition
            condition = self._evaluate_condition(args)
            self.conditional_stack[-1] = condition
    
    def _handle_else(self):
        """Handle #else directive"""
        if not self.conditional_stack:
            raise PreprocessorError("#else without #if")
        
        # Flip the condition
        self.conditional_stack[-1] = not self.conditional_stack[-1]
    
    def _handle_endif(self):
        """Handle #endif directive"""
        if not self.conditional_stack:
            raise PreprocessorError("#endif without #if")
        
        self.conditional_stack.pop()
    
    def _handle_include(self, args: str) -> List[str]:
        """Handle #include directive"""
        # Parse include filename
        args = args.strip()
        
        if args.startswith('"') and args.endswith('"'):
            # Local include "file.h"
            filename = args[1:-1]
            search_paths = [Path(self.current_file).parent] + [Path(p) for p in self.include_paths]
        elif args.startswith('<') and args.endswith('>'):
            # System include <file.h>
            filename = args[1:-1]
            search_paths = [Path(p) for p in self.include_paths]
        else:
            raise PreprocessorError(f"Invalid #include syntax: {args}")
        
        # Find the file
        include_file = None
        for search_path in search_paths:
            candidate = search_path / filename
            if candidate.exists():
                include_file = str(candidate)
                break
        
        if not include_file:
            raise PreprocessorError(f"Include file not found: {filename}")
        
        # Prevent recursive includes
        if include_file in self.included_files:
            return []
        
        self.included_files.add(include_file)
        
        # Recursively process included file
        old_file = self.current_file
        old_line = self.line_number
        
        try:
            included_lines = self.process_file(include_file)
            return included_lines
        finally:
            self.current_file = old_file
            self.line_number = old_line
            self.included_files.discard(include_file)
    
    def _handle_pragma(self, args: str):
        """Handle #pragma directive"""
        # Handle pragma directives (compiler-specific)
        if args.startswith('once'):
            # Include guard
            if self.current_file not in self.included_files:
                self.included_files.add(self.current_file)
        elif args.startswith('optimize'):
            # Optimization hint
            pass  # Implementation specific
    
    def _should_include_line(self) -> bool:
        """Check if current line should be included based on conditionals"""
        return all(self.conditional_stack) if self.conditional_stack else True
    
    def _expand_macros_in_line(self, line: str) -> str:
        """Expand all macros in a line"""
        expanded = line
        
        # Find all potential macro invocations
        for macro_name, macro in self.macros.items():
            if macro.is_function_like:
                # Function-like macro: NAME(args)
                pattern = rf'\b{re.escape(macro_name)}\s*\('
                matches = list(re.finditer(pattern, expanded))
                
                for match in reversed(matches):  # Process from right to left
                    start = match.start()
                    # Find matching closing parenthesis
                    paren_count = 0
                    end = start + len(match.group())
                    
                    for i in range(end, len(expanded)):
                        if expanded[i] == '(':
                            paren_count += 1
                        elif expanded[i] == ')':
                            if paren_count == 0:
                                end = i + 1
                                break
                            paren_count -= 1
                    
                    # Extract arguments
                    args_str = expanded[match.end()-1:end-1]  # Between parentheses
                    if args_str.startswith('(') and args_str.endswith(')'):
                        args_content = args_str[1:-1]
                        arguments = self._parse_macro_arguments(args_content)
                        
                        # Expand macro
                        expansion = self.expand_macro(macro_name, arguments)
                        expanded = expanded[:start] + expansion + expanded[end:]
            else:
                # Simple macro: just replace the name
                pattern = rf'\b{re.escape(macro_name)}\b'
                expansion = self.expand_macro(macro_name)
                expanded = re.sub(pattern, expansion, expanded)
        
        return expanded
    
    def _parse_macro_arguments(self, args_str: str) -> List[str]:
        """Parse macro arguments from string"""
        if not args_str.strip():
            return []
        
        arguments = []
        current_arg = ""
        paren_depth = 0
        
        for char in args_str:
            if char == ',' and paren_depth == 0:
                arguments.append(current_arg.strip())
                current_arg = ""
            else:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                current_arg += char
        
        if current_arg.strip():
            arguments.append(current_arg.strip())
        
        return arguments
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate preprocessor condition"""
        # Simple condition evaluation for basic cases
        condition = condition.strip()
        
        # Handle defined() operator
        defined_match = re.search(r'defined\s*\(\s*(\w+)\s*\)', condition)
        if defined_match:
            macro_name = defined_match.group(1)
            is_def = self.is_defined(macro_name)
            condition = condition.replace(defined_match.group(0), str(is_def).lower())
        
        # Handle simple numeric comparisons
        if re.match(r'^\s*\d+\s*$', condition):
            return int(condition.strip()) != 0
        
        # Handle basic boolean logic (simplified)
        if '&&' in condition:
            parts = condition.split('&&')
            return all(self._evaluate_condition(part) for part in parts)
        
        if '||' in condition:
            parts = condition.split('||')
            return any(self._evaluate_condition(part) for part in parts)
        
        # Default: assume true for unknown conditions
        return True
    
    def get_macro_report(self) -> str:
        """Generate macro usage report"""
        report = []
        report.append("=" * 60)
        report.append("PREPROCESSOR MACRO REPORT")
        report.append("=" * 60)
        
        if not self.macros:
            report.append("No macros defined.")
            return "\n".join(report)
        
        # Built-in macros
        builtin_macros = [m for m in self.macros.values() if m.is_builtin]
        if builtin_macros:
            report.append("Built-in Macros:")
            for macro in builtin_macros:
                body = ' '.join(macro.body) if macro.body else ""
                report.append(f"  {macro.name} = {body}")
            report.append("")
        
        # User-defined macros
        user_macros = [m for m in self.macros.values() if not m.is_builtin]
        if user_macros:
            report.append("User-defined Macros:")
            for macro in user_macros:
                params = ""
                if macro.is_function_like:
                    param_strs = []
                    for param in macro.parameters:
                        param_str = param.name
                        if param.default_value:
                            param_str += f"={param.default_value}"
                        if param.is_variadic:
                            param_str += "..."
                        param_strs.append(param_str)
                    params = f"({', '.join(param_strs)})"
                
                body = ' '.join(macro.body) if macro.body else ""
                usage = f" (used {macro.usage_count} times)" if macro.usage_count > 0 else ""
                defined_at = f" [defined at {macro.defined_at}]" if macro.defined_at else ""
                
                report.append(f"  {macro.name}{params} = {body}{usage}{defined_at}")
        
        report.append("=" * 60)
        return "\n".join(report)

class PreprocessorError(Exception):
    """Exception raised for preprocessor errors"""
    pass

# Standard preprocessor macros for HLASM
STANDARD_MACROS = {
    'BYTE_SIZE': '1',
    'WORD_SIZE': '2', 
    'DWORD_SIZE': '4',
    'QWORD_SIZE': '8',
    'POINTER_SIZE': '8',
    'MAX_INT8': '127',
    'MIN_INT8': '-128',
    'MAX_UINT8': '255',
    'MAX_INT16': '32767',
    'MIN_INT16': '-32768',
    'MAX_UINT16': '65535',
    'MAX_INT32': '2147483647',
    'MIN_INT32': '-2147483648',
    'MAX_UINT32': '4294967295',
    'NULL': '0',
    'TRUE': '1',
    'FALSE': '0',
}