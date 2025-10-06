#!/usr/bin/env python3

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path

class ErrorLevel(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    FATAL = "fatal"

@dataclass
class SourceLocation:
    """Represents a location in source code"""
    filename: str
    line: int
    column: int
    
    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}"

@dataclass
class CompilerMessage:
    """Represents a compiler message (error, warning, info)"""
    level: ErrorLevel
    message: str
    location: Optional[SourceLocation] = None
    error_code: Optional[str] = None
    suggestion: Optional[str] = None
    related_locations: List[SourceLocation] = field(default_factory=list)
    
    def __str__(self) -> str:
        prefix = {
            ErrorLevel.INFO: "info",
            ErrorLevel.WARNING: "warning", 
            ErrorLevel.ERROR: "error",
            ErrorLevel.FATAL: "fatal error"
        }[self.level]
        
        if self.location:
            result = f"{self.location}: {prefix}: {self.message}"
        else:
            result = f"{prefix}: {self.message}"
        
        if self.error_code:
            result += f" [{self.error_code}]"
        
        if self.suggestion:
            result += f"\n  suggestion: {self.suggestion}"
        
        return result

class ErrorHandler:
    """Advanced error handling and reporting system"""
    
    def __init__(self):
        self.messages: List[CompilerMessage] = []
        self.error_count = 0
        self.warning_count = 0
        self.current_file = ""
        self.context_lines: Dict[str, List[str]] = {}
        self.max_errors = 100
        self.warnings_as_errors = False
        
        # Error categories
        self.error_categories = {
            "SYNTAX": "Syntax errors",
            "TYPE": "Type checking errors", 
            "SYMBOL": "Symbol resolution errors",
            "SEMANTIC": "Semantic analysis errors",
            "CODEGEN": "Code generation errors",
            "OPTIMIZATION": "Optimization warnings",
            "COMPATIBILITY": "Platform compatibility issues"
        }
    
    def set_current_file(self, filename: str):
        """Set the current file being processed"""
        self.current_file = filename
        if filename and Path(filename).exists():
            try:
                with open(filename, 'r') as f:
                    self.context_lines[filename] = f.readlines()
            except:
                self.context_lines[filename] = []
    
    def add_error(self, message: str, line: int = 0, column: int = 0, 
                  error_code: str = None, suggestion: str = None) -> CompilerMessage:
        """Add an error message"""
        location = SourceLocation(self.current_file, line, column) if line > 0 else None
        error = CompilerMessage(ErrorLevel.ERROR, message, location, error_code, suggestion)
        
        self.messages.append(error)
        self.error_count += 1
        
        # Stop compilation if too many errors
        if self.error_count >= self.max_errors:
            self.add_fatal(f"Too many errors ({self.max_errors}), stopping compilation")
        
        return error
    
    def add_warning(self, message: str, line: int = 0, column: int = 0,
                   error_code: str = None, suggestion: str = None) -> CompilerMessage:
        """Add a warning message"""
        location = SourceLocation(self.current_file, line, column) if line > 0 else None
        
        level = ErrorLevel.ERROR if self.warnings_as_errors else ErrorLevel.WARNING
        warning = CompilerMessage(level, message, location, error_code, suggestion)
        
        self.messages.append(warning)
        
        if self.warnings_as_errors:
            self.error_count += 1
        else:
            self.warning_count += 1
        
        return warning
    
    def add_info(self, message: str, line: int = 0, column: int = 0) -> CompilerMessage:
        """Add an info message"""
        location = SourceLocation(self.current_file, line, column) if line > 0 else None
        info = CompilerMessage(ErrorLevel.INFO, message, location)
        self.messages.append(info)
        return info
    
    def add_fatal(self, message: str, line: int = 0, column: int = 0) -> CompilerMessage:
        """Add a fatal error message"""
        location = SourceLocation(self.current_file, line, column) if line > 0 else None
        fatal = CompilerMessage(ErrorLevel.FATAL, message, location)
        self.messages.append(fatal)
        self.error_count += 1
        return fatal
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return self.error_count > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return self.warning_count > 0
    
    def clear(self):
        """Clear all messages"""
        self.messages.clear()
        self.error_count = 0
        self.warning_count = 0
    
    def get_messages_by_level(self, level: ErrorLevel) -> List[CompilerMessage]:
        """Get all messages of a specific level"""
        return [msg for msg in self.messages if msg.level == level]
    
    def get_messages_for_file(self, filename: str) -> List[CompilerMessage]:
        """Get all messages for a specific file"""
        return [msg for msg in self.messages 
                if msg.location and msg.location.filename == filename]
    
    def format_message_with_context(self, message: CompilerMessage) -> str:
        """Format a message with source code context"""
        if not message.location or message.location.filename not in self.context_lines:
            return str(message)
        
        result = [str(message)]
        
        lines = self.context_lines[message.location.filename]
        line_num = message.location.line
        
        if 1 <= line_num <= len(lines):
            # Show context lines
            start_line = max(1, line_num - 2)
            end_line = min(len(lines), line_num + 2)
            
            for i in range(start_line, end_line + 1):
                line_content = lines[i - 1].rstrip()
                marker = " --> " if i == line_num else "     "
                result.append(f"{i:4d}{marker}{line_content}")
                
                # Show column pointer for error line
                if i == line_num and message.location.column > 0:
                    spaces = " " * (8 + message.location.column - 1)
                    result.append(f"{spaces}^")
        
        return "\n".join(result)
    
    def generate_report(self, show_context: bool = True, 
                       filter_level: Optional[ErrorLevel] = None) -> str:
        """Generate a comprehensive error report"""
        if not self.messages:
            return "No messages to report."
        
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("COMPILER DIAGNOSTIC REPORT")
        report.append("=" * 80)
        
        # Summary
        report.append(f"Errors: {self.error_count}")
        report.append(f"Warnings: {self.warning_count}")
        report.append(f"Total messages: {len(self.messages)}")
        report.append("")
        
        # Group messages by file
        files = {}
        for msg in self.messages:
            if filter_level and msg.level != filter_level:
                continue
                
            filename = msg.location.filename if msg.location else "<unknown>"
            if filename not in files:
                files[filename] = []
            files[filename].append(msg)
        
        # Report by file
        for filename, file_messages in files.items():
            if len(files) > 1:  # Only show filename if multiple files
                report.append(f"File: {filename}")
                report.append("-" * 40)
            
            for msg in sorted(file_messages, key=lambda m: (m.location.line if m.location else 0)):
                if show_context:
                    report.append(self.format_message_with_context(msg))
                else:
                    report.append(str(msg))
                report.append("")
        
        # Error categories summary
        if self.error_count > 0:
            report.append("Error Categories:")
            categories = {}
            for msg in self.messages:
                if msg.level in [ErrorLevel.ERROR, ErrorLevel.FATAL]:
                    category = self._categorize_error(msg.message)
                    categories[category] = categories.get(category, 0) + 1
            
            for category, count in categories.items():
                report.append(f"  {category}: {count}")
            report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def _categorize_error(self, message: str) -> str:
        """Categorize an error message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['syntax', 'parse', 'unexpected', 'expected']):
            return "SYNTAX"
        elif any(word in message_lower for word in ['type', 'incompatible', 'conversion']):
            return "TYPE"
        elif any(word in message_lower for word in ['undefined', 'undeclared', 'symbol', 'redefinition']):
            return "SYMBOL"
        elif any(word in message_lower for word in ['semantic', 'invalid', 'illegal']):
            return "SEMANTIC"
        elif any(word in message_lower for word in ['code generation', 'codegen']):
            return "CODEGEN"
        else:
            return "OTHER"
    
    def save_report(self, filename: str, show_context: bool = True):
        """Save error report to file"""
        report = self.generate_report(show_context)
        with open(filename, 'w') as f:
            f.write(report)

class SyntaxChecker:
    """Advanced syntax checking utilities"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        
        # Syntax patterns
        self.patterns = {
            'valid_identifier': re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
            'valid_label': re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*:$'),
            'valid_number': re.compile(r'^-?\d+(\.\d+)?$'),
            'valid_string': re.compile(r'^".*"$|^\'.*\'$'),
            'balanced_brackets': re.compile(r'^\[.*\]$'),
            'balanced_parens': re.compile(r'^\(.*\)$'),
        }
        
        # Reserved keywords
        self.reserved_keywords = {
            'if', 'else', 'endif', 'while', 'endwhile', 'for', 'endfor',
            'function', 'endfunction', 'var', 'const', 'print', 'println',
            'input', 'exit', 'break', 'continue', 'call', 'return',
            'switch', 'case', 'default', 'endswitch', 'do'
        }
    
    def check_identifier(self, identifier: str, line: int) -> bool:
        """Check if identifier is valid"""
        if not self.patterns['valid_identifier'].match(identifier):
            self.error_handler.add_error(
                f"Invalid identifier '{identifier}'",
                line,
                error_code="E001",
                suggestion="Identifiers must start with letter or underscore"
            )
            return False
        
        if identifier.lower() in self.reserved_keywords:
            self.error_handler.add_warning(
                f"Identifier '{identifier}' conflicts with reserved keyword",
                line,
                error_code="W001"
            )
        
        return True
    
    def check_balanced_brackets(self, text: str, line: int) -> bool:
        """Check for balanced brackets and parentheses"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for i, char in enumerate(text):
            if char in pairs:
                stack.append((char, i))
            elif char in pairs.values():
                if not stack:
                    self.error_handler.add_error(
                        f"Unmatched closing bracket '{char}'",
                        line, i + 1,
                        error_code="E002"
                    )
                    return False
                
                open_char, open_pos = stack.pop()
                if pairs[open_char] != char:
                    self.error_handler.add_error(
                        f"Mismatched brackets: '{open_char}' and '{char}'",
                        line, i + 1,
                        error_code="E003"
                    )
                    return False
        
        if stack:
            open_char, pos = stack[-1]
            self.error_handler.add_error(
                f"Unmatched opening bracket '{open_char}'",
                line, pos + 1,
                error_code="E004"
            )
            return False
        
        return True
    
    def check_string_literal(self, string_literal: str, line: int) -> bool:
        """Check string literal syntax"""
        if not self.patterns['valid_string'].match(string_literal):
            self.error_handler.add_error(
                f"Invalid string literal: {string_literal}",
                line,
                error_code="E005",
                suggestion="Strings must be enclosed in quotes"
            )
            return False
        
        # Check for unterminated strings
        if len(string_literal) < 2:
            self.error_handler.add_error(
                "Unterminated string literal",
                line,
                error_code="E006"
            )
            return False
        
        quote_char = string_literal[0]
        if string_literal[-1] != quote_char:
            self.error_handler.add_error(
                "Unterminated string literal",
                line,
                error_code="E006"
            )
            return False
        
        return True
    
    def check_number_literal(self, number_literal: str, line: int) -> bool:
        """Check number literal syntax"""
        if not self.patterns['valid_number'].match(number_literal):
            self.error_handler.add_error(
                f"Invalid number literal: {number_literal}",
                line,
                error_code="E007",
                suggestion="Numbers must be in decimal format"
            )
            return False
        
        return True

class SemanticAnalyzer:
    """Semantic analysis and validation"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.declared_variables = set()
        self.declared_functions = set()
        self.current_function = None
        self.loop_depth = 0
    
    def enter_function(self, function_name: str, line: int):
        """Enter function scope"""
        if function_name in self.declared_functions:
            self.error_handler.add_error(
                f"Function '{function_name}' already declared",
                line,
                error_code="E101"
            )
        else:
            self.declared_functions.add(function_name)
        
        self.current_function = function_name
    
    def exit_function(self):
        """Exit function scope"""
        self.current_function = None
    
    def enter_loop(self):
        """Enter loop scope"""
        self.loop_depth += 1
    
    def exit_loop(self):
        """Exit loop scope"""
        if self.loop_depth > 0:
            self.loop_depth -= 1
    
    def check_variable_declaration(self, var_name: str, line: int):
        """Check variable declaration"""
        if var_name in self.declared_variables:
            self.error_handler.add_warning(
                f"Variable '{var_name}' redeclared",
                line,
                error_code="W101"
            )
        else:
            self.declared_variables.add(var_name)
    
    def check_variable_usage(self, var_name: str, line: int):
        """Check variable usage"""
        if var_name not in self.declared_variables:
            self.error_handler.add_error(
                f"Undefined variable '{var_name}'",
                line,
                error_code="E102",
                suggestion=f"Declare variable with '%var {var_name}'"
            )
    
    def check_function_call(self, func_name: str, line: int):
        """Check function call"""
        if func_name not in self.declared_functions:
            self.error_handler.add_error(
                f"Undefined function '{func_name}'",
                line,
                error_code="E103",
                suggestion=f"Declare function with '%function {func_name}'"
            )
    
    def check_break_continue(self, statement: str, line: int):
        """Check break/continue statements"""
        if self.loop_depth == 0:
            self.error_handler.add_error(
                f"'{statement}' statement outside of loop",
                line,
                error_code="E104"
            )
    
    def check_return_statement(self, line: int):
        """Check return statement"""
        if self.current_function is None:
            self.error_handler.add_error(
                "'return' statement outside of function",
                line,
                error_code="E105"
            )

# Global error handler instance
global_error_handler = ErrorHandler()