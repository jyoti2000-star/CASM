#!/usr/bin/env python3
from typing import List, Optional
from .core.tokens import Token, LexerError
from .core.parser import ParseError

class CASMError(Exception):
    """Base class for CASM errors"""
    
    def __init__(self, message: str, line: int = 0, column: int = 0, context: str = ""):
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        super().__init__(self._format_error())
    
    def _format_error(self) -> str:
        """Format error message with context"""
        if self.line > 0:
            location = f"at line {self.line}, column {self.column}"
            if self.context:
                return f"Error {location}: {self.message}\n  {self.context}"
            else:
                return f"Error {location}: {self.message}"
        else:
            return f"Error: {self.message}"

class ErrorReporter:
    """Centralized error reporting"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def add_error(self, error: CASMError):
        """Add an error"""
        self.errors.append(error)
    
    def add_warning(self, message: str, line: int = 0, column: int = 0):
        """Add a warning"""
        warning = CASMError(message, line, column)
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are warnings"""
        return len(self.warnings) > 0
    
    def print_errors(self):
        """Print all errors"""
        for error in self.errors:
            print(f"[ERROR] {error}")
    
    def print_warnings(self):
        """Print all warnings"""
        for warning in self.warnings:
            print(f"[WARNING] {warning}")
    
    def print_summary(self):
        """Print error summary"""
        if self.has_errors():
            print(f"\n{len(self.errors)} error(s) found:")
            self.print_errors()
        
        if self.has_warnings():
            print(f"\n{len(self.warnings)} warning(s) found:")
            self.print_warnings()
    
    def clear(self):
        """Clear all errors and warnings"""
        self.errors.clear()
        self.warnings.clear()

# Global error reporter
error_reporter = ErrorReporter()