from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import sys

class DiagnosticLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    NOTE = "note"

@dataclass
class Diagnostic:
    level: DiagnosticLevel
    message: str
    location: Optional[object] = None
    notes: List[str] = field(default_factory=list)
    fix_suggestion: Optional[str] = None
    
    def format(self, with_colors: bool = True) -> str:
        colors = {
            DiagnosticLevel.ERROR: "\033[91m",
            DiagnosticLevel.WARNING: "\033[93m",
            DiagnosticLevel.INFO: "\033[94m",
            DiagnosticLevel.NOTE: "\033[96m",
        }
        reset = "\033[0m"
        
        if not with_colors:
            colors = {k: "" for k in colors}
            reset = ""
        
        color = colors[self.level]
        result = f"{color}{self.level.value}: {self.message}{reset}"
        
        if self.location:
            result = f"{self.location}: {result}"
            if getattr(self.location, 'raw_line', None):
                result += f"\n  {self.location.raw_line}"
                result += f"\n  {' ' * (self.location.column - 1)}^"
        
        for note in self.notes:
            result += f"\n{colors[DiagnosticLevel.NOTE]}note: {note}{reset}"
        
        if self.fix_suggestion:
            result += f"\n{colors[DiagnosticLevel.INFO]}suggestion: {self.fix_suggestion}{reset}"
        
        return result

class DiagnosticEngine:
    def __init__(self):
        self.diagnostics: List[Diagnostic] = []
        self.error_count = 0
        self.warning_count = 0
    
    def report(self, diag: Diagnostic):
        self.diagnostics.append(diag)
        if diag.level == DiagnosticLevel.ERROR:
            self.error_count += 1
        elif diag.level == DiagnosticLevel.WARNING:
            self.warning_count += 1
    
    def error(self, message: str, location: Optional[object] = None, **kwargs):
        self.report(Diagnostic(DiagnosticLevel.ERROR, message, location, **kwargs))
    
    def warning(self, message: str, location: Optional[object] = None, **kwargs):
        self.report(Diagnostic(DiagnosticLevel.WARNING, message, location, **kwargs))
    
    def info(self, message: str, location: Optional[object] = None, **kwargs):
        self.report(Diagnostic(DiagnosticLevel.INFO, message, location, **kwargs))
    
    def has_errors(self) -> bool:
        return self.error_count > 0
    
    def print_all(self, with_colors: bool = True):
        for diag in self.diagnostics:
            print(diag.format(with_colors), file=sys.stderr)
