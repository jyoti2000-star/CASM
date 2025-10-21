from enum import Enum, auto
from typing import List, Tuple, Dict, Optional
from .tokens import TokenType, Token, SourceLocation
from .diagnostics import DiagnosticEngine

class LexerMode(Enum):
    NORMAL = auto()
    ASSEMBLY = auto()
    C_INLINE = auto()
    STRING = auto()
    COMMENT = auto()

class CASMLexer:
    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.diagnostics = DiagnosticEngine()
        
        # Lexer state
        self.position = 0
        self.line = 1
        self.column = 1
        self.mode_stack: List[LexerMode] = [LexerMode.NORMAL]
        
        # Keyword mapping
        self.keywords = self._init_keywords()
        
        # Operator mapping (order matters - longest first)
        self.operators = self._init_operators()
        
        # Punctuation mapping
        self.punctuation = self._init_punctuation()

    # ... (rest of implementation copied from original core.py)
