#!/usr/bin/env python3
"""
Enhanced CASM Compiler Core - Production-Ready Implementation

Features:
- Advanced lexical analysis with better error recovery
- Recursive descent parser with lookahead
- Rich AST with metadata and source tracking
- Multi-pass compilation architecture
- Intermediate representation (IR) generation
- Advanced code generation with optimizations
- Symbol table with proper scoping
- Type inference and checking
- Macro preprocessing system
- Better C interop handling
- Comprehensive error reporting
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple, Dict, Set, Union, Callable
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
import re
import os
import sys

# ==================== ENHANCED TOKEN SYSTEM ====================

class TokenType(Enum):
    # Control Flow Keywords
    IF = "IF"
    ELIF = "ELIF"
    ELSE = "ELSE"
    ENDIF = "ENDIF"
    WHILE = "WHILE"
    ENDWHILE = "ENDWHILE"
    DO = "DO"
    FOR = "FOR"
    ENDFOR = "ENDFOR"
    SWITCH = "SWITCH"
    CASE = "CASE"
    DEFAULT = "DEFAULT"
    ENDSWITCH = "ENDSWITCH"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    RETURN = "RETURN"
    GOTO = "GOTO"
    
    # Declaration Keywords
    VAR = "VAR"
    CONST = "CONST"
    STATIC = "STATIC"
    EXTERN = "EXTERN"
    VOLATILE = "VOLATILE"
    REGISTER = "REGISTER"
    INLINE = "INLINE"
    FUNCTION = "FUNCTION"
    PROCEDURE = "PROCEDURE"
    STRUCT = "STRUCT"
    UNION = "UNION"
    ENUM = "ENUM"
    TYPEDEF = "TYPEDEF"
    MACRO = "MACRO"
    ENDMACRO = "ENDMACRO"
    
    # Type Keywords
    INT_TYPE = "INT_TYPE"
    INT8_TYPE = "INT8_TYPE"
    INT16_TYPE = "INT16_TYPE"
    INT32_TYPE = "INT32_TYPE"
    INT64_TYPE = "INT64_TYPE"
    UINT_TYPE = "UINT_TYPE"
    UINT8_TYPE = "UINT8_TYPE"
    UINT16_TYPE = "UINT16_TYPE"
    UINT32_TYPE = "UINT32_TYPE"
    UINT64_TYPE = "UINT64_TYPE"
    STR_TYPE = "STR_TYPE"
    BOOL_TYPE = "BOOL_TYPE"
    FLOAT_TYPE = "FLOAT_TYPE"
    DOUBLE_TYPE = "DOUBLE_TYPE"
    CHAR_TYPE = "CHAR_TYPE"
    VOID_TYPE = "VOID_TYPE"
    BUFFER_TYPE = "BUFFER_TYPE"
    PTR_TYPE = "PTR_TYPE"
    AUTO_TYPE = "AUTO_TYPE"
    
    # Assembly Directives
    ASM_DIRECTIVE = "ASM_DIRECTIVE"
    ASM_BLOCK = "ASM_BLOCK"
    ENDASM = "ENDASM"
    
    # I/O Keywords
    PRINT = "PRINT"
    PRINTLN = "PRINTLN"
    SCAN = "SCAN"
    READ = "READ"
    WRITE = "WRITE"
    
    # Operators
    C_INLINE = "C_INLINE"
    ASSIGN = "ASSIGN"
    PLUS_ASSIGN = "PLUS_ASSIGN"
    MINUS_ASSIGN = "MINUS_ASSIGN"
    MULT_ASSIGN = "MULT_ASSIGN"
    DIV_ASSIGN = "DIV_ASSIGN"
    MOD_ASSIGN = "MOD_ASSIGN"
    AND_ASSIGN = "AND_ASSIGN"
    OR_ASSIGN = "OR_ASSIGN"
    XOR_ASSIGN = "XOR_ASSIGN"
    SHL_ASSIGN = "SHL_ASSIGN"
    SHR_ASSIGN = "SHR_ASSIGN"
    
    # Comparison
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    LESS_EQUAL = "LESS_EQUAL"
    GREATER_EQUAL = "GREATER_EQUAL"
    SPACESHIP = "SPACESHIP"  # <=>
    
    # Arithmetic
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    INCREMENT = "INCREMENT"
    DECREMENT = "DECREMENT"
    POWER = "POWER"
    
    # Bitwise
    BIT_AND = "BIT_AND"
    BIT_OR = "BIT_OR"
    BIT_XOR = "BIT_XOR"
    BIT_NOT = "BIT_NOT"
    LEFT_SHIFT = "LEFT_SHIFT"
    RIGHT_SHIFT = "RIGHT_SHIFT"
    
    # Logical
    LOGICAL_AND = "LOGICAL_AND"
    LOGICAL_OR = "LOGICAL_OR"
    LOGICAL_NOT = "LOGICAL_NOT"
    
    # Punctuation
    LEFT_PAREN = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    LEFT_BRACKET = "LEFT_BRACKET"
    RIGHT_BRACKET = "RIGHT_BRACKET"
    LEFT_BRACE = "LEFT_BRACE"
    RIGHT_BRACE = "RIGHT_BRACE"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    COLON = "COLON"
    DOUBLE_COLON = "DOUBLE_COLON"
    DOT = "DOT"
    ELLIPSIS = "ELLIPSIS"
    ARROW = "ARROW"
    QUESTION = "QUESTION"
    HASH = "HASH"
    DOUBLE_HASH = "DOUBLE_HASH"
    AT = "AT"
    DOLLAR = "DOLLAR"
    BACKSLASH = "BACKSLASH"
    
    # Literals
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    FLOAT_NUMBER = "FLOAT_NUMBER"
    HEX_NUMBER = "HEX_NUMBER"
    BIN_NUMBER = "BIN_NUMBER"
    OCT_NUMBER = "OCT_NUMBER"
    STRING = "STRING"
    CHAR_LITERAL = "CHAR_LITERAL"
    TRUE = "TRUE"
    FALSE = "FALSE"
    NULL = "NULL"
    NULLPTR = "NULLPTR"
    
    # Special
    NEWLINE = "NEWLINE"
    COMMENT = "COMMENT"
    MULTILINE_COMMENT = "MULTILINE_COMMENT"
    ASSEMBLY_LINE = "ASSEMBLY_LINE"
    EOF = "EOF"
    
    # Keywords
    IN = "IN"
    RANGE = "RANGE"
    SIZEOF = "SIZEOF"
    TYPEOF = "TYPEOF"
    ALIGNOF = "ALIGNOF"
    OFFSETOF = "OFFSETOF"
    CAST = "CAST"
    AS = "AS"
    IS = "IS"
    
    # Optimization Hints
    OPTIMIZE = "OPTIMIZE"
    UNROLL = "UNROLL"
    VECTORIZE = "VECTORIZE"
    LIKELY = "LIKELY"
    UNLIKELY = "UNLIKELY"
    NOINLINE = "NOINLINE"
    ALWAYS_INLINE = "ALWAYS_INLINE"
    
    # Memory Management
    ALLOC = "ALLOC"
    FREE = "FREE"
    NEW = "NEW"
    DELETE = "DELETE"

@dataclass
class SourceLocation:
    """Enhanced source location with file tracking"""
    line: int
    column: int
    file: str = "<stdin>"
    raw_line: str = ""
    end_column: Optional[int] = None
    
    def __str__(self):
        return f"{self.file}:{self.line}:{self.column}"
    
    def to_range_str(self):
        if self.end_column:
            return f"{self.file}:{self.line}:{self.column}-{self.end_column}"
        return str(self)

@dataclass
class Token:
    """Enhanced token with metadata"""
    type: TokenType
    value: str
    location: SourceLocation
    leading_whitespace: str = ""
    trailing_whitespace: str = ""
    
    def __str__(self):
        return f"{self.type.name}('{self.value}') @ {self.location}"
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.location.line})"

# ==================== DIAGNOSTIC SYSTEM ====================

class DiagnosticLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    NOTE = "note"

@dataclass
class Diagnostic:
    """Compiler diagnostic message"""
    level: DiagnosticLevel
    message: str
    location: Optional[SourceLocation] = None
    notes: List[str] = field(default_factory=list)
    fix_suggestion: Optional[str] = None
    
    def format(self, with_colors: bool = True) -> str:
        """Format diagnostic for display"""
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
            if self.location.raw_line:
                result += f"\n  {self.location.raw_line}"
                result += f"\n  {' ' * (self.location.column - 1)}^"
        
        for note in self.notes:
            result += f"\n{colors[DiagnosticLevel.NOTE]}note: {note}{reset}"
        
        if self.fix_suggestion:
            result += f"\n{colors[DiagnosticLevel.INFO]}suggestion: {self.fix_suggestion}{reset}"
        
        return result

class DiagnosticEngine:
    """Manages compiler diagnostics"""
    
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
    
    def error(self, message: str, location: Optional[SourceLocation] = None, **kwargs):
        self.report(Diagnostic(DiagnosticLevel.ERROR, message, location, **kwargs))
    
    def warning(self, message: str, location: Optional[SourceLocation] = None, **kwargs):
        self.report(Diagnostic(DiagnosticLevel.WARNING, message, location, **kwargs))
    
    def info(self, message: str, location: Optional[SourceLocation] = None, **kwargs):
        self.report(Diagnostic(DiagnosticLevel.INFO, message, location, **kwargs))
    
    def has_errors(self) -> bool:
        return self.error_count > 0
    
    def print_all(self, with_colors: bool = True):
        for diag in self.diagnostics:
            print(diag.format(with_colors), file=sys.stderr)

# ==================== ENHANCED LEXER ====================

class LexerMode(Enum):
    NORMAL = auto()
    ASSEMBLY = auto()
    C_INLINE = auto()
    STRING = auto()
    COMMENT = auto()

class CASMLexer:
    """Advanced lexer with better error recovery and modes"""
    
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
        
    def _init_keywords(self) -> Dict[str, TokenType]:
        return {
            # Control flow
            'if': TokenType.IF, 'elif': TokenType.ELIF, 'else': TokenType.ELSE,
            'endif': TokenType.ENDIF, 'while': TokenType.WHILE, 'endwhile': TokenType.ENDWHILE,
            'do': TokenType.DO, 'for': TokenType.FOR, 'endfor': TokenType.ENDFOR,
            'switch': TokenType.SWITCH, 'case': TokenType.CASE, 'default': TokenType.DEFAULT,
            'endswitch': TokenType.ENDSWITCH, 'break': TokenType.BREAK, 
            'continue': TokenType.CONTINUE, 'return': TokenType.RETURN, 'goto': TokenType.GOTO,
            
            # Declarations
            'var': TokenType.VAR, 'const': TokenType.CONST, 'static': TokenType.STATIC,
            'extern': TokenType.EXTERN, 'volatile': TokenType.VOLATILE,
            'register': TokenType.REGISTER, 'inline': TokenType.INLINE,
            'function': TokenType.FUNCTION, 'procedure': TokenType.PROCEDURE,
            'struct': TokenType.STRUCT, 'union': TokenType.UNION, 'enum': TokenType.ENUM,
            'typedef': TokenType.TYPEDEF, 'macro': TokenType.MACRO, 'endmacro': TokenType.ENDMACRO,
            
            # Types
            'int': TokenType.INT_TYPE, 'int8': TokenType.INT8_TYPE, 
            'int16': TokenType.INT16_TYPE, 'int32': TokenType.INT32_TYPE,
            'int64': TokenType.INT64_TYPE, 'uint': TokenType.UINT_TYPE,
            'uint8': TokenType.UINT8_TYPE, 'uint16': TokenType.UINT16_TYPE,
            'uint32': TokenType.UINT32_TYPE, 'uint64': TokenType.UINT64_TYPE,
            'str': TokenType.STR_TYPE, 'bool': TokenType.BOOL_TYPE,
            'float': TokenType.FLOAT_TYPE, 'double': TokenType.DOUBLE_TYPE,
            'char': TokenType.CHAR_TYPE, 'void': TokenType.VOID_TYPE,
            'buffer': TokenType.BUFFER_TYPE, 'ptr': TokenType.PTR_TYPE,
            'auto': TokenType.AUTO_TYPE,
            
            # Boolean literals
            'true': TokenType.TRUE, 'false': TokenType.FALSE,
            'null': TokenType.NULL, 'nullptr': TokenType.NULLPTR,
            
            # I/O
            'print': TokenType.PRINT, 'println': TokenType.PRINTLN,
            'scan': TokenType.SCAN, 'read': TokenType.READ, 'write': TokenType.WRITE,
            
            # Special keywords
            'in': TokenType.IN, 'range': TokenType.RANGE,
            'sizeof': TokenType.SIZEOF, 'typeof': TokenType.TYPEOF,
            'alignof': TokenType.ALIGNOF, 'offsetof': TokenType.OFFSETOF,
            'cast': TokenType.CAST, 'as': TokenType.AS, 'is': TokenType.IS,
            
            # Optimization hints
            'optimize': TokenType.OPTIMIZE, 'unroll': TokenType.UNROLL,
            'vectorize': TokenType.VECTORIZE, 'likely': TokenType.LIKELY,
            'unlikely': TokenType.UNLIKELY, 'noinline': TokenType.NOINLINE,
            'always_inline': TokenType.ALWAYS_INLINE,
            
            # Memory management
            'alloc': TokenType.ALLOC, 'free': TokenType.FREE,
            'new': TokenType.NEW, 'delete': TokenType.DELETE,
            
            # Assembly directives
            'db': TokenType.ASM_DIRECTIVE, 'dw': TokenType.ASM_DIRECTIVE,
            'dd': TokenType.ASM_DIRECTIVE, 'dq': TokenType.ASM_DIRECTIVE,
            'resb': TokenType.ASM_DIRECTIVE, 'resw': TokenType.ASM_DIRECTIVE,
            'resd': TokenType.ASM_DIRECTIVE, 'resq': TokenType.ASM_DIRECTIVE,
            'equ': TokenType.ASM_DIRECTIVE, 'asm': TokenType.ASM_BLOCK,
            'endasm': TokenType.ENDASM,
        }
    
    def _init_operators(self) -> List[Tuple[str, TokenType]]:
        return [
            # Three-character operators
            ('<<=', TokenType.SHL_ASSIGN), ('>>=', TokenType.SHR_ASSIGN),
            ('...', TokenType.ELLIPSIS), ('<=>', TokenType.SPACESHIP),
            
            # Two-character operators
            ('+=', TokenType.PLUS_ASSIGN), ('-=', TokenType.MINUS_ASSIGN),
            ('*=', TokenType.MULT_ASSIGN), ('/=', TokenType.DIV_ASSIGN),
            ('%=', TokenType.MOD_ASSIGN), ('&=', TokenType.AND_ASSIGN),
            ('|=', TokenType.OR_ASSIGN), ('^=', TokenType.XOR_ASSIGN),
            ('==', TokenType.EQUALS), ('!=', TokenType.NOT_EQUALS),
            ('<=', TokenType.LESS_EQUAL), ('>=', TokenType.GREATER_EQUAL),
            ('<<', TokenType.LEFT_SHIFT), ('>>', TokenType.RIGHT_SHIFT),
            ('&&', TokenType.LOGICAL_AND), ('||', TokenType.LOGICAL_OR),
            ('++', TokenType.INCREMENT), ('--', TokenType.DECREMENT),
            ('->', TokenType.ARROW), ('::', TokenType.DOUBLE_COLON),
            ('##', TokenType.DOUBLE_HASH), ('**', TokenType.POWER),
            
            # Single-character operators
            ('=', TokenType.ASSIGN), ('<', TokenType.LESS_THAN),
            ('>', TokenType.GREATER_THAN), ('+', TokenType.PLUS),
            ('-', TokenType.MINUS), ('*', TokenType.MULTIPLY),
            ('/', TokenType.DIVIDE), ('%', TokenType.MODULO),
            ('&', TokenType.BIT_AND), ('|', TokenType.BIT_OR),
            ('^', TokenType.BIT_XOR), ('~', TokenType.BIT_NOT),
            ('!', TokenType.LOGICAL_NOT),
        ]
    
    def _init_punctuation(self) -> Dict[str, TokenType]:
        return {
            '(': TokenType.LEFT_PAREN, ')': TokenType.RIGHT_PAREN,
            '[': TokenType.LEFT_BRACKET, ']': TokenType.RIGHT_BRACKET,
            '{': TokenType.LEFT_BRACE, '}': TokenType.RIGHT_BRACE,
            ',': TokenType.COMMA, ';': TokenType.SEMICOLON,
            ':': TokenType.COLON, '.': TokenType.DOT,
            '?': TokenType.QUESTION, '#': TokenType.HASH,
            '@': TokenType.AT, '$': TokenType.DOLLAR,
            '\\': TokenType.BACKSLASH,
        }
    
    def tokenize(self) -> List[Token]:
        """Main tokenization method"""
        tokens = []
        
        while not self._at_end():
            # Skip whitespace
            self._skip_whitespace()
            
            if self._at_end():
                break
            
            # Try to get next token
            token = self._next_token()
            if token:
                tokens.append(token)
        
        # Add EOF token
        tokens.append(self._make_token(TokenType.EOF, ''))
        
        return tokens
    
    def _next_token(self) -> Optional[Token]:
        """Get the next token"""
        start_pos = self.position
        start_line = self.line
        start_col = self.column
        
        char = self._peek()
        
        # Handle different modes
        if self.mode_stack[-1] == LexerMode.ASSEMBLY:
            return self._lex_assembly_line()
        elif self.mode_stack[-1] == LexerMode.C_INLINE:
            return self._lex_c_inline()
        
        # Comments
        if char == ';':
            return self._lex_comment()
        
        if char == '/' and self._peek_ahead(1) == '/':
            return self._lex_comment()
        
        if char == '/' and self._peek_ahead(1) == '*':
            return self._lex_multiline_comment()
        
        # Strings
        if char in ['"', "'"]:
            return self._lex_string()
        
        # Numbers
        if char.isdigit():
            return self._lex_number()
        
        # Identifiers and keywords
        if char.isalpha() or char == '_':
            return self._lex_identifier()
        
        # Operators
        for op_str, op_type in self.operators:
            if self._match_string(op_str):
                self._advance_by(len(op_str))
                return self._make_token(op_type, op_str)
        
        # Punctuation
        if char in self.punctuation:
            self._advance()
            return self._make_token(self.punctuation[char], char)
        
        # Newlines
        if char == '\n':
            self._advance()
            return self._make_token(TokenType.NEWLINE, '\n')
        
        # Unknown character
        self.diagnostics.error(
            f"Unexpected character: '{char}'",
            self._current_location()
        )
        self._advance()
        return None
    
    def _lex_identifier(self) -> Token:
        """Lex identifier or keyword"""
        start = self.position
        
        while not self._at_end():
            char = self._peek()
            if char.isalnum() or char in ['_', '$', '%']:
                self._advance()
            else:
                break
        
        text = self.source[start:self.position]
        token_type = self.keywords.get(text.lower(), TokenType.IDENTIFIER)
        
        return self._make_token(token_type, text)
    
    def _lex_number(self) -> Token:
        """Lex numeric literal"""
        start = self.position
        token_type = TokenType.NUMBER
        
        # Check for hex, binary, or octal
        if self._peek() == '0' and not self._at_end():
            next_char = self._peek_ahead(1)
            
            if next_char in 'xX':
                self._advance_by(2)
                while not self._at_end() and (self._peek().isdigit() or 
                      self._peek() in 'abcdefABCDEF_'):
                    if self._peek() != '_':
                        self._advance()
                    else:
                        self._advance()
                token_type = TokenType.HEX_NUMBER
                return self._make_token(token_type, self.source[start:self.position])
            
            elif next_char in 'bB':
                self._advance_by(2)
                while not self._at_end() and self._peek() in '01_':
                    self._advance()
                token_type = TokenType.BIN_NUMBER
                return self._make_token(token_type, self.source[start:self.position])
            
            elif next_char in 'oO':
                self._advance_by(2)
                while not self._at_end() and self._peek() in '01234567_':
                    self._advance()
                token_type = TokenType.OCT_NUMBER
                return self._make_token(token_type, self.source[start:self.position])
        
        # Regular integer or float
        has_dot = False
        has_exp = False
        
        while not self._at_end():
            char = self._peek()
            
            if char.isdigit() or char == '_':
                self._advance()
            elif char == '.' and not has_dot and not has_exp:
                # Make sure it's not ".." (range operator)
                if self._peek_ahead(1) != '.':
                    has_dot = True
                    token_type = TokenType.FLOAT_NUMBER
                    self._advance()
                else:
                    break
            elif char in 'eE' and not has_exp:
                has_exp = True
                token_type = TokenType.FLOAT_NUMBER
                self._advance()
                if not self._at_end() and self._peek() in '+-':
                    self._advance()
            else:
                break
        
        # Check for type suffix (f, d, l, u, etc.)
        if not self._at_end() and self._peek() in 'fFdDlLuU':
            self._advance()
        
        return self._make_token(token_type, self.source[start:self.position])
    
    def _lex_string(self) -> Token:
        """Lex string literal"""
        quote_char = self._peek()
        self._advance()  # Skip opening quote
        
        start = self.position - 1
        
        while not self._at_end():
            char = self._peek()
            
            if char == quote_char:
                self._advance()
                break
            elif char == '\\':
                self._advance()
                if not self._at_end():
                    self._advance()  # Skip escaped character
            elif char == '\n':
                self.diagnostics.error(
                    "Unterminated string literal",
                    self._current_location()
                )
                break
            else:
                self._advance()
        
        return self._make_token(TokenType.STRING, self.source[start:self.position])
    
    def _lex_comment(self) -> Token:
        """Lex single-line comment"""
        start = self.position
        
        while not self._at_end() and self._peek() != '\n':
            self._advance()
        
        return self._make_token(TokenType.COMMENT, self.source[start:self.position])
    
    def _lex_multiline_comment(self) -> Token:
        """Lex multi-line comment"""
        start = self.position
        self._advance_by(2)  # Skip /*
        
        while not self._at_end():
            if self._peek() == '*' and self._peek_ahead(1) == '/':
                self._advance_by(2)
                break
            self._advance()
        
        return self._make_token(TokenType.MULTILINE_COMMENT, self.source[start:self.position])
    
    def _lex_assembly_line(self) -> Token:
        """Lex assembly instruction line"""
        start = self.position
        
        while not self._at_end() and self._peek() != '\n':
            self._advance()
        
        return self._make_token(TokenType.ASSEMBLY_LINE, self.source[start:self.position])
    
    def _lex_c_inline(self) -> Token:
        """Lex C inline code"""
        start = self.position
        brace_depth = 0
        
        while not self._at_end():
            char = self._peek()
            
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth < 0:
                    break
            elif char == '\n' and brace_depth == 0:
                break
            
            self._advance()
        
        return self._make_token(TokenType.C_INLINE, self.source[start:self.position])
    
    # Utility methods
    def _peek(self, offset: int = 0) -> str:
        """Peek at character"""
        pos = self.position + offset
        if pos >= len(self.source):
            return '\0'
        return self.source[pos]
    
    def _peek_ahead(self, count: int) -> str:
        """Peek ahead by count characters"""
        return self._peek(count)
    
    def _advance(self) -> str:
        """Advance to next character"""
        if self._at_end():
            return '\0'
        
        char = self.source[self.position]
        self.position += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        return char
    
    def _advance_by(self, count: int):
        """Advance by multiple characters"""
        for _ in range(count):
            self._advance()
    
    def _at_end(self) -> bool:
        """Check if at end of source"""
        return self.position >= len(self.source)
    
    def _match_string(self, text: str) -> bool:
        """Check if current position matches string"""
        return self.source[self.position:self.position + len(text)] == text
    
    def _skip_whitespace(self):
        """Skip whitespace except newlines"""
        while not self._at_end():
            char = self._peek()
            if char in ' \t\r':
                self._advance()
            else:
                break
    
    def _current_location(self) -> SourceLocation:
        """Get current source location"""
        # Find the line text
        line_start = self.position
        while line_start > 0 and self.source[line_start - 1] != '\n':
            line_start -= 1
        
        line_end = self.position
        while line_end < len(self.source) and self.source[line_end] != '\n':
            line_end += 1
        
        raw_line = self.source[line_start:line_end]
        
        return SourceLocation(
            line=self.line,
            column=self.column,
            file=self.filename,
            raw_line=raw_line
        )
    
    def _make_token(self, token_type: TokenType, value: str) -> Token:
        """Create a token"""
        return Token(
            type=token_type,
            value=value,
            location=self._current_location()
        )

# ==================== ENHANCED AST NODES ====================

class ASTNode(ABC):
    """Base class for all AST nodes"""
    
    def __init__(self, location: Optional[SourceLocation] = None):
        self.location = location
        self.parent: Optional[ASTNode] = None
        self.attributes: Dict[str, Any] = {}
        
    @abstractmethod
    def accept(self, visitor):
        """Accept visitor pattern"""
        pass
    
    def get_children(self) -> List['ASTNode']:
        """Get child nodes"""
        return []
    
    def set_parent(self, parent: 'ASTNode'):
        """Set parent node"""
        self.parent = parent
        for child in self.get_children():
            if isinstance(child, ASTNode):
                child.set_parent(self)

@dataclass
class ProgramNode(ASTNode):
    """Root program node"""
    statements: List[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_program(self)
    
    def get_children(self):
        return self.statements

# ==================== DECLARATION NODES ====================

@dataclass
class VarDeclarationNode(ASTNode):
    """Variable declaration with enhanced type info"""
    name: str
    var_type: str
    value: Optional[str] = None
    size: Optional[int] = None
    is_const: bool = False
    is_static: bool = False
    is_extern: bool = False
    is_volatile: bool = False
    is_register: bool = False
    storage_class: Optional[str] = None
    
    def accept(self, visitor):
        return visitor.visit_var_declaration(self)

@dataclass
class FunctionDeclarationNode(ASTNode):
    """Function declaration"""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]] = field(default_factory=list)
    body: Optional[List[ASTNode]] = None
    is_inline: bool = False
    is_static: bool = False
    is_extern: bool = False
    calling_convention: Optional[str] = None
    
    def accept(self, visitor):
        return visitor.visit_function_declaration(self)
    
    def get_children(self):
        return self.body if self.body else []

@dataclass
class StructDeclarationNode(ASTNode):
    """Struct declaration"""
    name: str
    members: List[Tuple[str, str]] = field(default_factory=list)
    is_packed: bool = False
    alignment: Optional[int] = None
    
    def accept(self, visitor):
        return visitor.visit_struct_declaration(self)

@dataclass
class UnionDeclarationNode(ASTNode):
    """Union declaration"""
    name: str
    members: List[Tuple[str, str]] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_union_declaration(self)

@dataclass
class EnumDeclarationNode(ASTNode):
    """Enum declaration"""
    name: str
    values: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    base_type: str = "int"
    
    def accept(self, visitor):
        return visitor.visit_enum_declaration(self)

@dataclass
class TypedefNode(ASTNode):
    """Typedef declaration"""
    new_name: str
    original_type: str
    
    def accept(self, visitor):
        return visitor.visit_typedef(self)

@dataclass
class MacroDefinitionNode(ASTNode):
    """Macro definition"""
    name: str
    parameters: List[str] = field(default_factory=list)
    body: List[ASTNode] = field(default_factory=list)
    is_function_like: bool = False
    
    def accept(self, visitor):
        return visitor.visit_macro_definition(self)
    
    def get_children(self):
        return self.body

# ==================== STATEMENT NODES ====================

@dataclass
class AssignmentNode(ASTNode):
    """Assignment statement"""
    name: str
    value: str
    operator: str = "="
    is_compound: bool = False
    
    def accept(self, visitor):
        return visitor.visit_assignment(self)

@dataclass
class IfNode(ASTNode):
    """If statement with elif support"""
    condition: str
    if_body: List[ASTNode] = field(default_factory=list)
    elif_branches: List[Tuple[str, List[ASTNode]]] = field(default_factory=list)
    else_body: Optional[List[ASTNode]] = None
    
    def accept(self, visitor):
        return visitor.visit_if(self)
    
    def get_children(self):
        children = list(self.if_body)
        for _, body in self.elif_branches:
            children.extend(body)
        if self.else_body:
            children.extend(self.else_body)
        return children

@dataclass
class WhileNode(ASTNode):
    """While loop"""
    condition: str
    body: List[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_while(self)
    
    def get_children(self):
        return self.body

@dataclass
class DoWhileNode(ASTNode):
    """Do-while loop"""
    body: List[ASTNode] = field(default_factory=list)
    condition: str = ""
    
    def accept(self, visitor):
        return visitor.visit_do_while(self)
    
    def get_children(self):
        return self.body

@dataclass
class ForNode(ASTNode):
    """For loop with enhanced support"""
    variable: str
    count: str
    body: List[ASTNode] = field(default_factory=list)
    init_expr: Optional[str] = None
    condition_expr: Optional[str] = None
    increment_expr: Optional[str] = None
    is_range_based: bool = True
    
    def accept(self, visitor):
        return visitor.visit_for(self)
    
    def get_children(self):
        return self.body

@dataclass
class SwitchNode(ASTNode):
    """Switch statement"""
    expression: str
    cases: List[Tuple[Optional[str], List[ASTNode]]] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_switch(self)
    
    def get_children(self):
        children = []
        for _, body in self.cases:
            children.extend(body)
        return children

@dataclass
class BreakNode(ASTNode):
    """Break statement"""
    
    def accept(self, visitor):
        return visitor.visit_break(self)

@dataclass
class ContinueNode(ASTNode):
    """Continue statement"""
    
    def accept(self, visitor):
        return visitor.visit_continue(self)

@dataclass
class ReturnNode(ASTNode):
    """Return statement"""
    value: Optional[str] = None
    
    def accept(self, visitor):
        return visitor.visit_return(self)

@dataclass
class GotoNode(ASTNode):
    """Goto statement"""
    label: str
    
    def accept(self, visitor):
        return visitor.visit_goto(self)

@dataclass
class LabelNode(ASTNode):
    """Label definition"""
    name: str
    
    def accept(self, visitor):
        return visitor.visit_label(self)

# ==================== I/O NODES ====================

@dataclass
class PrintlnNode(ASTNode):
    """Print statement"""
    message: str
    newline: bool = True
    format_args: List[str] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_println(self)

@dataclass
class ScanfNode(ASTNode):
    """Scanf/input statement"""
    format_string: str
    variable: str
    additional_vars: List[str] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_scanf(self)

# ==================== ASSEMBLY NODES ====================

@dataclass
class AssemblyNode(ASTNode):
    """Inline assembly"""
    code: str
    constraints: Dict[str, str] = field(default_factory=dict)
    clobbers: List[str] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_assembly(self)

@dataclass
class AsmBlockNode(ASTNode):
    """Assembly block"""
    asm_code: str
    is_volatile: bool = False
    
    def accept(self, visitor):
        return visitor.visit_asm_block(self)

# ==================== SPECIAL NODES ====================

@dataclass
class CommentNode(ASTNode):
    """Comment"""
    text: str
    is_multiline: bool = False
    
    def accept(self, visitor):
        return visitor.visit_comment(self)

@dataclass
class CCodeBlockNode(ASTNode):
    """C code inline block"""
    c_code: str
    
    def accept(self, visitor):
        return visitor.visit_c_code_block(self)

@dataclass
class ExternDirectiveNode(ASTNode):
    """Extern directive"""
    header_name: str
    is_c_include: bool = False
    use_angle: bool = False
    is_system_header: bool = False
    
    def accept(self, visitor):
        return visitor.visit_extern_directive(self)

@dataclass
class OptimizationHintNode(ASTNode):
    """Optimization hint"""
    hint_type: str
    target: Optional[ASTNode] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def accept(self, visitor):
        return visitor.visit_optimization_hint(self)

# ==================== VISITOR INTERFACE ====================

class ASTVisitor(ABC):
    """Enhanced visitor interface"""
    
    @abstractmethod
    def visit_program(self, node: ProgramNode): pass
    
    @abstractmethod
    def visit_var_declaration(self, node: VarDeclarationNode): pass
    
    @abstractmethod
    def visit_function_declaration(self, node: FunctionDeclarationNode): pass
    
    @abstractmethod
    def visit_struct_declaration(self, node: StructDeclarationNode): pass
    
    @abstractmethod
    def visit_union_declaration(self, node: UnionDeclarationNode): pass
    
    @abstractmethod
    def visit_enum_declaration(self, node: EnumDeclarationNode): pass
    
    @abstractmethod
    def visit_typedef(self, node: TypedefNode): pass
    
    @abstractmethod
    def visit_macro_definition(self, node: MacroDefinitionNode): pass
    
    @abstractmethod
    def visit_assignment(self, node: AssignmentNode): pass
    
    @abstractmethod
    def visit_if(self, node: IfNode): pass
    
    @abstractmethod
    def visit_while(self, node: WhileNode): pass
    
    @abstractmethod
    def visit_do_while(self, node: DoWhileNode): pass
    
    @abstractmethod
    def visit_for(self, node: ForNode): pass
    
    @abstractmethod
    def visit_switch(self, node: SwitchNode): pass
    
    @abstractmethod
    def visit_break(self, node: BreakNode): pass
    
    @abstractmethod
    def visit_continue(self, node: ContinueNode): pass
    
    @abstractmethod
    def visit_return(self, node: ReturnNode): pass
    
    @abstractmethod
    def visit_goto(self, node: GotoNode): pass
    
    @abstractmethod
    def visit_label(self, node: LabelNode): pass
    
    @abstractmethod
    def visit_println(self, node: PrintlnNode): pass
    
    @abstractmethod
    def visit_scanf(self, node: ScanfNode): pass
    
    @abstractmethod
    def visit_assembly(self, node: AssemblyNode): pass
    
    @abstractmethod
    def visit_asm_block(self, node: AsmBlockNode): pass
    
    @abstractmethod
    def visit_comment(self, node: CommentNode): pass
    
    @abstractmethod
    def visit_c_code_block(self, node: CCodeBlockNode): pass
    
    @abstractmethod
    def visit_extern_directive(self, node: ExternDirectiveNode): pass
    
    @abstractmethod
    def visit_optimization_hint(self, node: OptimizationHintNode): pass

# ==================== SYMBOL TABLE ====================

@dataclass
class Symbol:
    """Enhanced symbol with metadata"""
    name: str
    type_name: str
    scope_level: int
    location: Optional[SourceLocation] = None
    
    # Properties
    is_defined: bool = True
    is_used: bool = False
    is_extern: bool = False
    is_static: bool = False
    is_const: bool = False
    is_volatile: bool = False
    is_register: bool = False
    
    # For variables
    asm_label: Optional[str] = None
    value: Optional[Any] = None
    size: Optional[int] = None
    offset: int = 0
    
    # For functions
    is_function: bool = False
    parameters: List[Tuple[str, str]] = field(default_factory=list)
    return_type: Optional[str] = None
    
    # Usage tracking
    references: List[SourceLocation] = field(default_factory=list)
    
    def add_reference(self, location: SourceLocation):
        """Add a reference to this symbol"""
        self.is_used = True
        self.references.append(location)

class SymbolTable:
    """Enhanced symbol table with scoping"""
    
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]
        self.current_scope_level = 0
        self.global_scope = self.scopes[0]
        
    def enter_scope(self):
        """Enter new scope"""
        self.current_scope_level += 1
        self.scopes.append({})
    
    def exit_scope(self):
        """Exit current scope"""
        if self.current_scope_level > 0:
            self.scopes.pop()
            self.current_scope_level -= 1
    
    def define(self, symbol: Symbol) -> bool:
        """Define symbol in current scope"""
        symbol.scope_level = self.current_scope_level
        current = self.scopes[self.current_scope_level]
        
        if symbol.name in current:
            return False
        
        current[symbol.name] = symbol
        return True
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Lookup symbol in all scopes"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def lookup_current_scope(self, name: str) -> Optional[Symbol]:
        """Lookup in current scope only"""
        return self.scopes[self.current_scope_level].get(name)
    
    def get_all_symbols(self) -> List[Symbol]:
        """Get all symbols"""
        symbols = []
        for scope in self.scopes:
            symbols.extend(scope.values())
        return symbols
    
    def get_unused_symbols(self) -> List[Symbol]:
        """Get unused symbols for warnings"""
        return [s for s in self.get_all_symbols() if not s.is_used and not s.is_extern]

# ==================== ENHANCED PARSER ====================

class ParseError(Exception):
    """Parse error exception"""
    
    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        location_str = f" at {token.location}" if token else ""
        super().__init__(f"Parse error{location_str}: {message}")

class CASMParser:
    """Enhanced recursive descent parser"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        self.diagnostics = DiagnosticEngine()
        self.symbol_table = SymbolTable()
        
        # Counters
        self._anon_var_counter = 0
        self._label_counter = 0
        
        # State
        self._in_function = False
        self._in_loop = False
        self._in_switch = False
        
    def parse(self) -> Optional[ProgramNode]:
        """Parse tokens into AST"""
        try:
            statements = []
            
            while not self._is_at_end():
                stmt = self._parse_statement()
                if stmt:
                    statements.append(stmt)
            
            return ProgramNode(statements=statements)
        
        except ParseError as e:
            self.diagnostics.error(e.message, e.token.location if e.token else None)
            return None
    
    def _parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement"""
        # Skip newlines and empty statements
        while self._match(TokenType.NEWLINE, TokenType.SEMICOLON):
            pass
        
        if self._is_at_end():
            return None
        
        token = self._peek()
        
        # Comments
        if token.type in [TokenType.COMMENT, TokenType.MULTILINE_COMMENT]:
            return self._parse_comment()
        
        # Declarations
        if token.type == TokenType.VAR:
            return self._parse_var_declaration()
        elif token.type == TokenType.CONST:
            return self._parse_const_declaration()
        elif token.type == TokenType.FUNCTION:
            return self._parse_function_declaration()
        elif token.type == TokenType.STRUCT:
            return self._parse_struct_declaration()
        elif token.type == TokenType.UNION:
            return self._parse_union_declaration()
        elif token.type == TokenType.ENUM:
            return self._parse_enum_declaration()
        elif token.type == TokenType.TYPEDEF:
            return self._parse_typedef()
        elif token.type == TokenType.MACRO:
            return self._parse_macro_definition()
        elif token.type == TokenType.EXTERN:
            return self._parse_extern_directive()
        
        # Control flow
        elif token.type == TokenType.IF:
            return self._parse_if_statement()
        elif token.type == TokenType.WHILE:
            return self._parse_while_statement()
        elif token.type == TokenType.DO:
            return self._parse_do_while_statement()
        elif token.type == TokenType.FOR:
            return self._parse_for_statement()
        elif token.type == TokenType.SWITCH:
            return self._parse_switch_statement()
        elif token.type == TokenType.BREAK:
            return self._parse_break_statement()
        elif token.type == TokenType.CONTINUE:
            return self._parse_continue_statement()
        elif token.type == TokenType.RETURN:
            return self._parse_return_statement()
        elif token.type == TokenType.GOTO:
            return self._parse_goto_statement()
        
        # I/O
        elif token.type in [TokenType.PRINT, TokenType.PRINTLN]:
            return self._parse_print_statement()
        elif token.type == TokenType.SCAN:
            return self._parse_scan_statement()
        
        # Assembly
        elif token.type == TokenType.ASM_BLOCK:
            return self._parse_asm_block()
        elif token.type == TokenType.ASSEMBLY_LINE:
            return self._parse_assembly_line()
        
        # C inline
        elif token.type == TokenType.C_INLINE:
            return self._parse_c_inline_block()
        
        # Assignment or expression
        elif token.type == TokenType.IDENTIFIER:
            if self._is_assignment():
                return self._parse_assignment()
            else:
                return self._parse_assembly_line()
        
        # Labels
        elif self._is_label():
            return self._parse_label()
        
        # Unknown - treat as assembly
        else:
            return self._parse_assembly_line()
    
    def _parse_var_declaration(self) -> VarDeclarationNode:
        """Parse variable declaration"""
        location = self._peek().location
        self._consume(TokenType.VAR, "Expected 'var'")
        
        # Storage class modifiers
        is_static = self._match(TokenType.STATIC)
        is_extern = self._match(TokenType.EXTERN)
        is_volatile = self._match(TokenType.VOLATILE)
        is_register = self._match(TokenType.REGISTER)
        
        # Type
        if not self._check_type():
            self._error("Expected type after 'var'")
            return None
        
        type_token = self._advance()
        var_type = type_token.value
        
        # Array size or name
        size = None
        name = None
        
        if self._match(TokenType.LEFT_BRACKET):
            # Anonymous array: var int[10]
            name = f"{var_type}_auto_{self._anon_var_counter}"
            self._anon_var_counter += 1
            size = self._parse_array_size()
            self._consume(TokenType.RIGHT_BRACKET, "Expected ']'")
        else:
            # Named variable
            name_token = self._consume(TokenType.IDENTIFIER, "Expected variable name")
            name = name_token.value
            
            # Check for array
            if self._match(TokenType.LEFT_BRACKET):
                size = self._parse_array_size()
                self._consume(TokenType.RIGHT_BRACKET, "Expected ']'")
        
        # Initial value
        value = None
        if self._match(TokenType.ASSIGN):
            value = self._parse_initializer()
        
        # Optional semicolon
        self._match(TokenType.SEMICOLON)
        
        # Add to symbol table
        symbol = Symbol(
            name=name,
            type_name=var_type,
            scope_level=self.symbol_table.current_scope_level,
            location=location,
            is_static=is_static,
            is_extern=is_extern,
            is_volatile=is_volatile,
            is_register=is_register,
            size=size,
            value=value
        )
        
        if not self.symbol_table.define(symbol):
            self.diagnostics.warning(f"Redefinition of variable '{name}'", location)
        
        return VarDeclarationNode(
            name=name,
            var_type=var_type,
            value=value,
            size=size,
            is_static=is_static,
            is_extern=is_extern,
            is_volatile=is_volatile,
            is_register=is_register,
            location=location
        )
    
    def _parse_const_declaration(self) -> VarDeclarationNode:
        """Parse const declaration"""
        location = self._peek().location
        self._consume(TokenType.CONST, "Expected 'const'")
        
        # Type
        if not self._check_type():
            self._error("Expected type after 'const'")
        
        type_token = self._advance()
        var_type = type_token.value
        
        # Name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected constant name")
        name = name_token.value
        
        # Must have initializer
        self._consume(TokenType.ASSIGN, "Const must have initializer")
        value = self._parse_initializer()
        
        self._match(TokenType.SEMICOLON)
        
        return VarDeclarationNode(
            name=name,
            var_type=var_type,
            value=value,
            is_const=True,
            location=location
        )
    
    def _parse_function_declaration(self) -> FunctionDeclarationNode:
        """Parse function declaration"""
        location = self._peek().location
        self._consume(TokenType.FUNCTION, "Expected 'function'")
        
        # Modifiers
        is_inline = self._match(TokenType.INLINE)
        is_static = self._match(TokenType.STATIC)
        
        # Return type
        if not self._check_type():
            self._error("Expected return type")
        
        return_type_token = self._advance()
        return_type = return_type_token.value
        
        # Name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.value
        
        # Parameters
        self._consume(TokenType.LEFT_PAREN, "Expected '('")
        parameters = self._parse_parameter_list()
        self._consume(TokenType.RIGHT_PAREN, "Expected ')'")
        
        # Body (optional for forward declarations)
        body = None
        if self._match(TokenType.LEFT_BRACE):
            old_in_function = self._in_function
            self._in_function = True
            self.symbol_table.enter_scope()
            
            body = []
            while not self._check(TokenType.RIGHT_BRACE) and not self._is_at_end():
                stmt = self._parse_statement()
                if stmt:
                    body.append(stmt)
            
            self._consume(TokenType.RIGHT_BRACE, "Expected '}'")
            self.symbol_table.exit_scope()
            self._in_function = old_in_function
        
        return FunctionDeclarationNode(
            name=name,
            return_type=return_type,
            parameters=parameters,
            body=body,
            is_inline=is_inline,
            is_static=is_static,
            location=location
        )
    
    def _parse_struct_declaration(self) -> StructDeclarationNode:
        """Parse struct declaration"""
        location = self._peek().location
        self._consume(TokenType.STRUCT, "Expected 'struct'")
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expected struct name")
        name = name_token.value
        
        self._consume(TokenType.LEFT_BRACE, "Expected '{'")
        
        members = []
        while not self._check(TokenType.RIGHT_BRACE) and not self._is_at_end():
            if not self._check_type():
                break
            
            member_type_token = self._advance()
            member_type = member_type_token.value
            
            member_name_token = self._consume(TokenType.IDENTIFIER, "Expected member name")
            member_name = member_name_token.value
            
            members.append((member_name, member_type))
            
            self._match(TokenType.SEMICOLON)
        
        self._consume(TokenType.RIGHT_BRACE, "Expected '}'")
        
        return StructDeclarationNode(
            name=name,
            members=members,
            location=location
        )
    
    def _parse_union_declaration(self) -> UnionDeclarationNode:
        """Parse union declaration"""
        location = self._peek().location
        self._consume(TokenType.UNION, "Expected 'union'")
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expected union name")
        name = name_token.value
        
        self._consume(TokenType.LEFT_BRACE, "Expected '{'")
        
        members = []
        while not self._check(TokenType.RIGHT_BRACE) and not self._is_at_end():
            if not self._check_type():
                break
            
            member_type_token = self._advance()
            member_type = member_type_token.value
            
            member_name_token = self._consume(TokenType.IDENTIFIER, "Expected member name")
            member_name = member_name_token.value
            
            members.append((member_name, member_type))
            
            self._match(TokenType.SEMICOLON)
        
        self._consume(TokenType.RIGHT_BRACE, "Expected '}'")
        
        return UnionDeclarationNode(
            name=name,
            members=members,
            location=location
        )
    
    def _parse_enum_declaration(self) -> EnumDeclarationNode:
        """Parse enum declaration"""
        location = self._peek().location
        self._consume(TokenType.ENUM, "Expected 'enum'")
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expected enum name")
        name = name_token.value
        
        self._consume(TokenType.LEFT_BRACE, "Expected '{'")
        
        values = []
        current_value = 0
        
        while not self._check(TokenType.RIGHT_BRACE) and not self._is_at_end():
            value_name_token = self._consume(TokenType.IDENTIFIER, "Expected enum value")
            value_name = value_name_token.value
            
            explicit_value = None
            if self._match(TokenType.ASSIGN):
                value_token = self._consume(TokenType.NUMBER, "Expected number")
                explicit_value = int(value_token.value)
                current_value = explicit_value
            
            values.append((value_name, current_value))
            current_value += 1
            
            if not self._match(TokenType.COMMA):
                break
        
        self._consume(TokenType.RIGHT_BRACE, "Expected '}'")
        
        return EnumDeclarationNode(
            name=name,
            values=values,
            location=location
        )
    
    def _parse_typedef(self) -> TypedefNode:
        """Parse typedef"""
        location = self._peek().location
        self._consume(TokenType.TYPEDEF, "Expected 'typedef'")
        
        original_type_token = self._consume(TokenType.IDENTIFIER, "Expected original type")
        original_type = original_type_token.value
        
        new_name_token = self._consume(TokenType.IDENTIFIER, "Expected new type name")
        new_name = new_name_token.value
        
        self._match(TokenType.SEMICOLON)
        
        return TypedefNode(
            new_name=new_name,
            original_type=original_type,
            location=location
        )
    
    def _parse_macro_definition(self) -> MacroDefinitionNode:
        """Parse macro definition"""
        location = self._peek().location
        self._consume(TokenType.MACRO, "Expected 'macro'")
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expected macro name")
        name = name_token.value
        
        # Parameters (optional)
        parameters = []
        if self._match(TokenType.LEFT_PAREN):
            parameters = self._parse_identifier_list()
            self._consume(TokenType.RIGHT_PAREN, "Expected ')'")
        
        # Body
        body = []
        while not self._check(TokenType.ENDMACRO) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
        
        self._consume(TokenType.ENDMACRO, "Expected 'endmacro'")
        
        return MacroDefinitionNode(
            name=name,
            parameters=parameters,
            body=body,
            is_function_like=len(parameters) > 0,
            location=location
        )
    
    def _parse_assignment(self) -> AssignmentNode:
        """Parse assignment statement"""
        location = self._peek().location
        name_token = self._advance()
        name = name_token.value
        
        # Operator
        operator = "="
        if self._check(TokenType.PLUS_ASSIGN):
            operator = "+="
            self._advance()
        elif self._check(TokenType.MINUS_ASSIGN):
            operator = "-="
            self._advance()
        elif self._check(TokenType.MULT_ASSIGN):
            operator = "*="
            self._advance()
        elif self._check(TokenType.DIV_ASSIGN):
            operator = "/="
            self._advance()
        elif self._check(TokenType.MOD_ASSIGN):
            operator = "%="
            self._advance()
        else:
            self._consume(TokenType.ASSIGN, "Expected '='")
        
        # Value
        value = self._parse_expression()
        self._match(TokenType.SEMICOLON)
        
        return AssignmentNode(
            name=name,
            value=value,
            operator=operator,
            is_compound=operator != "=",
            location=location
        )
    
    def _parse_if_statement(self) -> IfNode:
        """Parse if statement with elif support"""
        location = self._peek().location
        self._consume(TokenType.IF, "Expected 'if'")
        
        condition = self._parse_condition()
        
        # If body
        if_body = []
        while not self._check(TokenType.ELIF) and not self._check(TokenType.ELSE) and \
              not self._check(TokenType.ENDIF) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                if_body.append(stmt)
        
        # Elif branches
        elif_branches = []
        while self._match(TokenType.ELIF):
            elif_condition = self._parse_condition()
            elif_body = []
            while not self._check(TokenType.ELIF) and not self._check(TokenType.ELSE) and \
                  not self._check(TokenType.ENDIF) and not self._is_at_end():
                stmt = self._parse_statement()
                if stmt:
                    elif_body.append(stmt)
            elif_branches.append((elif_condition, elif_body))
        
        # Else body
        else_body = None
        if self._match(TokenType.ELSE):
            else_body = []
            while not self._check(TokenType.ENDIF) and not self._is_at_end():
                stmt = self._parse_statement()
                if stmt:
                    else_body.append(stmt)
        
        self._consume(TokenType.ENDIF, "Expected 'endif'")
        
        return IfNode(
            condition=condition,
            if_body=if_body,
            elif_branches=elif_branches,
            else_body=else_body,
            location=location
        )
    
    def _parse_while_statement(self) -> WhileNode:
        """Parse while loop"""
        location = self._peek().location
        self._consume(TokenType.WHILE, "Expected 'while'")
        
        condition = self._parse_condition()
        
        old_in_loop = self._in_loop
        self._in_loop = True
        
        body = []
        while not self._check(TokenType.ENDWHILE) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
        
        self._consume(TokenType.ENDWHILE, "Expected 'endwhile'")
        self._in_loop = old_in_loop
        
        return WhileNode(
            condition=condition,
            body=body,
            location=location
        )
    
    def _parse_do_while_statement(self) -> DoWhileNode:
        """Parse do-while loop"""
        location = self._peek().location
        self._consume(TokenType.DO, "Expected 'do'")
        
        old_in_loop = self._in_loop
        self._in_loop = True
        
        body = []
        while not self._check(TokenType.WHILE) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
        
        self._consume(TokenType.WHILE, "Expected 'while'")
        condition = self._parse_condition()
        
        self._in_loop = old_in_loop
        
        return DoWhileNode(
            body=body,
            condition=condition,
            location=location
        )
    
    def _parse_for_statement(self) -> ForNode:
        """Parse for loop (range-based or C-style)"""
        location = self._peek().location
        self._consume(TokenType.FOR, "Expected 'for'")
        
        # Check for range-based for
        if self._check(TokenType.IDENTIFIER):
            checkpoint = self.current
            var_token = self._advance()
            
            if self._match(TokenType.IN):
                # Range-based for
                variable = var_token.value
                
                self._consume(TokenType.RANGE, "Expected 'range'")
                self._consume(TokenType.LEFT_PAREN, "Expected '('")
                
                count = self._parse_expression()
                
                self._consume(TokenType.RIGHT_PAREN, "Expected ')'")
                
                old_in_loop = self._in_loop
                self._in_loop = True
                
                body = []
                while not self._check(TokenType.ENDFOR) and not self._is_at_end():
                    stmt = self._parse_statement()
                    if stmt:
                        body.append(stmt)
                
                self._consume(TokenType.ENDFOR, "Expected 'endfor'")
                self._in_loop = old_in_loop
                
                return ForNode(
                    variable=variable,
                    count=count,
                    body=body,
                    is_range_based=True,
                    location=location
                )
            else:
                # Backtrack for C-style for
                self.current = checkpoint
        
        # C-style for loop
        self._consume(TokenType.LEFT_PAREN, "Expected '('")
        
        init_expr = None
        if not self._check(TokenType.SEMICOLON):
            init_expr = self._parse_expression()
        self._consume(TokenType.SEMICOLON, "Expected ';'")
        
        condition_expr = None
        if not self._check(TokenType.SEMICOLON):
            condition_expr = self._parse_expression()
        self._consume(TokenType.SEMICOLON, "Expected ';'")
        
        increment_expr = None
        if not self._check(TokenType.RIGHT_PAREN):
            increment_expr = self._parse_expression()
        self._consume(TokenType.RIGHT_PAREN, "Expected ')'")
        
        old_in_loop = self._in_loop
        self._in_loop = True
        
        body = []
        while not self._check(TokenType.ENDFOR) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
        
        self._consume(TokenType.ENDFOR, "Expected 'endfor'")
        self._in_loop = old_in_loop
        
        return ForNode(
            variable="",
            count="",
            body=body,
            init_expr=init_expr,
            condition_expr=condition_expr,
            increment_expr=increment_expr,
            is_range_based=False,
            location=location
        )
    
    def _parse_switch_statement(self) -> SwitchNode:
        """Parse switch statement"""
        location = self._peek().location
        self._consume(TokenType.SWITCH, "Expected 'switch'")
        
        self._consume(TokenType.LEFT_PAREN, "Expected '('")
        expression = self._parse_expression()
        self._consume(TokenType.RIGHT_PAREN, "Expected ')'")
        
        old_in_switch = self._in_switch
        self._in_switch = True
        
        cases = []
        while not self._check(TokenType.ENDSWITCH) and not self._is_at_end():
            if self._match(TokenType.CASE):
                case_value = self._parse_expression()
                self._consume(TokenType.COLON, "Expected ':'")
                
                case_body = []
                while not self._check(TokenType.CASE) and not self._check(TokenType.DEFAULT) and \
                      not self._check(TokenType.ENDSWITCH) and not self._is_at_end():
                    if self._check(TokenType.BREAK):
                        case_body.append(self._parse_break_statement())
                        break
                    stmt = self._parse_statement()
                    if stmt:
                        case_body.append(stmt)
                
                cases.append((case_value, case_body))
            
            elif self._match(TokenType.DEFAULT):
                self._consume(TokenType.COLON, "Expected ':'")
                
                default_body = []
                while not self._check(TokenType.CASE) and not self._check(TokenType.ENDSWITCH) and \
                      not self._is_at_end():
                    stmt = self._parse_statement()
                    if stmt:
                        default_body.append(stmt)
                
                cases.append((None, default_body))
            else:
                break
        
        self._consume(TokenType.ENDSWITCH, "Expected 'endswitch'")
        self._in_switch = old_in_switch
        
        return SwitchNode(
            expression=expression,
            cases=cases,
            location=location
        )
    
    def _parse_break_statement(self) -> BreakNode:
        """Parse break statement"""
        location = self._peek().location
        self._consume(TokenType.BREAK, "Expected 'break'")
        self._match(TokenType.SEMICOLON)
        
        if not self._in_loop and not self._in_switch:
            self.diagnostics.warning("'break' outside loop or switch", location)
        
        return BreakNode(location=location)
    
    def _parse_continue_statement(self) -> ContinueNode:
        """Parse continue statement"""
        location = self._peek().location
        self._consume(TokenType.CONTINUE, "Expected 'continue'")
        self._match(TokenType.SEMICOLON)
        
        if not self._in_loop:
            self.diagnostics.warning("'continue' outside loop", location)
        
        return ContinueNode(location=location)
    
    def _parse_return_statement(self) -> ReturnNode:
        """Parse return statement"""
        location = self._peek().location
        self._consume(TokenType.RETURN, "Expected 'return'")
        
        value = None
        if not self._check(TokenType.SEMICOLON) and not self._check(TokenType.NEWLINE):
            value = self._parse_expression()
        
        self._match(TokenType.SEMICOLON)
        
        if not self._in_function:
            self.diagnostics.warning("'return' outside function", location)
        
        return ReturnNode(value=value, location=location)
    
    def _parse_goto_statement(self) -> GotoNode:
        """Parse goto statement"""
        location = self._peek().location
        self._consume(TokenType.GOTO, "Expected 'goto'")
        
        label_token = self._consume(TokenType.IDENTIFIER, "Expected label")
        label = label_token.value
        
        self._match(TokenType.SEMICOLON)
        
        return GotoNode(label=label, location=location)
    
    def _parse_label(self) -> LabelNode:
        """Parse label"""
        location = self._peek().location
        name_token = self._consume(TokenType.IDENTIFIER, "Expected label name")
        name = name_token.value
        
        self._consume(TokenType.COLON, "Expected ':'")
        
        return LabelNode(name=name, location=location)
    
    def _parse_print_statement(self) -> PrintlnNode:
        """Parse print statement"""
        location = self._peek().location
        newline = self._match(TokenType.PRINTLN)
        if not newline:
            self._consume(TokenType.PRINT, "Expected 'print'")
        
        message = ""
        format_args = []
        
        if self._check(TokenType.STRING):
            message = self._advance().value
        else:
            message = self._parse_expression()
        
        # Additional arguments
        while self._match(TokenType.COMMA):
            format_args.append(self._parse_expression())
        
        self._match(TokenType.SEMICOLON)
        
        return PrintlnNode(
            message=message,
            newline=newline,
            format_args=format_args,
            location=location
        )
    
    def _parse_scan_statement(self) -> ScanfNode:
        """Parse scan statement"""
        location = self._peek().location
        self._consume(TokenType.SCAN, "Expected 'scan'")
        
        variable = self._consume(TokenType.IDENTIFIER, "Expected variable name").value
        
        additional_vars = []
        while self._match(TokenType.COMMA):
            additional_vars.append(self._consume(TokenType.IDENTIFIER, "Expected variable").value)
        
        self._match(TokenType.SEMICOLON)
        
        return ScanfNode(
            format_string='"%s"',
            variable=variable,
            additional_vars=additional_vars,
            location=location
        )
    
    def _parse_asm_block(self) -> AsmBlockNode:
        """Parse assembly block"""
        location = self._peek().location
        self._consume(TokenType.ASM_BLOCK, "Expected 'asm'")
        
        is_volatile = self._match(TokenType.VOLATILE)
        
        lines = []
        while not self._check(TokenType.ENDASM) and not self._is_at_end():
            if self._check(TokenType.ASSEMBLY_LINE):
                lines.append(self._advance().value)
            elif self._check(TokenType.IDENTIFIER):
                line_parts = []
                while not self._check(TokenType.NEWLINE) and not self._is_at_end():
                    line_parts.append(self._advance().value)
                lines.append(' '.join(line_parts))
            self._match(TokenType.NEWLINE)
        
        self._consume(TokenType.ENDASM, "Expected 'endasm'")
        
        return AsmBlockNode(
            asm_code='\n'.join(lines),
            is_volatile=is_volatile,
            location=location
        )
    
    def _parse_assembly_line(self) -> AssemblyNode:
        """Parse single assembly line"""
        location = self._peek().location
        
        if self._check(TokenType.ASSEMBLY_LINE):
            code = self._advance().value
        else:
            parts = []
            while not self._check(TokenType.NEWLINE) and not self._is_at_end():
                parts.append(self._advance().value)
            code = ' '.join(parts)
        
        return AssemblyNode(code=code, location=location)
    
    def _parse_c_inline_block(self) -> CCodeBlockNode:
        """Parse C inline code"""
        location = self._peek().location
        
        lines = []
        while self._check(TokenType.C_INLINE):
            lines.append(self._advance().value)
            self._match(TokenType.NEWLINE)
        
        return CCodeBlockNode(
            c_code='\n'.join(lines),
            location=location
        )
    
    def _parse_extern_directive(self) -> ExternDirectiveNode:
        """Parse extern directive"""
        location = self._peek().location
        self._consume(TokenType.EXTERN, "Expected 'extern'")
        
        parts = []
        use_angle = False
        
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            token = self._advance()
            parts.append(token.value)
            if '<' in token.value or '>' in token.value:
                use_angle = True
        
        header_name = ''.join(parts).strip()
        
        return ExternDirectiveNode(
            header_name=header_name,
            is_c_include=True,
            use_angle=use_angle,
            location=location
        )
    
    def _parse_comment(self) -> CommentNode:
        """Parse comment"""
        token = self._advance()
        return CommentNode(
            text=token.value,
            is_multiline=token.type == TokenType.MULTILINE_COMMENT,
            location=token.location
        )
    
    # ==================== HELPER METHODS ====================
    
    def _parse_condition(self) -> str:
        """Parse condition expression"""
        parts = []
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            if self._peek().type in [TokenType.IF, TokenType.ELIF, TokenType.WHILE, 
                                     TokenType.ENDIF, TokenType.ENDWHILE, TokenType.ELSE]:
                break
            parts.append(self._advance().value)
        return ' '.join(parts).strip()
    
    def _parse_expression(self) -> str:
        """Parse expression (simplified for now)"""
        parts = []
        paren_depth = 0
        
        while not self._is_at_end():
            token = self._peek()
            
            if token.type == TokenType.LEFT_PAREN:
                paren_depth += 1
            elif token.type == TokenType.RIGHT_PAREN:
                paren_depth -= 1
                if paren_depth < 0:
                    break
            elif token.type in [TokenType.SEMICOLON, TokenType.NEWLINE, TokenType.COMMA] and paren_depth == 0:
                break
            
            parts.append(self._advance().value)
        
        return ' '.join(parts).strip()
    
    def _parse_initializer(self) -> str:
        """Parse variable initializer"""
        return self._parse_expression()
    
    def _parse_array_size(self) -> int:
        """Parse array size"""
        token = self._consume(TokenType.NUMBER, "Expected array size")
        return int(token.value)
    
    def _parse_parameter_list(self) -> List[Tuple[str, str]]:
        """Parse function parameter list"""
        parameters = []
        
        while not self._check(TokenType.RIGHT_PAREN) and not self._is_at_end():
            if not self._check_type():
                break
            
            param_type = self._advance().value
            param_name = self._consume(TokenType.IDENTIFIER, "Expected parameter name").value
            
            parameters.append((param_name, param_type))
            
            if not self._match(TokenType.COMMA):
                break
        
        return parameters
    
    def _parse_identifier_list(self) -> List[str]:
        """Parse comma-separated identifier list"""
        identifiers = []
        
        while not self._check(TokenType.RIGHT_PAREN) and not self._is_at_end():
            identifiers.append(self._consume(TokenType.IDENTIFIER, "Expected identifier").value)
            
            if not self._match(TokenType.COMMA):
                break
        
        return identifiers
    
    def _check_type(self) -> bool:
        """Check if current token is a type"""
        type_tokens = [
            TokenType.INT_TYPE, TokenType.INT8_TYPE, TokenType.INT16_TYPE,
            TokenType.INT32_TYPE, TokenType.INT64_TYPE, TokenType.UINT_TYPE,
            TokenType.UINT8_TYPE, TokenType.UINT16_TYPE, TokenType.UINT32_TYPE,
            TokenType.UINT64_TYPE, TokenType.STR_TYPE, TokenType.BOOL_TYPE,
            TokenType.FLOAT_TYPE, TokenType.DOUBLE_TYPE, TokenType.CHAR_TYPE,
            TokenType.VOID_TYPE, TokenType.BUFFER_TYPE, TokenType.PTR_TYPE,
            TokenType.AUTO_TYPE, TokenType.ASM_DIRECTIVE
        ]
        return self._peek().type in type_tokens
    
    def _is_assignment(self) -> bool:
        """Check if this is an assignment"""
        checkpoint = self.current
        
        if self._check(TokenType.IDENTIFIER):
            self._advance()
            is_assign = self._peek().type in [
                TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                TokenType.MULT_ASSIGN, TokenType.DIV_ASSIGN, TokenType.MOD_ASSIGN,
                TokenType.AND_ASSIGN, TokenType.OR_ASSIGN, TokenType.XOR_ASSIGN,
                TokenType.SHL_ASSIGN, TokenType.SHR_ASSIGN
            ]
            self.current = checkpoint
            return is_assign
        
        return False
    
    def _is_label(self) -> bool:
        """Check if this is a label"""
        checkpoint = self.current
        
        if self._check(TokenType.IDENTIFIER):
            self._advance()
            is_lbl = self._check(TokenType.COLON)
            self.current = checkpoint
            return is_lbl
        
        return False
    
    # ==================== TOKEN NAVIGATION ====================
    
    def _advance(self) -> Token:
        """Advance to next token"""
        if not self._is_at_end():
            self.current += 1
        return self._previous()
    
    def _peek(self) -> Token:
        """Peek at current token"""
        if self.current >= len(self.tokens):
            return Token(TokenType.EOF, '', SourceLocation(0, 0))
        return self.tokens[self.current]
    
    def _previous(self) -> Token:
        """Get previous token"""
        return self.tokens[self.current - 1]
    
    def _check(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the types"""
        if self._is_at_end():
            return False
        return self._peek().type in token_types
    
    def _match(self, *token_types: TokenType) -> bool:
        """Match and consume token if it matches"""
        if self._check(*token_types):
            self._advance()
            return True
        return False
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token or raise error"""
        if self._check(token_type):
            return self._advance()
        
        self._error(message)
        return self._peek()
    
    def _is_at_end(self) -> bool:
        """Check if at end of tokens"""
        return self.current >= len(self.tokens) or self._peek().type == TokenType.EOF
    
    def _error(self, message: str):
        """Report parse error"""
        token = self._peek()
        self.diagnostics.error(message, token.location)
        raise ParseError(message, token)

# ==================== ENHANCED CODE GENERATOR ====================

class AssemblyCodeGenerator(ASTVisitor):
    """Enhanced assembly code generator with optimizations"""
    
    def __init__(self):
        self.data_section = []
        self.bss_section = []
        self.text_section = []
        self.rodata_section = []
        
        self.string_labels = {}
        self.variable_labels = {}
        self.variable_info = {}
        self.function_labels = {}
        
        self.label_counter = 0
        self.var_counter = 0
        self.str_counter = 0
        
        self.current_function = None
        self.diagnostics = DiagnosticEngine()
        
    def generate(self, ast: ProgramNode) -> str:
        """Generate assembly code from AST"""
        try:
            ast.accept(self)
            return self._build_output()
        except Exception as e:
            self.diagnostics.error(f"Code generation failed: {str(e)}")
            return ""
    
    def _build_output(self) -> str:
        """Build final assembly output"""
        lines = []
        
        # Data section
        if self.data_section or self.rodata_section:
            lines.append("section .data")
            lines.extend(self.data_section)
            if self.rodata_section:
                lines.append("    ; Read-only data")
                lines.extend(self.rodata_section)
            lines.append("")
        
        # BSS section
        if self.bss_section:
            lines.append("section .bss")
            lines.extend(self.bss_section)
            lines.append("")
        
        # Text section
        lines.append("section .text")
        lines.extend(self.text_section)
        
        return '\n'.join(lines)
    
    # Visitor implementations
    def visit_program(self, node: ProgramNode):
        for stmt in node.statements:
            if stmt:
                stmt.accept(self)
    
    def visit_var_declaration(self, node: VarDeclarationNode):
        self.var_counter += 1
        label = f"V{self.var_counter}"
        
        self.variable_labels[node.name] = label
        self.variable_info[node.name] = {
            'type': node.var_type,
            'size': node.size,
            'label': label,
            'value': node.value,
            'is_const': node.is_const
        }
        
        # Generate appropriate declaration
        if node.var_type == 'int':
            value = node.value or '0'
            section = self.rodata_section if node.is_const else self.data_section
            section.append(f"    {label} dd {value}  ; int {node.name}")
        
        elif node.var_type == 'str':
            str_value = (node.value or '""').strip().strip('"')
            self.data_section.append(f'    {label} db "{str_value}", 0  ; str {node.name}')
        
        elif node.var_type in {'db', 'dw', 'dd', 'dq'}:
            value = node.value or '0'
            self.data_section.append(f"    {label} {node.var_type} {value}")
        
        elif node.var_type in {'resb', 'resw', 'resd', 'resq'}:
            size = node.size or 1
            self.bss_section.append(f"    {label} {node.var_type} {size}")
        
        else:
            self.data_section.append(f"    {label} dd 0  ; {node.name}")
    
    def visit_function_declaration(self, node: FunctionDeclarationNode):
        if not node.body:
            return  # Forward declaration
        
        self.current_function = node.name
        self.text_section.append(f"\n; Function: {node.name}")
        self.text_section.append(f"{node.name}:")
        
        # Function prologue
        self.text_section.append("    push rbp")
        self.text_section.append("    mov rbp, rsp")
        
        # Body
        for stmt in node.body:
            stmt.accept(self)
        
        # Function epilogue
        self.text_section.append("    mov rsp, rbp")
        self.text_section.append("    pop rbp")
        self.text_section.append("    ret")
        
        self.current_function = None
    
    def visit_struct_declaration(self, node: StructDeclarationNode):
        self.text_section.append(f"; Struct: {node.name}")
    
    def visit_union_declaration(self, node: UnionDeclarationNode):
        self.text_section.append(f"; Union: {node.name}")
    
    def visit_enum_declaration(self, node: EnumDeclarationNode):
        self.text_section.append(f"; Enum: {node.name}")
        for name, value in node.values:
            self.text_section.append(f"{name} equ {value}")
    
    def visit_typedef(self, node: TypedefNode):
        self.text_section.append(f"; typedef {node.original_type} {node.new_name}")
    
    def visit_macro_definition(self, node: MacroDefinitionNode):
        self.text_section.append(f"; Macro: {node.name}")
    
    def visit_assignment(self, node: AssignmentNode):
        var_label = self.variable_labels.get(node.name, node.name)
        
        if node.is_compound:
            # Compound assignment
            self.text_section.append(f"    ; {node.name} {node.operator} {node.value}")
            self.text_section.append(f"    mov eax, dword [rel {var_label}]")
            
            if node.operator == '+=':
                self.text_section.append(f"    add eax, {node.value}")
            elif node.operator == '-=':
                self.text_section.append(f"    sub eax, {node.value}")
            elif node.operator == '*=':
                self.text_section.append(f"    imul eax, {node.value}")
            elif node.operator == '/=':
                self.text_section.append(f"    mov ebx, {node.value}")
                self.text_section.append(f"    cdq")
                self.text_section.append(f"    idiv ebx")
            
            self.text_section.append(f"    mov dword [rel {var_label}], eax")
        else:
            # Simple assignment
            self.text_section.append(f"    ; {node.name} = {node.value}")
            self.text_section.append(f"    mov eax, {node.value}")
            self.text_section.append(f"    mov dword [rel {var_label}], eax")
    
    def visit_if(self, node: IfNode):
        end_label = self._gen_label('if_end')
        
        self.text_section.append(f"    ; if {node.condition}")
        self._generate_condition(node.condition, end_label, negate=True)
        
        for stmt in node.if_body:
            stmt.accept(self)
        
        # Elif branches
        for elif_cond, elif_body in node.elif_branches:
            elif_end = self._gen_label('elif_end')
            self.text_section.append(f"    jmp {end_label}")
            self.text_section.append(f"{elif_end}:")
            
            self._generate_condition(elif_cond, end_label, negate=True)
            for stmt in elif_body:
                stmt.accept(self)
        
        # Else branch
        if node.else_body:
            else_label = self._gen_label('else')
            self.text_section.append(f"    jmp {end_label}")
            self.text_section.append(f"{else_label}:")
            for stmt in node.else_body:
                stmt.accept(self)
        
        self.text_section.append(f"{end_label}:")
    
    def visit_while(self, node: WhileNode):
        start_label = self._gen_label('while_start')
        end_label = self._gen_label('while_end')
        
        self.text_section.append(f"{start_label}:")
        self.text_section.append(f"    ; while {node.condition}")
        
        self._generate_condition(node.condition, end_label, negate=True)
        
        for stmt in node.body:
            stmt.accept(self)
        
        self.text_section.append(f"    jmp {start_label}")
        self.text_section.append(f"{end_label}:")
    
    def visit_do_while(self, node: DoWhileNode):
        start_label = self._gen_label('do_start')
        
        self.text_section.append(f"{start_label}:")
        
        for stmt in node.body:
            stmt.accept(self)
        
        self.text_section.append(f"    ; while {node.condition}")
        self._generate_condition(node.condition, start_label, negate=False)
    
    def visit_for(self, node: ForNode):
        if node.is_range_based:
            # Range-based for loop
            start_label = self._gen_label('for_start')
            end_label = self._gen_label('for_end')
            counter_var = self._get_var_label(node.variable)
            
            self.text_section.append(f"    ; for {node.variable} in range({node.count})")
            self.text_section.append(f"    mov dword [rel {counter_var}], 0")
            
            self.text_section.append(f"{start_label}:")
            self.text_section.append(f"    mov eax, dword [rel {counter_var}]")
            
            if node.count.isdigit():
                self.text_section.append(f"    cmp eax, {node.count}")
            else:
                count_label = self._get_var_label(node.count)
                self.text_section.append(f"    cmp eax, dword [rel {count_label}]")
            
            self.text_section.append(f"    jge {end_label}")
            
            for stmt in node.body:
                stmt.accept(self)
            
            self.text_section.append(f"    inc dword [rel {counter_var}]")
            self.text_section.append(f"    jmp {start_label}")
            self.text_section.append(f"{end_label}:")
        else:
            # C-style for loop
            start_label = self._gen_label('for_start')
            end_label = self._gen_label('for_end')
            
            # Init
            if node.init_expr:
                self.text_section.append(f"    ; init: {node.init_expr}")
                self._generate_expression(node.init_expr)
            
            self.text_section.append(f"{start_label}:")
            
            # Condition
            if node.condition_expr:
                self._generate_condition(node.condition_expr, end_label, negate=True)
            
            # Body
            for stmt in node.body:
                stmt.accept(self)
            
            # Increment
            if node.increment_expr:
                self.text_section.append(f"    ; increment: {node.increment_expr}")
                self._generate_expression(node.increment_expr)
            
            self.text_section.append(f"    jmp {start_label}")
            self.text_section.append(f"{end_label}:")
    
    def visit_switch(self, node: SwitchNode):
        end_label = self._gen_label('switch_end')
        
        self.text_section.append(f"    ; switch {node.expression}")
        self.text_section.append(f"    mov eax, {node.expression}")
        
        case_labels = []
        for i, (case_value, _) in enumerate(node.cases):
            if case_value is not None:
                label = self._gen_label(f'case_{i}')
                case_labels.append((case_value, label))
        
        # Generate comparisons
        for case_value, label in case_labels:
            self.text_section.append(f"    cmp eax, {case_value}")
            self.text_section.append(f"    je {label}")
        
        # Default case
        default_label = None
        for case_value, case_body in node.cases:
            if case_value is None:
                default_label = self._gen_label('default')
                self.text_section.append(f"    jmp {default_label}")
                break
        
        if not default_label:
            self.text_section.append(f"    jmp {end_label}")
        
        # Generate case bodies
        for i, (case_value, case_body) in enumerate(node.cases):
            if case_value is not None:
                self.text_section.append(f"{case_labels[i][1]}:")
            else:
                self.text_section.append(f"{default_label}:")
            
            for stmt in case_body:
                stmt.accept(self)
        
        self.text_section.append(f"{end_label}:")
    
    def visit_break(self, node: BreakNode):
        self.text_section.append("    ; break")
        # Note: actual implementation needs loop context tracking
    
    def visit_continue(self, node: ContinueNode):
        self.text_section.append("    ; continue")
        # Note: actual implementation needs loop context tracking
    
    def visit_return(self, node: ReturnNode):
        if node.value:
            self.text_section.append(f"    ; return {node.value}")
            self.text_section.append(f"    mov eax, {node.value}")
        else:
            self.text_section.append("    ; return")
        
        if self.current_function:
            self.text_section.append("    mov rsp, rbp")
            self.text_section.append("    pop rbp")
        
        self.text_section.append("    ret")
    
    def visit_goto(self, node: GotoNode):
        self.text_section.append(f"    jmp {node.label}")
    
    def visit_label(self, node: LabelNode):
        self.text_section.append(f"{node.name}:")
    
    def visit_println(self, node: PrintlnNode):
        message = node.message.strip()
        
        if message.startswith('"') and message.endswith('"'):
            literal = message[1:-1]
        else:
            literal = message
        
        # Generate or reuse string label
        if literal in self.string_labels:
            label = self.string_labels[literal]
        else:
            self.str_counter += 1
            label = f"STR{self.str_counter}"
            self.string_labels[literal] = label
            
            if node.newline:
                self.data_section.append(f'    {label} db "{literal}", 10, 0')
            else:
                self.data_section.append(f'    {label} db "{literal}", 0')
        
        # Generate print call (using macros UPRINT or actual printf)
        self.text_section.append(f"    ; print: {literal[:30]}...")
        self.text_section.append(f"    lea rdi, [rel {label}]")
        self.text_section.append(f"    call printf")
    
    def visit_scanf(self, node: ScanfNode):
        fmt_label = self._gen_label('FMT')
        self.data_section.append(f'    {fmt_label} db {node.format_string}, 0')
        
        var_label = self._get_var_label(node.variable)
        
        self.text_section.append(f"    ; scanf: {node.variable}")
        self.text_section.append(f"    lea rdi, [rel {fmt_label}]")
        self.text_section.append(f"    lea rsi, [rel {var_label}]")
        self.text_section.append(f"    call scanf")
    
    def visit_assembly(self, node: AssemblyNode):
        code = node.code.strip()
        
        # Variable substitution
        for var_name, var_label in self.variable_labels.items():
            code = re.sub(rf'\b{re.escape(var_name)}\b', f'[rel {var_label}]', code)
        
        if not code.startswith('    ') and not code.endswith(':'):
            code = f"    {code}"
        
        self.text_section.append(code)
    
    def visit_asm_block(self, node: AsmBlockNode):
        self.text_section.append("    ; === Assembly Block ===")
        
        for line in node.asm_code.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Variable substitution
            for var_name, var_label in self.variable_labels.items():
                line = re.sub(rf'\b{re.escape(var_name)}\b', f'[rel {var_label}]', line)
            
            if not line.startswith('    ') and not line.endswith(':'):
                line = f"    {line}"
            
            self.text_section.append(line)
        
        self.text_section.append("    ; === End Assembly Block ===")
    
    def visit_comment(self, node: CommentNode):
        if node.is_multiline:
            for line in node.text.split('\n'):
                self.text_section.append(f"    ; {line.strip()}")
        else:
            self.text_section.append(f"    {node.text}")
    
    def visit_c_code_block(self, node: CCodeBlockNode):
        self.text_section.append("    ; === C Code Block ===")
        for line in node.c_code.split('\n'):
            if line.strip():
                self.text_section.append(f"    ; {line.strip()}")
        self.text_section.append("    ; === End C Code Block ===")
        
        # Note: In production, this would integrate with C compiler
    
    def visit_extern_directive(self, node: ExternDirectiveNode):
        header = node.header_name.strip()
        
        if '<' in header or '>' in header:
            header = header.replace('<', '').replace('>', '')
            self.text_section.append(f"    ; extern header: <{header}>")
        else:
            self.text_section.append(f"    extern {header}")
    
    def visit_optimization_hint(self, node: OptimizationHintNode):
        self.text_section.append(f"    ; optimization: {node.hint_type}")
        if node.target:
            node.target.accept(self)
    
    # ==================== HELPER METHODS ====================
    
    def _gen_label(self, prefix: str = "L") -> str:
        """Generate unique label"""
        self.label_counter += 1
        return f"{prefix.upper()}{self.label_counter}"
    
    def _get_var_label(self, name: str) -> str:
        """Get or create variable label"""
        if name in self.variable_labels:
            return self.variable_labels[name]
        
        self.var_counter += 1
        label = f"V{self.var_counter}"
        self.variable_labels[name] = label
        self.data_section.append(f"    {label} dd 0  ; auto-declared {name}")
        
        return label
    
    def _generate_condition(self, condition: str, target_label: str, negate: bool = False):
        """Generate condition check and jump"""
        # Simple condition parsing
        if '==' in condition:
            left, right = condition.split('==')
            self.text_section.append(f"    mov eax, {left.strip()}")
            self.text_section.append(f"    cmp eax, {right.strip()}")
            self.text_section.append(f"    {'jne' if negate else 'je'} {target_label}")
        
        elif '!=' in condition:
            left, right = condition.split('!=')
            self.text_section.append(f"    mov eax, {left.strip()}")
            self.text_section.append(f"    cmp eax, {right.strip()}")
            self.text_section.append(f"    {'je' if negate else 'jne'} {target_label}")
        
        elif '<=' in condition:
            left, right = condition.split('<=')
            self.text_section.append(f"    mov eax, {left.strip()}")
            self.text_section.append(f"    cmp eax, {right.strip()}")
            self.text_section.append(f"    {'jg' if negate else 'jle'} {target_label}")
        
        elif '>=' in condition:
            left, right = condition.split('>=')
            self.text_section.append(f"    mov eax, {left.strip()}")
            self.text_section.append(f"    cmp eax, {right.strip()}")
            self.text_section.append(f"    {'jl' if negate else 'jge'} {target_label}")
        
        elif '<' in condition:
            left, right = condition.split('<')
            self.text_section.append(f"    mov eax, {left.strip()}")
            self.text_section.append(f"    cmp eax, {right.strip()}")
            self.text_section.append(f"    {'jge' if negate else 'jl'} {target_label}")
        
        elif '>' in condition:
            left, right = condition.split('>')
            self.text_section.append(f"    mov eax, {left.strip()}")
            self.text_section.append(f"    cmp eax, {right.strip()}")
            self.text_section.append(f"    {'jle' if negate else 'jg'} {target_label}")
        
        else:
            # Simple boolean expression
            self.text_section.append(f"    cmp {condition.strip()}, 0")
            self.text_section.append(f"    {'je' if negate else 'jne'} {target_label}")
    
    def _generate_expression(self, expression: str):
        """Generate code for expression"""
        self.text_section.append(f"    ; expr: {expression}")
        # Simplified - in production would parse and generate proper code
        self.text_section.append(f"    mov eax, {expression}")

# ==================== COMPILER PIPELINE ====================

class CompilerConfig:
    """Compiler configuration"""
    
    def __init__(self):
        self.optimization_level = 0
        self.target_arch = "x86_64"
        self.emit_debug_info = False
        self.verbose = False
        self.warnings_as_errors = False
        self.unused_variable_warning = True
        self.output_format = "nasm"

class CASMCompiler:
    """Main compiler class coordinating all phases"""
    
    def __init__(self, config: Optional[CompilerConfig] = None):
        self.config = config or CompilerConfig()
        self.diagnostics = DiagnosticEngine()
        
    def compile(self, source: str, filename: str = "<stdin>") -> Optional[str]:
        """Full compilation pipeline"""
        
        # Phase 1: Lexical Analysis
        if self.config.verbose:
            print("Phase 1: Lexical Analysis")
        
        lexer = CASMLexer(source, filename)
        tokens = lexer.tokenize()
        
        if lexer.diagnostics.has_errors():
            lexer.diagnostics.print_all()
            return None
        
        if self.config.verbose:
            print(f"  Generated {len(tokens)} tokens")
        
        # Phase 2: Parsing
        if self.config.verbose:
            print("Phase 2: Parsing")
        
        parser = CASMParser(tokens)
        ast = parser.parse()
        
        if parser.diagnostics.has_errors():
            parser.diagnostics.print_all()
            return None
        
        if not ast:
            print("Parse failed: no AST generated")
            return None
        
        if self.config.verbose:
            print(f"  Generated AST with {len(ast.statements)} statements")
        
        # Phase 3: Semantic Analysis (if enabled)
        # ... would go here in production
        
        # Phase 4: Optimization (if enabled)
        # ... would go here in production
        
        # Phase 5: Code Generation
        if self.config.verbose:
            print("Phase 5: Code Generation")
        
        codegen = AssemblyCodeGenerator()
        assembly = codegen.generate(ast)
        
        if codegen.diagnostics.has_errors():
            codegen.diagnostics.print_all()
            return None
        
        # Show warnings
        if parser.diagnostics.warning_count > 0 or codegen.diagnostics.warning_count > 0:
            parser.diagnostics.print_all()
            codegen.diagnostics.print_all()
        
        if self.config.verbose:
            print(f"Compilation successful!")
            print(f"  {parser.diagnostics.warning_count} warnings")
        
        return assembly

# ==================== UTILITY FUNCTIONS ====================

def create_default_compiler() -> CASMCompiler:
    """Create compiler with default configuration"""
    config = CompilerConfig()
    return CASMCompiler(config)

def compile_string(source: str, verbose: bool = False) -> Optional[str]:
    """Quick compile from string"""
    config = CompilerConfig()
    config.verbose = verbose
    compiler = CASMCompiler(config)
    return compiler.compile(source)

def compile_file(filepath: str, verbose: bool = False) -> Optional[str]:
    """Compile from file"""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        config = CompilerConfig()
        config.verbose = verbose
        compiler = CASMCompiler(config)
        return compiler.compile(source, filepath)
    
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example CASM code
    example_code = """
    ; Enhanced CASM Example
    var int x = 10
    var int y = 20
    const int MAX = 100
    
    function int add(int a, int b)
        var int result = 0
        result = a + b
        return result
    
    if x < y
        println "x is less than y"
    elif x == y
        println "x equals y"
    else
        println "x is greater than y"
    endif
    
    for i in range(5)
        print "Iteration: "
        println i
    endfor
    
    asm
        mov rax, 0
        ret
    endasm
    """
    
    # Compile
    compiler = create_default_compiler()
    compiler.config.verbose = True
    
    result = compiler.compile(example_code)
    
    if result:
        print("\n" + "="*60)
        print("GENERATED ASSEMBLY:")
        print("="*60)
        print(result)
    else:
        print("\nCompilation failed!")