#!/usr/bin/env python3
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class TokenType(Enum):
    """Essential token types for CASM - focused on core functionality"""
    
    # Control flow keywords
    IF = "IF"
    ELSE = "ELSE"
    ENDIF = "ENDIF"
    WHILE = "WHILE"
    ENDWHILE = "ENDWHILE"
    FOR = "FOR"
    ENDFOR = "ENDFOR"
    
    # Variable declarations
    VAR = "VAR"
    EXTERN = "EXTERN"
    
    # Data types
    INT_TYPE = "INT_TYPE"
    STR_TYPE = "STR_TYPE"
    BOOL_TYPE = "BOOL_TYPE"
    FLOAT_TYPE = "FLOAT_TYPE"
    BUFFER_TYPE = "BUFFER_TYPE"
    
    # Essential I/O
    PRINT = "PRINT"           # print statement
    SCAN = "SCAN"             # scan statement
    
    # C Code Integration
    C_INLINE = "C_INLINE"          # Inline C expressions
    
    # Operators
    ASSIGN = "ASSIGN"          # =
    EQUALS = "EQUALS"          # ==
    NOT_EQUALS = "NOT_EQUALS"  # !=
    LESS_THAN = "LESS_THAN"    # <
    GREATER_THAN = "GREATER_THAN"  # >
    LESS_EQUAL = "LESS_EQUAL"      # <=
    GREATER_EQUAL = "GREATER_EQUAL"  # >=
    PLUS = "PLUS"              # +
    MINUS = "MINUS"            # -
    MULTIPLY = "MULTIPLY"      # *
    DIVIDE = "DIVIDE"          # /
    MODULO = "MODULO"          # %
    
    # Punctuation
    LEFT_PAREN = "LEFT_PAREN"      # (
    RIGHT_PAREN = "RIGHT_PAREN"    # )
    LEFT_BRACKET = "LEFT_BRACKET"  # [
    RIGHT_BRACKET = "RIGHT_BRACKET"  # ]
    LEFT_BRACE = "LEFT_BRACE"      # {
    RIGHT_BRACE = "RIGHT_BRACE"    # }
    COMMA = "COMMA"                # ,
    SEMICOLON = "SEMICOLON"        # ;
    COLON = "COLON"                # :
    
    # Literals
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    
    # Special
    NEWLINE = "NEWLINE"
    COMMENT = "COMMENT"
    ASSEMBLY_LINE = "ASSEMBLY_LINE"  # Raw assembly code
    
    # Control
    EOF = "EOF"
    
    # Range support for for-loops
    IN = "IN"
    RANGE = "RANGE"

@dataclass
class Token:
    """Token data structure with location information"""
    type: TokenType
    value: str
    line: int
    column: int
    raw_line: str = ""
    
    def __str__(self) -> str:
        return f"{self.type.value}:{self.value}"
    
    def __repr__(self) -> str:
        return f"Token({self.type.value}, '{self.value}', {self.line}:{self.column})"

class LexerError(Exception):
    """Exception for lexical analysis errors"""
    
    def __init__(self, message: str, line: int, column: int, context: str = ""):
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        super().__init__(f"Lexer error at line {line}, column {column}: {message}")