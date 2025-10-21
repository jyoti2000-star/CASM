from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class TokenType(Enum):
    # Control Flow
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    ENDIF = auto()
    WHILE = auto()
    ENDWHILE = auto()
    FOR = auto()
    ENDFOR = auto()
    DO = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    ENDSWITCH = auto()
    
    # Declarations
    VAR = auto()
    CONST = auto()
    EXTERN = auto()
    STATIC = auto()
    VOLATILE = auto()
    REGISTER = auto()
    INLINE = auto()
    FUNCTION = auto()
    PROCEDURE = auto()
    STRUCT = auto()
    UNION = auto()
    ENUM = auto()
    TYPEDEF = auto()
    
    # Types
    INT_TYPE = auto()
    INT8_TYPE = auto()
    INT16_TYPE = auto()
    INT32_TYPE = auto()
    INT64_TYPE = auto()
    UINT_TYPE = auto()
    UINT8_TYPE = auto()
    UINT16_TYPE = auto()
    UINT32_TYPE = auto()
    UINT64_TYPE = auto()
    STR_TYPE = auto()
    BOOL_TYPE = auto()
    FLOAT_TYPE = auto()
    DOUBLE_TYPE = auto()
    CHAR_TYPE = auto()
    VOID_TYPE = auto()
    BUFFER_TYPE = auto()
    PTR_TYPE = auto()
    
    # Assembly Directives
    ASM_DIRECTIVE = auto()
    ASM_BLOCK = auto()
    
    # I/O
    PRINT = auto()
    PRINTLN = auto()
    SCAN = auto()
    READ = auto()
    WRITE = auto()
    
    # Operators
    C_INLINE = auto()
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    MULT_ASSIGN = auto()
    DIV_ASSIGN = auto()
    MOD_ASSIGN = auto()
    AND_ASSIGN = auto()
    OR_ASSIGN = auto()
    XOR_ASSIGN = auto()
    SHL_ASSIGN = auto()
    SHR_ASSIGN = auto()
    
    # Comparison
    EQUALS = auto()
    NOT_EQUALS = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    
    # Arithmetic
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    INCREMENT = auto()
    DECREMENT = auto()
    POWER = auto()
    
    # Bitwise
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_NOT = auto()
    LEFT_SHIFT = auto()
    RIGHT_SHIFT = auto()
    
    # Logical
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    LOGICAL_NOT = auto()
    
    # Punctuation
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    SEMICOLON = auto()
    COLON = auto()
    DOT = auto()
    ARROW = auto()
    QUESTION = auto()
    HASH = auto()
    AT = auto()
    DOLLAR = auto()
    AMPERSAND = auto()
    PIPE = auto()
    
    # Literals
    IDENTIFIER = auto()
    NUMBER = auto()
    FLOAT_NUMBER = auto()
    HEX_NUMBER = auto()
    BIN_NUMBER = auto()
    OCT_NUMBER = auto()
    STRING = auto()
    CHAR_LITERAL = auto()
    TRUE = auto()
    FALSE = auto()
    NULL = auto()
    
    # Special
    NEWLINE = auto()
    COMMENT = auto()
    ASSEMBLY_LINE = auto()
    EOF = auto()
    IN = auto()
    RANGE = auto()
    SIZEOF = auto()
    TYPEOF = auto()
    ALIGNOF = auto()
    OFFSETOF = auto()
    
    # Optimization hints
    OPTIMIZE = auto()
    UNROLL = auto()
    VECTORIZE = auto()
    LIKELY = auto()
    UNLIKELY = auto()


@dataclass
class SourceLocation:
    line: int
    column: int
    file: str = "<stdin>"
    raw_line: str = ""
    
    def __str__(self):
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class Token:
    type: TokenType
    value: str
    location: SourceLocation
    
    def __str__(self):
        return f"{self.type.name}({self.value}) at {self.location}"
