#!/usr/bin/env python3
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple, Dict, Set, Union
from abc import ABC, abstractmethod
import re
import os
from collections import defaultdict

#  ENHANCED TOKEN SYSTEM 
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
    """Enhanced source location tracking"""
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

#  ENHANCED TYPE SYSTEM 
class TypeKind(Enum):
    PRIMITIVE = auto()
    POINTER = auto()
    ARRAY = auto()
    STRUCT = auto()
    UNION = auto()
    FUNCTION = auto()
    VOID = auto()

@dataclass
class Type:
    """Advanced type representation"""
    kind: TypeKind
    name: str
    size: int = 0
    alignment: int = 0
    is_const: bool = False
    is_volatile: bool = False
    is_signed: bool = True
    
    # For complex types
    base_type: Optional['Type'] = None  # For pointers/arrays
    members: Dict[str, 'Type'] = field(default_factory=dict)  # For structs/unions
    param_types: List['Type'] = field(default_factory=list)  # For functions
    return_type: Optional['Type'] = None  # For functions
    array_size: Optional[int] = None
    
    def __str__(self):
        qualifiers = []
        if self.is_const:
            qualifiers.append("const")
        if self.is_volatile:
            qualifiers.append("volatile")
        
        qual_str = " ".join(qualifiers) + " " if qualifiers else ""
        
        if self.kind == TypeKind.POINTER:
            return f"{qual_str}{self.base_type}*"
        elif self.kind == TypeKind.ARRAY:
            return f"{qual_str}{self.base_type}[{self.array_size}]"
        else:
            return f"{qual_str}{self.name}"
    
    def is_compatible_with(self, other: 'Type') -> bool:
        """Check type compatibility"""
        if self.kind != other.kind:
            return False
        if self.kind == TypeKind.PRIMITIVE:
            return self.name == other.name or self._is_numeric() and other._is_numeric()
        if self.kind == TypeKind.POINTER:
            return self.base_type.is_compatible_with(other.base_type)
        return self.name == other.name
    
    def _is_numeric(self) -> bool:
        return self.name in {'int', 'int8', 'int16', 'int32', 'int64', 
                            'uint', 'uint8', 'uint16', 'uint32', 'uint64',
                            'float', 'double', 'char'}

# Type factory for common types
class TypeFactory:
    _types_cache = {}
    
    @classmethod
    def get_primitive(cls, name: str, size: int, signed: bool = True) -> Type:
        key = (name, size, signed)
        if key not in cls._types_cache:
            cls._types_cache[key] = Type(
                kind=TypeKind.PRIMITIVE,
                name=name,
                size=size,
                alignment=size,
                is_signed=signed
            )
        return cls._types_cache[key]
    
    @classmethod
    def get_pointer(cls, base: Type) -> Type:
        return Type(
            kind=TypeKind.POINTER,
            name=f"{base.name}*",
            size=8,  # 64-bit pointer
            alignment=8,
            base_type=base
        )
    
    @classmethod
    def get_array(cls, base: Type, size: int) -> Type:
        return Type(
            kind=TypeKind.ARRAY,
            name=f"{base.name}[{size}]",
            size=base.size * size,
            alignment=base.alignment,
            base_type=base,
            array_size=size
        )

# Predefined types
INT_TYPE = TypeFactory.get_primitive("int", 4)
INT8_TYPE = TypeFactory.get_primitive("int8", 1)
INT16_TYPE = TypeFactory.get_primitive("int16", 2)
INT32_TYPE = TypeFactory.get_primitive("int32", 4)
INT64_TYPE = TypeFactory.get_primitive("int64", 8)
UINT_TYPE = TypeFactory.get_primitive("uint", 4, signed=False)
FLOAT_TYPE = TypeFactory.get_primitive("float", 4)
DOUBLE_TYPE = TypeFactory.get_primitive("double", 8)
CHAR_TYPE = TypeFactory.get_primitive("char", 1)
BOOL_TYPE = TypeFactory.get_primitive("bool", 1)
VOID_TYPE = Type(kind=TypeKind.VOID, name="void", size=0, alignment=1)

#  SYMBOL TABLE WITH SCOPING 
@dataclass
class Symbol:
    """Enhanced symbol representation"""
    name: str
    type: Type
    scope_level: int
    is_defined: bool = True
    is_extern: bool = False
    is_static: bool = False
    is_register: bool = False
    location: Optional[SourceLocation] = None
    asm_label: Optional[str] = None
    value: Optional[Any] = None
    is_function: bool = False
    is_parameter: bool = False
    offset: int = 0  # Stack offset or struct offset

class SymbolTable:
    """Advanced symbol table with lexical scoping"""
    
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]  # Stack of scopes
        self.current_scope_level = 0
        self.global_scope = self.scopes[0]
        
    def enter_scope(self):
        """Enter a new lexical scope"""
        self.current_scope_level += 1
        self.scopes.append({})
        
    def exit_scope(self):
        """Exit current scope"""
        if self.current_scope_level > 0:
            self.scopes.pop()
            self.current_scope_level -= 1
    
    def define(self, symbol: Symbol) -> bool:
        """Define a symbol in current scope"""
        symbol.scope_level = self.current_scope_level
        current = self.scopes[self.current_scope_level]
        
        if symbol.name in current:
            return False  # Already defined in this scope
        
        current[symbol.name] = symbol
        return True
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Lookup symbol in all visible scopes"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def lookup_current_scope(self, name: str) -> Optional[Symbol]:
        """Lookup symbol only in current scope"""
        return self.scopes[self.current_scope_level].get(name)
    
    def get_all_symbols(self) -> List[Symbol]:
        """Get all symbols from all scopes"""
        symbols = []
        for scope in self.scopes:
            symbols.extend(scope.values())
        return symbols

#  ENHANCED AST NODES 
class ASTNode(ABC):
    def __init__(self, location: Optional[SourceLocation] = None):
        self.location = location
        self.type: Optional[Type] = None  # Type annotation after semantic analysis
    
    @abstractmethod
    def accept(self, visitor):
        pass

@dataclass
class ProgramNode(ASTNode):
    statements: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_program(self)

@dataclass
class FunctionNode(ASTNode):
    name: str
    return_type: Type
    parameters: List[Tuple[str, Type]]
    body: List[ASTNode]
    is_inline: bool = False
    
    def accept(self, visitor):
        return visitor.visit_function(self)

@dataclass
class StructNode(ASTNode):
    name: str
    members: List[Tuple[str, Type]]
    
    def accept(self, visitor):
        return visitor.visit_struct(self)

@dataclass
class VarDeclarationNode(ASTNode):
    name: str
    var_type: Type
    initial_value: Optional['ExpressionNode'] = None
    is_const: bool = False
    is_static: bool = False
    is_extern: bool = False
    
    def accept(self, visitor):
        return visitor.visit_var_declaration(self)

@dataclass
class AssignmentNode(ASTNode):
    target: 'ExpressionNode'
    value: 'ExpressionNode'
    operator: str = "="  # =, +=, -=, etc.
    
    def accept(self, visitor):
        return visitor.visit_assignment(self)

# Expression nodes
class ExpressionNode(ASTNode):
    """Base class for all expressions"""
    pass

@dataclass
class BinaryOpNode(ExpressionNode):
    left: ExpressionNode
    operator: str
    right: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_binary_op(self)

@dataclass
class UnaryOpNode(ExpressionNode):
    operator: str
    operand: ExpressionNode
    is_prefix: bool = True
    
    def accept(self, visitor):
        return visitor.visit_unary_op(self)

@dataclass
class LiteralNode(ExpressionNode):
    value: Any
    literal_type: Type
    
    def accept(self, visitor):
        return visitor.visit_literal(self)

@dataclass
class IdentifierNode(ExpressionNode):
    name: str
    
    def accept(self, visitor):
        return visitor.visit_identifier(self)

@dataclass
class ArrayAccessNode(ExpressionNode):
    array: ExpressionNode
    index: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_array_access(self)

@dataclass
class MemberAccessNode(ExpressionNode):
    object: ExpressionNode
    member: str
    is_pointer: bool = False  # True for ->, False for .
    
    def accept(self, visitor):
        return visitor.visit_member_access(self)

@dataclass
class FunctionCallNode(ExpressionNode):
    function: ExpressionNode
    arguments: List[ExpressionNode]
    
    def accept(self, visitor):
        return visitor.visit_function_call(self)

@dataclass
class CastNode(ExpressionNode):
    expression: ExpressionNode
    target_type: Type
    
    def accept(self, visitor):
        return visitor.visit_cast(self)

@dataclass
class TernaryNode(ExpressionNode):
    condition: ExpressionNode
    true_expr: ExpressionNode
    false_expr: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_ternary(self)

# Control flow nodes
@dataclass
class IfNode(ASTNode):
    condition: ExpressionNode
    if_body: List[ASTNode]
    elif_branches: List[Tuple[ExpressionNode, List[ASTNode]]] = field(default_factory=list)
    else_body: Optional[List[ASTNode]] = None
    
    def accept(self, visitor):
        return visitor.visit_if(self)

@dataclass
class WhileNode(ASTNode):
    condition: ExpressionNode
    body: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_while(self)

@dataclass
class DoWhileNode(ASTNode):
    body: List[ASTNode]
    condition: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_do_while(self)

@dataclass
class ForNode(ASTNode):
    init: Optional[ASTNode]
    condition: Optional[ExpressionNode]
    increment: Optional[ASTNode]
    body: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_for(self)

@dataclass
class SwitchNode(ASTNode):
    expression: ExpressionNode
    cases: List[Tuple[Optional[ExpressionNode], List[ASTNode]]]  # None for default
    
    def accept(self, visitor):
        return visitor.visit_switch(self)

@dataclass
class BreakNode(ASTNode):
    def accept(self, visitor):
        return visitor.visit_break(self)

@dataclass
class ContinueNode(ASTNode):
    def accept(self, visitor):
        return visitor.visit_continue(self)

@dataclass
class ReturnNode(ASTNode):
    value: Optional[ExpressionNode] = None
    
    def accept(self, visitor):
        return visitor.visit_return(self)

# I/O and special nodes
@dataclass
class PrintNode(ASTNode):
    expressions: List[ExpressionNode]
    newline: bool = True
    
    def accept(self, visitor):
        return visitor.visit_print(self)

@dataclass
class ScanfNode(ASTNode):
    format_expr: ExpressionNode
    targets: List[ExpressionNode]
    
    def accept(self, visitor):
        return visitor.visit_scanf(self)

@dataclass
class AssemblyNode(ASTNode):
    code: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    clobbers: List[str] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_assembly(self)

@dataclass
class CommentNode(ASTNode):
    text: str
    
    def accept(self, visitor):
        return visitor.visit_comment(self)

@dataclass
class CCodeBlockNode(ASTNode):
    c_code: str
    
    def accept(self, visitor):
        return visitor.visit_c_code_block(self)

@dataclass
class OptimizationHintNode(ASTNode):
    hint_type: str  # "unroll", "vectorize", "likely", etc.
    target: ASTNode
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def accept(self, visitor):
        return visitor.visit_optimization_hint(self)

#  ENHANCED LEXER 
class CASMLexer:
    """Advanced lexer with better error handling and more token types"""
    
    def __init__(self, filename: str = "<stdin>"):
        self.filename = filename
        self.current_line = 1
        self.current_column = 1
        self.current_line_text = ""
        self._in_c_block = False
        self._brace_depth = 0
        
        self.keywords = {
            # Control flow
            'if': TokenType.IF, 'else': TokenType.ELSE, 'elif': TokenType.ELIF,
            'endif': TokenType.ENDIF, 'while': TokenType.WHILE, 'endwhile': TokenType.ENDWHILE,
            'for': TokenType.FOR, 'endfor': TokenType.ENDFOR, 'do': TokenType.DO,
            'break': TokenType.BREAK, 'continue': TokenType.CONTINUE, 'return': TokenType.RETURN,
            'switch': TokenType.SWITCH, 'case': TokenType.CASE, 'default': TokenType.DEFAULT,
            'endswitch': TokenType.ENDSWITCH,
            
            # Declarations
            'var': TokenType.VAR, 'const': TokenType.CONST, 'extern': TokenType.EXTERN,
            'static': TokenType.STATIC, 'volatile': TokenType.VOLATILE, 
            'register': TokenType.REGISTER, 'inline': TokenType.INLINE,
            'function': TokenType.FUNCTION, 'procedure': TokenType.PROCEDURE,
            'struct': TokenType.STRUCT, 'union': TokenType.UNION, 'enum': TokenType.ENUM,
            'typedef': TokenType.TYPEDEF,
            
            # Types
            'int': TokenType.INT_TYPE, 'int8': TokenType.INT8_TYPE, 'int16': TokenType.INT16_TYPE,
            'int32': TokenType.INT32_TYPE, 'int64': TokenType.INT64_TYPE,
            'uint': TokenType.UINT_TYPE, 'uint8': TokenType.UINT8_TYPE, 
            'uint16': TokenType.UINT16_TYPE, 'uint32': TokenType.UINT32_TYPE,
            'uint64': TokenType.UINT64_TYPE, 'str': TokenType.STR_TYPE,
            'bool': TokenType.BOOL_TYPE, 'float': TokenType.FLOAT_TYPE,
            'double': TokenType.DOUBLE_TYPE, 'char': TokenType.CHAR_TYPE,
            'void': TokenType.VOID_TYPE, 'buffer': TokenType.BUFFER_TYPE,
            'ptr': TokenType.PTR_TYPE,
            
            # Literals
            'true': TokenType.TRUE, 'false': TokenType.FALSE, 'null': TokenType.NULL,
            
            # I/O
            'print': TokenType.PRINT, 'println': TokenType.PRINTLN,
            'scan': TokenType.SCAN, 'read': TokenType.READ, 'write': TokenType.WRITE,
            
            # Special
            'in': TokenType.IN, 'range': TokenType.RANGE,
            'sizeof': TokenType.SIZEOF, 'typeof': TokenType.TYPEOF,
            'alignof': TokenType.ALIGNOF, 'offsetof': TokenType.OFFSETOF,
            
            # Optimization hints
            'optimize': TokenType.OPTIMIZE, 'unroll': TokenType.UNROLL,
            'vectorize': TokenType.VECTORIZE, 'likely': TokenType.LIKELY,
            'unlikely': TokenType.UNLIKELY,
            
            # Assembly directives
            'db': TokenType.ASM_DIRECTIVE, 'dw': TokenType.ASM_DIRECTIVE,
            'dd': TokenType.ASM_DIRECTIVE, 'dq': TokenType.ASM_DIRECTIVE,
            'resb': TokenType.ASM_DIRECTIVE, 'resw': TokenType.ASM_DIRECTIVE,
            'resd': TokenType.ASM_DIRECTIVE, 'resq': TokenType.ASM_DIRECTIVE,
            'equ': TokenType.ASM_DIRECTIVE,
        }
        
        # Multi-character operators (order matters - longer first)
        self.operators = [
            ('<<=', TokenType.SHL_ASSIGN), ('>>=', TokenType.SHR_ASSIGN),
            ('+=', TokenType.PLUS_ASSIGN), ('-=', TokenType.MINUS_ASSIGN),
            ('*=', TokenType.MULT_ASSIGN), ('/=', TokenType.DIV_ASSIGN),
            ('%=', TokenType.MOD_ASSIGN), ('&=', TokenType.AND_ASSIGN),
            ('|=', TokenType.OR_ASSIGN), ('^=', TokenType.XOR_ASSIGN),
            ('==', TokenType.EQUALS), ('!=', TokenType.NOT_EQUALS),
            ('<=', TokenType.LESS_EQUAL), ('>=', TokenType.GREATER_EQUAL),
            ('<<', TokenType.LEFT_SHIFT), ('>>', TokenType.RIGHT_SHIFT),
            ('&&', TokenType.LOGICAL_AND), ('||', TokenType.LOGICAL_OR),
            ('++', TokenType.INCREMENT), ('--', TokenType.DECREMENT),
            ('->', TokenType.ARROW), ('**', TokenType.POWER),
            ('=', TokenType.ASSIGN), ('<', TokenType.LESS_THAN),
            ('>', TokenType.GREATER_THAN), ('+', TokenType.PLUS),
            ('-', TokenType.MINUS), ('*', TokenType.MULTIPLY),
            ('/', TokenType.DIVIDE), ('%', TokenType.MODULO),
            ('&', TokenType.BIT_AND), ('|', TokenType.BIT_OR),
            ('^', TokenType.BIT_XOR), ('~', TokenType.BIT_NOT),
            ('!', TokenType.LOGICAL_NOT),
        ]
        
        self.punctuation = {
            '(': TokenType.LEFT_PAREN, ')': TokenType.RIGHT_PAREN,
            '[': TokenType.LEFT_BRACKET, ']': TokenType.RIGHT_BRACKET,
            '{': TokenType.LEFT_BRACE, '}': TokenType.RIGHT_BRACE,
            ',': TokenType.COMMA, ';': TokenType.SEMICOLON,
            ':': TokenType.COLON, '.': TokenType.DOT,
            '?': TokenType.QUESTION, '#': TokenType.HASH,
            '@': TokenType.AT, '$': TokenType.DOLLAR,
        }
    
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize input text"""
        tokens = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            self.current_line = line_num
            self.current_line_text = line
            self.current_column = 1
            
            line_tokens = self._tokenize_line(line)
            tokens.extend(line_tokens)
            
            if line.strip() and line_tokens:
                tokens.append(self._make_token(TokenType.NEWLINE, '\n'))
        
        tokens.append(self._make_token(TokenType.EOF, ''))
        return tokens
    
    def _tokenize_line(self, line: str) -> List[Token]:
        """Tokenize a single line"""
        tokens = []
        i = 0
        
        while i < len(line):
            # Skip whitespace
            if line[i].isspace():
                self.current_column += 1
                i += 1
                continue
            
            # Comments
            if i + 1 < len(line) and line[i:i+2] == '//':
                comment = line[i:]
                tokens.append(self._make_token(TokenType.COMMENT, comment))
                break
            
            if line[i] == ';':
                comment = line[i:]
                tokens.append(self._make_token(TokenType.COMMENT, comment))
                break
            
            # Strings
            if line[i] in ['"', "'"]:
                string_val, new_i = self._extract_string(line, i)
                tokens.append(self._make_token(TokenType.STRING, string_val))
                i = new_i
                continue
            
            # Character literals
            if line[i] == "'" and i + 2 < len(line):
                char_val, new_i = self._extract_char(line, i)
                tokens.append(self._make_token(TokenType.CHAR_LITERAL, char_val))
                i = new_i
                continue
            
            # Numbers
            if line[i].isdigit() or (line[i] == '0' and i + 1 < len(line) and line[i+1] in 'xXbBoO'):
                number_val, number_type, new_i = self._extract_number(line, i)
                tokens.append(self._make_token(number_type, number_val))
                i = new_i
                continue
            
            # Identifiers and keywords
            if line[i].isalpha() or line[i] == '_':
                identifier, new_i = self._extract_identifier(line, i)
                token_type = self.keywords.get(identifier.lower(), TokenType.IDENTIFIER)
                tokens.append(self._make_token(token_type, identifier))
                i = new_i
                continue
            
            # Operators
            matched = False
            for op_text, op_type in self.operators:
                if line[i:i+len(op_text)] == op_text:
                    tokens.append(self._make_token(op_type, op_text))
                    i += len(op_text)
                    self.current_column += len(op_text)
                    matched = True
                    break
            
            if matched:
                continue
            
            # Punctuation
            if line[i] in self.punctuation:
                token_type = self.punctuation[line[i]]
                tokens.append(self._make_token(token_type, line[i]))
                i += 1
                self.current_column += 1
                continue
            
            # Unknown character - treat as assembly
            assembly_code = line[i:].strip()
            if assembly_code:
                tokens.append(self._make_token(TokenType.ASSEMBLY_LINE, assembly_code))
            break
        
        return tokens
    
    def _make_token(self, token_type: TokenType, value: str) -> Token:
        """Create a token with location info"""
        location = SourceLocation(
            line=self.current_line,
            column=self.current_column,
            file=self.filename,
            raw_line=self.current_line_text
        )
        return Token(type=token_type, value=value, location=location)
    
    def _extract_string(self, line: str, start: int) -> Tuple[str, int]:
        """Extract string literal"""
        quote_char = line[start]
        i = start + 1
        result = quote_char
        
        while i < len(line):
            char = line[i]
            result += char
            
            if char == quote_char:
                i += 1
                break
            elif char == '\\' and i + 1 < len(line):
                i += 1
                if i < len(line):
                    result += line[i]
            i += 1
        
        self.current_column += len(result)
        return result, i
    
    def _extract_char(self, line: str, start: int) -> Tuple[str, int]:
        """Extract character literal"""
        i = start + 1
        result = "'"
        
        if i < len(line):
            if line[i] == '\\' and i + 1 < len(line):
                result += line[i]
                i += 1
                if i < len(line):
                    result += line[i]
                    i += 1
            else:
                result += line[i]
                i += 1
        
        if i < len(line) and line[i] == "'":
            result += "'"
            i += 1
        
        self.current_column += len(result)
        return result, i
    
    def _extract_number(self, line: str, start: int) -> Tuple[str, TokenType, int]:
        """Extract number literal (int, float, hex, binary, octal)"""
        i = start
        result = ""
        number_type = TokenType.NUMBER
        
        # Hex, binary, or octal
        if line[i] == '0' and i + 1 < len(line):
            if line[i+1] in 'xX':
                result = line[i:i+2]
                i += 2
                while i < len(line) and line[i] in '0123456789abcdefABCDEF_':
                    if line[i] != '_':
                        result += line[i]
                    i += 1
                number_type = TokenType.HEX_NUMBER
                self.current_column += len(result)
                return result, number_type, i
            
            elif line[i+1] in 'bB':
                result = line[i:i+2]
                i += 2
                while i < len(line) and line[i] in '01_':
                    if line[i] != '_':
                        result += line[i]
                    i += 1
                number_type = TokenType.BIN_NUMBER
                self.current_column += len(result)
                return result, number_type, i
            
            elif line[i+1] in 'oO':
                result = line[i:i+2]
                i += 2
                while i < len(line) and line[i] in '01234567_':
                    if line[i] != '_':
                        result += line[i]
                    i += 1
                number_type = TokenType.OCT_NUMBER
                self.current_column += len(result)
                return result, number_type, i
        
        # Regular number (int or float)
        has_dot = False
        has_exp = False
        
        while i < len(line):
            char = line[i]
            
            if char.isdigit() or char == '_':
                if char != '_':
                    result += char
                i += 1
            elif char == '.' and not has_dot and not has_exp:
                has_dot = True
                result += char
                i += 1
                number_type = TokenType.FLOAT_NUMBER
            elif char in 'eE' and not has_exp:
                has_exp = True
                result += char
                i += 1
                number_type = TokenType.FLOAT_NUMBER
                if i < len(line) and line[i] in '+-':
                    result += line[i]
                    i += 1
            else:
                break
        
        self.current_column += len(result)
        return result, number_type, i
    
    def _extract_identifier(self, line: str, start: int) -> Tuple[str, int]:
        """Extract identifier"""
        i = start
        result = ""
        
        while i < len(line):
            char = line[i]
            if char.isalnum() or char in ['_', '%', ']:
                result += char
                i += 1
            else:
                break
        
        self.current_column += len(result)
        return result, i

#  SEMANTIC ANALYZER 
class SemanticError(Exception):
    """Semantic analysis error"""
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        self.message = message
        self.location = location
        super().__init__(f"Semantic error at {location}: {message}" if location else message)

class SemanticAnalyzer(ABC):
    """Performs semantic analysis and type checking"""
    
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[SemanticError] = []
        self.warnings: List[str] = []
        self.current_function: Optional[Symbol] = None
        self.loop_depth = 0
        
    def analyze(self, ast: ProgramNode) -> bool:
        """Analyze AST and return True if no errors"""
        try:
            self._analyze_program(ast)
            return len(self.errors) == 0
        except SemanticError as e:
            self.errors.append(e)
            return False
    
    def _analyze_program(self, node: ProgramNode):
        """Analyze program node"""
        for stmt in node.statements:
            self._analyze_node(stmt)
    
    def _analyze_node(self, node: ASTNode):
        """Dispatch to appropriate analyzer"""
        if isinstance(node, VarDeclarationNode):
            self._analyze_var_declaration(node)
        elif isinstance(node, FunctionNode):
            self._analyze_function(node)
        elif isinstance(node, AssignmentNode):
            self._analyze_assignment(node)
        elif isinstance(node, IfNode):
            self._analyze_if(node)
        elif isinstance(node, WhileNode):
            self._analyze_while(node)
        elif isinstance(node, ForNode):
            self._analyze_for(node)
        elif isinstance(node, ReturnNode):
            self._analyze_return(node)
        elif isinstance(node, ExpressionNode):
            self._analyze_expression(node)
    
    def _analyze_var_declaration(self, node: VarDeclarationNode):
        """Analyze variable declaration"""
        # Check if already defined in current scope
        existing = self.symbol_table.lookup_current_scope(node.name)
        if existing:
            self.errors.append(SemanticError(
                f"Variable '{node.name}' already defined in current scope",
                node.location
            ))
            return
        
        # Check initial value type if present
        if node.initial_value:
            expr_type = self._analyze_expression(node.initial_value)
            if expr_type and not node.var_type.is_compatible_with(expr_type):
                self.errors.append(SemanticError(
                    f"Type mismatch: cannot assign {expr_type} to {node.var_type}",
                    node.location
                ))
        
        # Add to symbol table
        symbol = Symbol(
            name=node.name,
            type=node.var_type,
            scope_level=self.symbol_table.current_scope_level,
            location=node.location,
            is_extern=node.is_extern,
            is_static=node.is_static
        )
        self.symbol_table.define(symbol)
        node.type = node.var_type
    
    def _analyze_function(self, node: FunctionNode):
        """Analyze function definition"""
        # Create function type
        param_types = [param_type for _, param_type in node.parameters]
        func_type = Type(
            kind=TypeKind.FUNCTION,
            name=node.name,
            param_types=param_types,
            return_type=node.return_type
        )
        
        # Add function to symbol table
        symbol = Symbol(
            name=node.name,
            type=func_type,
            scope_level=0,  # Functions are always global
            location=node.location,
            is_function=True
        )
        
        if not self.symbol_table.define(symbol):
            self.errors.append(SemanticError(
                f"Function '{node.name}' already defined",
                node.location
            ))
            return
        
        # Enter function scope
        old_function = self.current_function
        self.current_function = symbol
        self.symbol_table.enter_scope()
        
        # Add parameters to scope
        for param_name, param_type in node.parameters:
            param_symbol = Symbol(
                name=param_name,
                type=param_type,
                scope_level=self.symbol_table.current_scope_level,
                is_parameter=True
            )
            self.symbol_table.define(param_symbol)
        
        # Analyze function body
        for stmt in node.body:
            self._analyze_node(stmt)
        
        # Exit function scope
        self.symbol_table.exit_scope()
        self.current_function = old_function
    
    def _analyze_assignment(self, node: AssignmentNode):
        """Analyze assignment"""
        target_type = self._analyze_expression(node.target)
        value_type = self._analyze_expression(node.value)
        
        if target_type and value_type:
            if not target_type.is_compatible_with(value_type):
                self.errors.append(SemanticError(
                    f"Type mismatch: cannot assign {value_type} to {target_type}",
                    node.location
                ))
    
    def _analyze_if(self, node: IfNode):
        """Analyze if statement"""
        cond_type = self._analyze_expression(node.condition)
        
        if cond_type and cond_type.name != 'bool':
            self.warnings.append(f"Condition should be boolean, got {cond_type}")
        
        self.symbol_table.enter_scope()
        for stmt in node.if_body:
            self._analyze_node(stmt)
        self.symbol_table.exit_scope()
        
        for elif_cond, elif_body in node.elif_branches:
            self._analyze_expression(elif_cond)
            self.symbol_table.enter_scope()
            for stmt in elif_body:
                self._analyze_node(stmt)
            self.symbol_table.exit_scope()
        
        if node.else_body:
            self.symbol_table.enter_scope()
            for stmt in node.else_body:
                self._analyze_node(stmt)
            self.symbol_table.exit_scope()
    
    def _analyze_while(self, node: WhileNode):
        """Analyze while loop"""
        self._analyze_expression(node.condition)
        self.loop_depth += 1
        self.symbol_table.enter_scope()
        
        for stmt in node.body:
            self._analyze_node(stmt)
        
        self.symbol_table.exit_scope()
        self.loop_depth -= 1
    
    def _analyze_for(self, node: ForNode):
        """Analyze for loop"""
        self.symbol_table.enter_scope()
        
        if node.init:
            self._analyze_node(node.init)
        if node.condition:
            self._analyze_expression(node.condition)
        if node.increment:
            self._analyze_node(node.increment)
        
        self.loop_depth += 1
        for stmt in node.body:
            self._analyze_node(stmt)
        self.loop_depth -= 1
        
        self.symbol_table.exit_scope()
    
    def _analyze_return(self, node: ReturnNode):
        """Analyze return statement"""
        if not self.current_function:
            self.errors.append(SemanticError(
                "Return statement outside function",
                node.location
            ))
            return
        
        return_type = self.current_function.type.return_type
        
        if node.value:
            value_type = self._analyze_expression(node.value)
            if value_type and not return_type.is_compatible_with(value_type):
                self.errors.append(SemanticError(
                    f"Return type mismatch: expected {return_type}, got {value_type}",
                    node.location
                ))
        elif return_type.kind != TypeKind.VOID:
            self.errors.append(SemanticError(
                f"Function must return a value of type {return_type}",
                node.location
            ))
    
    def _analyze_expression(self, node: ExpressionNode) -> Optional[Type]:
        """Analyze expression and return its type"""
        if isinstance(node, LiteralNode):
            node.type = node.literal_type
            return node.literal_type
        
        elif isinstance(node, IdentifierNode):
            symbol = self.symbol_table.lookup(node.name)
            if not symbol:
                self.errors.append(SemanticError(
                    f"Undefined variable '{node.name}'",
                    node.location
                ))
                return None
            node.type = symbol.type
            return symbol.type
        
        elif isinstance(node, BinaryOpNode):
            left_type = self._analyze_expression(node.left)
            right_type = self._analyze_expression(node.right)
            
            if left_type and right_type:
                # Type checking for operators
                if node.operator in ['+', '-', '*', '/', '%']:
                    if left_type._is_numeric() and right_type._is_numeric():
                        # Result type promotion
                        node.type = left_type if left_type.size >= right_type.size else right_type
                        return node.type
                    else:
                        self.errors.append(SemanticError(
                            f"Arithmetic operators require numeric types",
                            node.location
                        ))
                
                elif node.operator in ['==', '!=', '<', '>', '<=', '>=']:
                    if left_type.is_compatible_with(right_type):
                        node.type = BOOL_TYPE
                        return BOOL_TYPE
                    else:
                        self.errors.append(SemanticError(
                            f"Cannot compare {left_type} with {right_type}",
                            node.location
                        ))
                
                elif node.operator in ['&&', '||']:
                    node.type = BOOL_TYPE
                    return BOOL_TYPE
            
            return None
        
        elif isinstance(node, UnaryOpNode):
            operand_type = self._analyze_expression(node.operand)
            
            if operand_type:
                if node.operator in ['++', '--']:
                    if operand_type._is_numeric():
                        node.type = operand_type
                        return operand_type
                elif node.operator == '!':
                    node.type = BOOL_TYPE
                    return BOOL_TYPE
                elif node.operator in ['+', '-', '~']:
                    if operand_type._is_numeric():
                        node.type = operand_type
                        return operand_type
            
            return None
        
        elif isinstance(node, ArrayAccessNode):
            array_type = self._analyze_expression(node.array)
            index_type = self._analyze_expression(node.index)
            
            if array_type and array_type.kind == TypeKind.ARRAY:
                node.type = array_type.base_type
                return array_type.base_type
            elif array_type and array_type.kind == TypeKind.POINTER:
                node.type = array_type.base_type
                return array_type.base_type
            else:
                self.errors.append(SemanticError(
                    f"Cannot index non-array type {array_type}",
                    node.location
                ))
            
            return None
        
        elif isinstance(node, FunctionCallNode):
            func_type = self._analyze_expression(node.function)
            
            if func_type and func_type.kind == TypeKind.FUNCTION:
                # Check argument count
                if len(node.arguments) != len(func_type.param_types):
                    self.errors.append(SemanticError(
                        f"Function expects {len(func_type.param_types)} arguments, got {len(node.arguments)}",
                        node.location
                    ))
                
                # Check argument types
                for i, (arg, param_type) in enumerate(zip(node.arguments, func_type.param_types)):
                    arg_type = self._analyze_expression(arg)
                    if arg_type and not param_type.is_compatible_with(arg_type):
                        self.errors.append(SemanticError(
                            f"Argument {i+1} type mismatch: expected {param_type}, got {arg_type}",
                            node.location
                        ))
                
                node.type = func_type.return_type
                return func_type.return_type
            
            return None
        
        elif isinstance(node, CastNode):
            self._analyze_expression(node.expression)
            node.type = node.target_type
            return node.target_type
        
        elif isinstance(node, TernaryNode):
            self._analyze_expression(node.condition)
            true_type = self._analyze_expression(node.true_expr)
            false_type = self._analyze_expression(node.false_expr)
            
            if true_type and false_type:
                if true_type.is_compatible_with(false_type):
                    node.type = true_type
                    return true_type
                else:
                    self.errors.append(SemanticError(
                        f"Ternary branches have incompatible types: {true_type} and {false_type}",
                        node.location
                    ))
            
            return None
        
        return None

#  OPTIMIZER 
class OptimizationPass(ABC):
    """Base class for optimization passes"""
    
    @abstractmethod
    def optimize(self, ast: ProgramNode) -> ProgramNode:
        pass

class ConstantFoldingPass(OptimizationPass):
    """Fold constant expressions at compile time"""
    
    def optimize(self, ast: ProgramNode) -> ProgramNode:
        return self._visit_program(ast)
    
    def _visit_program(self, node: ProgramNode) -> ProgramNode:
        node.statements = [self._visit_node(stmt) for stmt in node.statements]
        return node
    
    def _visit_node(self, node: ASTNode) -> ASTNode:
        if isinstance(node, BinaryOpNode):
            return self._fold_binary_op(node)
        elif isinstance(node, UnaryOpNode):
            return self._fold_unary_op(node)
        # Add more node types as needed
        return node
    
    def _fold_binary_op(self, node: BinaryOpNode) -> ASTNode:
        left = self._visit_node(node.left)
        right = self._visit_node(node.right)
        
        if isinstance(left, LiteralNode) and isinstance(right, LiteralNode):
            try:
                if node.operator == '+':
                    result = left.value + right.value
                elif node.operator == '-':
                    result = left.value - right.value
                elif node.operator == '*':
                    result = left.value * right.value
                elif node.operator == '/':
                    if right.value != 0:
                        result = left.value / right.value
                    else:
                        return node
                elif node.operator == '%':
                    if right.value != 0:
                        result = left.value % right.value
                    else:
                        return node
                else:
                    return node
                
                return LiteralNode(
                    value=result,
                    literal_type=left.literal_type,
                    location=node.location
                )
            except Exception:
                return node
        
        node.left = left
        node.right = right
        return node
    
    def _fold_unary_op(self, node: UnaryOpNode) -> ASTNode:
        operand = self._visit_node(node.operand)
        
        if isinstance(operand, LiteralNode):
            try:
                if node.operator == '-':
                    result = -operand.value
                elif node.operator == '+':
                    result = +operand.value
                elif node.operator == '!':
                    result = not operand.value
                elif node.operator == '~':
                    result = ~operand.value
                else:
                    return node
                
                return LiteralNode(
                    value=result,
                    literal_type=operand.literal_type,
                    location=node.location
                )
            except Exception:
                return node
        
        node.operand = operand
        return node

class DeadCodeEliminationPass(OptimizationPass):
    """Remove unreachable code"""
    
    def optimize(self, ast: ProgramNode) -> ProgramNode:
        ast.statements = self._eliminate_dead_code(ast.statements)
        return ast
    
    def _eliminate_dead_code(self, statements: List[ASTNode]) -> List[ASTNode]:
        result = []
        reachable = True
        
        for stmt in statements:
            if not reachable:
                break
            
            result.append(stmt)
            
            if isinstance(stmt, ReturnNode):
                reachable = False
            elif isinstance(stmt, BreakNode) or isinstance(stmt, ContinueNode):
                reachable = False
        
        return result

class Optimizer:
    """Main optimizer that runs multiple passes"""
    
    def __init__(self, optimization_level: int = 2):
        self.optimization_level = optimization_level
        self.passes: List[OptimizationPass] = []
        
        if optimization_level >= 1:
            self.passes.append(ConstantFoldingPass())
        
        if optimization_level >= 2:
            self.passes.append(DeadCodeEliminationPass())
    
    def optimize(self, ast: ProgramNode) -> ProgramNode:
        """Run all optimization passes"""
        for pass_obj in self.passes:
            ast = pass_obj.optimize(ast)
        return ast

#  ADVANCED CODE GENERATOR 
class CodeGenerationContext:
    """Context for code generation"""
    
    def __init__(self):
        self.temp_counter = 0
        self.label_counter = 0
        self.string_counter = 0
        self.register_pool = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11']
        self.used_registers: Set[str] = set()
        self.stack_offset = 0
        
    def allocate_temp(self) -> str:
        self.temp_counter += 1
        return f"T{self.temp_counter}"
    
    def allocate_label(self, prefix: str = "L") -> str:
        self.label_counter += 1
        return f"{prefix}{self.label_counter}"
    
    def allocate_string_label(self) -> str:
        self.string_counter += 1
        return f"STR{self.string_counter}"
    
    def allocate_register(self) -> Optional[str]:
        for reg in self.register_pool:
            if reg not in self.used_registers:
                self.used_registers.add(reg)
                return reg
        return None
    
    def free_register(self, reg: str):
        self.used_registers.discard(reg)

# Visitor interface for code generation
class ASTVisitor(ABC):
    @abstractmethod
    def visit_program(self, node: ProgramNode): pass
    @abstractmethod
    def visit_function(self, node: FunctionNode): pass
    @abstractmethod
    def visit_struct(self, node: StructNode): pass
    @abstractmethod
    def visit_var_declaration(self, node: VarDeclarationNode): pass
    @abstractmethod
    def visit_assignment(self, node: AssignmentNode): pass
    @abstractmethod
    def visit_binary_op(self, node: BinaryOpNode): pass
    @abstractmethod
    def visit_unary_op(self, node: UnaryOpNode): pass
    @abstractmethod
    def visit_literal(self, node: LiteralNode): pass
    @abstractmethod
    def visit_identifier(self, node: IdentifierNode): pass
    @abstractmethod
    def visit_array_access(self, node: ArrayAccessNode): pass
    @abstractmethod
    def visit_member_access(self, node: MemberAccessNode): pass
    @abstractmethod
    def visit_function_call(self, node: FunctionCallNode): pass
    @abstractmethod
    def visit_cast(self, node: CastNode): pass
    @abstractmethod
    def visit_ternary(self, node: TernaryNode): pass
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
    def visit_print(self, node: PrintNode): pass
    @abstractmethod
    def visit_scanf(self, node: ScanfNode): pass
    @abstractmethod
    def visit_assembly(self, node: AssemblyNode): pass
    @abstractmethod
    def visit_comment(self, node: CommentNode): pass
    @abstractmethod
    def visit_c_code_block(self, node: CCodeBlockNode): pass
    @abstractmethod
    def visit_optimization_hint(self, node: OptimizationHintNode): pass