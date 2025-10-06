#!/usr/bin/env python3

import re
from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

class TokenType(Enum):
    # Control flow
    IF = "IF"
    ELSE = "ELSE"
    ENDIF = "ENDIF"
    WHILE = "WHILE"
    ENDWHILE = "ENDWHILE"
    FOR = "FOR"
    ENDFOR = "ENDFOR"
    DO = "DO"
    SWITCH = "SWITCH"
    CASE = "CASE"
    DEFAULT = "DEFAULT"
    ENDSWITCH = "ENDSWITCH"
    
    # Functions
    FUNCTION = "FUNCTION"
    ENDFUNCTION = "ENDFUNCTION"
    CALL = "CALL"
    RETURN = "RETURN"
    
    # Variables
    VAR = "VAR"
    CONST = "CONST"
    
    # Standard library
    PRINT = "PRINT"
    PRINTLN = "PRINTLN"
    INPUT = "INPUT"
    SCANF = "SCANF"
    STRLEN = "STRLEN"
    STRCPY = "STRCPY"
    STRCMP = "STRCMP"
    MEMSET = "MEMSET"
    MEMCPY = "MEMCPY"
    ATOI = "ATOI"
    ITOA = "ITOA"
    EXIT = "EXIT"
    
    # Loop control
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    
    # Operators
    ASSIGN = "ASSIGN"
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    LESS_EQUAL = "LESS_EQUAL"
    GREATER_EQUAL = "GREATER_EQUAL"
    
    # Punctuation
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    COMMA = "COMMA"
    
    # Literals and identifiers
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    
    # Special
    NEWLINE = "NEWLINE"
    COMMENT = "COMMENT"
    ASSEMBLY = "ASSEMBLY"
    SECTION = "SECTION"
    IN = "IN"
    RANGE = "RANGE"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self):
        self.tokens = []
        self.current_line = 1
        self.current_column = 1
        
        # Keywords mapping
        self.keywords = {
            '%if': TokenType.IF,
            '%else': TokenType.ELSE,
            '%endif': TokenType.ENDIF,
            '%while': TokenType.WHILE,
            '%endwhile': TokenType.ENDWHILE,
            '%for': TokenType.FOR,
            '%endfor': TokenType.ENDFOR,
            '%do': TokenType.DO,
            '%switch': TokenType.SWITCH,
            '%case': TokenType.CASE,
            '%default': TokenType.DEFAULT,
            '%endswitch': TokenType.ENDSWITCH,
            '%function': TokenType.FUNCTION,
            '%endfunction': TokenType.ENDFUNCTION,
            '%call': TokenType.CALL,
            '%return': TokenType.RETURN,
            '%var': TokenType.VAR,
            '%const': TokenType.CONST,
            '%print': TokenType.PRINT,
            '%println': TokenType.PRINTLN,
            '%input': TokenType.INPUT,
            '%scanf': TokenType.SCANF,
            '%strlen': TokenType.STRLEN,
            '%strcpy': TokenType.STRCPY,
            '%strcmp': TokenType.STRCMP,
            '%memset': TokenType.MEMSET,
            '%memcpy': TokenType.MEMCPY,
            '%atoi': TokenType.ATOI,
            '%itoa': TokenType.ITOA,
            '%exit': TokenType.EXIT,
            '%break': TokenType.BREAK,
            '%continue': TokenType.CONTINUE,
            'in': TokenType.IN,
            'range': TokenType.RANGE,
        }
        
        # Operator patterns
        self.operators = [
            ('==', TokenType.EQUALS),
            ('!=', TokenType.NOT_EQUALS),
            ('<=', TokenType.LESS_EQUAL),
            ('>=', TokenType.GREATER_EQUAL),
            ('<', TokenType.LESS_THAN),
            ('>', TokenType.GREATER_THAN),
            ('=', TokenType.ASSIGN),
            ('[', TokenType.LBRACKET),
            (']', TokenType.RBRACKET),
            ('{', TokenType.LBRACE),
            ('}', TokenType.RBRACE),
            (',', TokenType.COMMA),
        ]
    
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize the input text"""
        self.tokens = []
        self.current_line = 1
        self.current_column = 1
        
        lines = text.split('\n')
        
        for line_text in lines:
            self._tokenize_line(line_text)
            self.current_line += 1
            self.current_column = 1
        
        return self.tokens
    
    def tokenize_file(self, filename: str) -> List[Token]:
        """Tokenize a file"""
        try:
            with open(filename, 'r') as f:
                text = f.read()
            return self.tokenize(text)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' not found")
    
    def _tokenize_line(self, line: str):
        """Tokenize a single line"""
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            self._add_token(TokenType.NEWLINE, '\n')
            return
        
        # Handle comments
        if stripped.startswith(';'):
            self._add_token(TokenType.COMMENT, stripped)
            self._add_token(TokenType.NEWLINE, '\n')
            return
        
        # Handle section directives
        if stripped.startswith('section'):
            self._add_token(TokenType.SECTION, stripped)
            self._add_token(TokenType.NEWLINE, '\n')
            return
        
        # Check if this is a high-level construct
        if self._is_hlasm_directive(stripped):
            self._tokenize_directive(line)
        else:
            # Regular assembly instruction
            self._add_token(TokenType.ASSEMBLY, line.rstrip())
        
        self._add_token(TokenType.NEWLINE, '\n')
    
    def _is_hlasm_directive(self, line: str) -> bool:
        """Check if line contains a high-level directive"""
        return line.startswith('%') or any(keyword in line for keyword in ['in', 'range'])
    
    def _tokenize_directive(self, line: str):
        """Tokenize a high-level directive"""
        i = 0
        while i < len(line):
            # Skip whitespace
            if line[i].isspace():
                i += 1
                self.current_column += 1
                continue
            
            # Check for operators (multi-character first)
            found_op = False
            for op_text, op_type in self.operators:
                if line[i:i+len(op_text)] == op_text:
                    self._add_token(op_type, op_text)
                    i += len(op_text)
                    self.current_column += len(op_text)
                    found_op = True
                    break
            
            if found_op:
                continue
            
            # Check for strings
            if line[i] in ['"', "'"]:
                string_val, new_i = self._extract_string(line, i)
                self._add_token(TokenType.STRING, string_val)
                i = new_i
                continue
            
            # Check for numbers
            if line[i].isdigit():
                num_val, new_i = self._extract_number(line, i)
                self._add_token(TokenType.NUMBER, num_val)
                i = new_i
                continue
            
            # Check for identifiers/keywords
            if line[i].isalpha() or line[i] in ['%', '_']:
                identifier, new_i = self._extract_identifier(line, i)
                token_type = self.keywords.get(identifier.lower(), TokenType.IDENTIFIER)
                self._add_token(token_type, identifier)
                i = new_i
                continue
            
            # Skip other characters
            i += 1
            self.current_column += 1
    
    def _extract_string(self, line: str, start: int) -> Tuple[str, int]:
        """Extract a string literal"""
        quote_char = line[start]
        i = start + 1
        result = quote_char
        
        while i < len(line):
            result += line[i]
            if line[i] == quote_char:
                return result, i + 1
            i += 1
        
        # Unclosed string
        return result, i
    
    def _extract_number(self, line: str, start: int) -> Tuple[str, int]:
        """Extract a number"""
        i = start
        result = ""
        
        while i < len(line) and (line[i].isdigit() or line[i] == '.'):
            result += line[i]
            i += 1
        
        return result, i
    
    def _extract_identifier(self, line: str, start: int) -> Tuple[str, int]:
        """Extract an identifier or keyword"""
        i = start
        result = ""
        
        while i < len(line) and (line[i].isalnum() or line[i] in ['%', '_']):
            result += line[i]
            i += 1
        
        return result, i
    
    def _add_token(self, token_type: TokenType, value: str):
        """Add a token to the list"""
        token = Token(token_type, value, self.current_line, self.current_column)
        self.tokens.append(token)
        self.current_column += len(value)
    
    def get_tokens_by_type(self, token_type: TokenType) -> List[Token]:
        """Get all tokens of a specific type"""
        return [token for token in self.tokens if token.type == token_type]
    
    def get_tokens_by_line(self, line_num: int) -> List[Token]:
        """Get all tokens from a specific line"""
        return [token for token in self.tokens if token.line == line_num]