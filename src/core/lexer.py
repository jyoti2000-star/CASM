#!/usr/bin/env python3
from typing import List, Tuple, Optional, Dict
import re
from .tokens import Token, TokenType, LexerError

class CASMLexer:
    """Clean lexical analyzer for CASM language"""
    
    def __init__(self):
        self.current_line = 1
        self.current_column = 1
        self.current_line_text = ""
        
        # Essential keywords only
        self.keywords = {
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'endif': TokenType.ENDIF,
            'while': TokenType.WHILE,
            'endwhile': TokenType.ENDWHILE,
            'for': TokenType.FOR,
            'endfor': TokenType.ENDFOR,
            'print': TokenType.PRINT,
            'scan': TokenType.SCAN,
            'var': TokenType.VAR,
            
            # Data types with @ prefix (handled separately)
            'int': TokenType.INT_TYPE,
            'str': TokenType.STR_TYPE,
            'bool': TokenType.BOOL_TYPE,
            'float': TokenType.FLOAT_TYPE,
            'buffer': TokenType.BUFFER_TYPE,
            'in': TokenType.IN,
            'range': TokenType.RANGE,
            'extern': TokenType.EXTERN,
        }
        
        # Operators (ordered by length for proper matching)
        self.operators = [
            ('==', TokenType.EQUALS),
            ('!=', TokenType.NOT_EQUALS),
            ('<=', TokenType.LESS_EQUAL),
            ('>=', TokenType.GREATER_EQUAL),
            ('=', TokenType.ASSIGN),
            ('<', TokenType.LESS_THAN),
            ('>', TokenType.GREATER_THAN),
            ('+', TokenType.PLUS),
            ('-', TokenType.MINUS),
            ('*', TokenType.MULTIPLY),
            ('/', TokenType.DIVIDE),
            ('%', TokenType.MODULO),
            # removed '@' legacy operator
        ]
        
        # Punctuation
        self.punctuation = {
            '(': TokenType.LEFT_PAREN,
            ')': TokenType.RIGHT_PAREN,
            '[': TokenType.LEFT_BRACKET,
            ']': TokenType.RIGHT_BRACKET,
            '{': TokenType.LEFT_BRACE,
            '}': TokenType.RIGHT_BRACE,
            ',': TokenType.COMMA,
            ';': TokenType.SEMICOLON,
            ':': TokenType.COLON,
        }
    
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize input text and return list of tokens"""
        tokens = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            self.current_line = line_num
            self.current_line_text = line
            self.current_column = 1
            
            line_tokens = self._tokenize_line(line)
            tokens.extend(line_tokens)
            
            # Add newline token if line is not empty
            if line.strip():
                tokens.append(Token(TokenType.NEWLINE, '\n', line_num, len(line) + 1, line))
        
        # Add EOF token
        tokens.append(Token(TokenType.EOF, '', self.current_line, self.current_column, ''))
        
        return tokens
    
    def _tokenize_line(self, line: str) -> List[Token]:
        """Tokenize a single line"""
        # Heuristic: if a line looks like C (ends with ';', starts with '#',
        # or uses C-style control constructs with parentheses), return a
        # single C_INLINE token so the parser can collect contiguous C lines.
        stripped_line = line.strip()
        if stripped_line:
            # C preprocessor or statement ending with semicolon
            if stripped_line.startswith('#') or stripped_line.endswith(';'):
                return [Token(TokenType.C_INLINE, stripped_line, self.current_line, 1, line)]
            # Do not treat C-style 'for/if/while(...)' as C here; loops are
            # CASM constructs and should be tokenized as such. C statements
            # remain only those ending with ';' or starting with '#'.

        tokens = []
        i = 0
        
        while i < len(line):
            # Skip whitespace
            if line[i].isspace():
                self.current_column += 1
                i += 1
                continue
            
            # Comments
            if line[i:i+1] == ';':
                comment_text = line[i:].rstrip()
                tokens.append(Token(TokenType.COMMENT, comment_text, 
                                  self.current_line, self.current_column, line))
                break  # Rest of line is comment
            
            # Strings
            if line[i] in ['"', "'"]:
                string_value, new_i = self._extract_string(line, i)
                tokens.append(Token(TokenType.STRING, string_value,
                                  self.current_line, self.current_column, line))
                i = new_i
                continue
            
            # Numbers
            if line[i].isdigit():
                number_value, new_i = self._extract_number(line, i)
                tokens.append(Token(TokenType.NUMBER, number_value,
                                  self.current_line, self.current_column, line))
                i = new_i
                continue
            
            # Keywords and identifiers
            if line[i].isalpha() or line[i] == '_':
                identifier, new_i = self._extract_identifier(line, i)
                
                # Check if it's a keyword
                token_type = self.keywords.get(identifier.lower(), TokenType.IDENTIFIER)
                
                tokens.append(Token(token_type, identifier,
                              self.current_line, self.current_column, line))
                
                i = new_i
                continue
            
            # Operators (check multi-character first)
            found_operator = False
            for op_text, op_type in self.operators:
                if line[i:i+len(op_text)] == op_text:
                    tokens.append(Token(op_type, op_text,
                                      self.current_line, self.current_column, line))
                    i += len(op_text)
                    self.current_column += len(op_text)
                    found_operator = True
                    break
            
            if found_operator:
                continue
            
            # Punctuation
            if line[i] in self.punctuation:
                token_type = self.punctuation[line[i]]
                tokens.append(Token(token_type, line[i],
                                  self.current_line, self.current_column, line))
                i += 1
                self.current_column += 1
                continue
            
            # If we reach here, it's likely raw assembly code
            # Treat the rest of the line as assembly
            assembly_code = line[i:].strip()
            if assembly_code:
                tokens.append(Token(TokenType.ASSEMBLY_LINE, assembly_code,
                                  self.current_line, self.current_column, line))
            break
        
        return tokens
    
    def _extract_string(self, line: str, start: int) -> Tuple[str, int]:
        """Extract a string literal"""
        quote_char = line[start]
        i = start + 1
        result = quote_char
        
        while i < len(line):
            char = line[i]
            result += char
            
            if char == quote_char:
                # End of string
                i += 1
                break
            elif char == '\\' and i + 1 < len(line):
                # Escape sequence
                i += 1
                if i < len(line):
                    result += line[i]
            
            i += 1
        
        self.current_column += len(result)
        return result, i
    
    def _extract_number(self, line: str, start: int) -> Tuple[str, int]:
        """Extract a number (integer or float)"""
        i = start
        result = ""
        has_dot = False
        
        while i < len(line):
            char = line[i]
            if char.isdigit():
                result += char
            elif char == '.' and not has_dot:
                has_dot = True
                result += char
            elif char in 'xXabcdefABCDEF' and result.startswith('0'):
                # Hexadecimal number
                result += char
            else:
                break
            i += 1
        
        self.current_column += len(result)
        return result, i
    
    def _extract_identifier(self, line: str, start: int) -> Tuple[str, int]:
        """Extract an identifier or keyword"""
        i = start
        result = ""
        
        # No legacy '%!' handling - legacy markers removed
        
        while i < len(line):
            char = line[i]
            if char.isalnum() or char in ['%', '_']:
                result += char
            else:
                break
            i += 1
        
        self.current_column += len(result)
        return result, i