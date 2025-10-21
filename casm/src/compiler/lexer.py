from typing import List, Tuple, Optional
from .tokens import TokenType, Token, SourceLocation

import re

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
            # allow letters, digits, underscore, percent and hyphen? keep conservative
            if char.isalnum() or char in ['_', '%']:
                result += char
                i += 1
            else:
                break
        
        self.current_column += len(result)
        return result, i
