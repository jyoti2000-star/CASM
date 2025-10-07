#!/usr/bin/env python3

"""
Parser for CASM language
Clean, focused parser for essential language constructs
"""

from typing import List, Optional
from .tokens import Token, TokenType, LexerError
from .ast_nodes import *
from ..utils.colors import print_info, print_error

class ParseError(Exception):
    """Exception for parsing errors"""
    
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"Parse error at line {token.line}, column {token.column}: {message}")

class CASMParser:
    """Parser for CASM language"""
    
    def __init__(self):
        self.tokens = []
        self.current = 0
    
    def parse(self, tokens: List[Token]) -> ProgramNode:
        """Parse tokens into AST"""
        self.tokens = tokens
        self.current = 0
        
        statements = []
        
        while not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        
        return ProgramNode(statements)
    
    def _parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement"""
        if self._is_at_end():
            return None
        
        token = self._peek()
        
        # Skip newlines and comments
        if token.type in [TokenType.NEWLINE, TokenType.EOF]:
            self._advance()
            return None
        
        if token.type == TokenType.COMMENT:
            self._advance()
            return CommentNode(token.value)
        
        # Parse language constructs
        if token.type == TokenType.VAR:
            return self._parse_var_declaration()
        elif token.type == TokenType.IF:
            return self._parse_if_statement()
        elif token.type == TokenType.WHILE:
            return self._parse_while_statement()
        elif token.type == TokenType.FOR:
            return self._parse_for_statement()
        elif token.type == TokenType.PRINTLN:
            return self._parse_println_statement()
        elif token.type == TokenType.SCANF:
            return self._parse_scanf_statement()
        elif token.type == TokenType.C_CODE_BLOCK:
            return self._parse_c_code_block()
        elif token.type == TokenType.EXTERN:
            return self._parse_extern_directive()
        elif token.type == TokenType.ASSEMBLY_LINE:
            return self._parse_assembly_line()
        else:
            # Unknown token, treat as assembly
            self._advance()
            return AssemblyNode(token.value)
    
    def _parse_var_declaration(self) -> VarDeclarationNode:
        """Parse variable declaration: %var name value"""
        self._consume(TokenType.VAR, "Expected '%var'")
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expected variable name")
        name = name_token.value
        
        # Rest of the line is the value
        value_parts = []
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            token = self._advance()
            if token.type != TokenType.EOF:
                value_parts.append(token.value)
        
        value = ' '.join(value_parts).strip()
        if not value:
            value = "0"  # Default value
        
        return VarDeclarationNode(name, value)
    
    def _parse_if_statement(self) -> IfNode:
        """Parse if statement: %if condition ... %else ... %endif"""
        self._consume(TokenType.IF, "Expected '%if'")
        
        # Parse condition (rest of line)
        condition_parts = []
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            token = self._advance()
            if token.type != TokenType.EOF:
                condition_parts.append(token.value)
        
        condition = ' '.join(condition_parts).strip()
        
        # Parse if body
        if_body = []
        while not self._check(TokenType.ELSE) and not self._check(TokenType.ENDIF) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                if_body.append(stmt)
        
        # Parse optional else
        else_body = None
        if self._check(TokenType.ELSE):
            self._advance()  # consume %else
            
            else_body = []
            while not self._check(TokenType.ENDIF) and not self._is_at_end():
                stmt = self._parse_statement()
                if stmt:
                    else_body.append(stmt)
        
        self._consume(TokenType.ENDIF, "Expected '%endif'")
        
        return IfNode(condition, if_body, else_body)
    
    def _parse_while_statement(self) -> WhileNode:
        """Parse while statement: %while condition ... %endwhile"""
        self._consume(TokenType.WHILE, "Expected '%while'")
        
        # Parse condition (rest of line)
        condition_parts = []
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            token = self._advance()
            if token.type != TokenType.EOF:
                condition_parts.append(token.value)
        
        condition = ' '.join(condition_parts).strip()
        
        # Parse body
        body = []
        while not self._check(TokenType.ENDWHILE) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
        
        self._consume(TokenType.ENDWHILE, "Expected '%endwhile'")
        
        return WhileNode(condition, body)
    
    def _parse_for_statement(self) -> ForNode:
        """Parse for statement: %for var in range(count) ... %endfor"""
        self._consume(TokenType.FOR, "Expected '%for'")
        
        var_token = self._consume(TokenType.IDENTIFIER, "Expected variable name")
        variable = var_token.value
        
        self._consume(TokenType.IN, "Expected 'in'")
        self._consume(TokenType.RANGE, "Expected 'range'")
        self._consume(TokenType.LEFT_PAREN, "Expected '('")
        
        count_token = self._consume(TokenType.NUMBER, "Expected number")
        count = count_token.value
        
        self._consume(TokenType.RIGHT_PAREN, "Expected ')'")
        
        # Parse body
        body = []
        while not self._check(TokenType.ENDFOR) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
        
        self._consume(TokenType.ENDFOR, "Expected '%endfor'")
        
        return ForNode(variable, count, body)
    
    def _parse_println_statement(self) -> PrintlnNode:
        """Parse println statement: %println message"""
        self._consume(TokenType.PRINTLN, "Expected '%println'")
        
        # Parse message (can be string or expression)
        if self._check(TokenType.STRING):
            message_token = self._advance()
            message = message_token.value
        else:
            # Collect rest of line as message
            message_parts = []
            while not self._check(TokenType.NEWLINE) and not self._is_at_end():
                token = self._advance()
                if token.type != TokenType.EOF:
                    message_parts.append(token.value)
            message = ' '.join(message_parts).strip()
        
        return PrintlnNode(message)
    
    def _parse_scanf_statement(self) -> ScanfNode:
        """Parse scanf statement: %scanf format variable"""
        self._consume(TokenType.SCANF, "Expected '%scanf'")
        
        format_token = self._consume(TokenType.STRING, "Expected format string")
        format_string = format_token.value
        
        var_token = self._consume(TokenType.IDENTIFIER, "Expected variable name")
        variable = var_token.value
        
        return ScanfNode(format_string, variable)
    
    def _parse_assembly_line(self) -> AssemblyNode:
        """Parse raw assembly line"""
        token = self._advance()
        return AssemblyNode(token.value)
    
    def _parse_c_code_block(self) -> CCodeBlockNode:
        """Parse C code block starting with %!"""
        self._consume(TokenType.C_CODE_BLOCK, "Expected '%!'")
        
        # For line-by-line C code, get the rest of the line
        c_code = ""
        if not self._is_at_end() and self._peek().type == TokenType.ASSEMBLY_LINE:
            c_code = self._advance().value
        
        return CCodeBlockNode(c_code)
    
    def _parse_extern_directive(self) -> 'ExternDirectiveNode':
        """Parse extern directive: %extern <filename>"""
        self._consume(TokenType.EXTERN, "Expected '%extern'")
        
        # Collect all remaining tokens on this line to form the header name
        header_parts = []
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            token = self._advance()
            if token.type != TokenType.EOF:
                header_parts.append(token.value)
        
        header_name = ''.join(header_parts).strip()
        
        # Clean up the header name (remove quotes if present)
        if header_name.startswith('"') and header_name.endswith('"'):
            header_name = header_name[1:-1]
        elif header_name.startswith('<') and header_name.endswith('>'):
            header_name = header_name[1:-1]
        
        return ExternDirectiveNode(header_name)
    
    # Utility methods
    def _advance(self) -> Token:
        """Consume and return current token"""
        if not self._is_at_end():
            self.current += 1
        return self._previous()
    
    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type"""
        if self._is_at_end():
            return False
        return self._peek().type == token_type
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error"""
        if self._check(token_type):
            return self._advance()
        
        current_token = self._peek()
        raise ParseError(f"{message}, got {current_token.type.value}", current_token)
    
    def _is_at_end(self) -> bool:
        """Check if we're at end of tokens"""
        return self.current >= len(self.tokens) or self._peek().type == TokenType.EOF
    
    def _peek(self) -> Token:
        """Return current token without advancing"""
        if self.current >= len(self.tokens):
            # Return EOF token
            return Token(TokenType.EOF, '', 0, 0, '')
        return self.tokens[self.current]
    
    def _previous(self) -> Token:
        """Return previous token"""
        return self.tokens[self.current - 1]