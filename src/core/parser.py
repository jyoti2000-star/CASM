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
        if token.type == TokenType.AT_SYMBOL:
            return self._parse_var_declaration()
        elif token.type == TokenType.IDENTIFIER:
            # Look ahead to see if this is an assignment or assembly
            if self._is_assignment():
                return self._parse_assignment()
            else:
                return self._parse_unknown_as_assembly()
        elif token.type == TokenType.IF:
            return self._parse_if_statement()
        elif token.type == TokenType.WHILE:
            return self._parse_while_statement()
        elif token.type == TokenType.FOR:
            return self._parse_for_statement()
        elif token.type == TokenType.PRINT:
            return self._parse_print_statement()
        elif token.type == TokenType.SCAN:
            return self._parse_scan_statement()
        elif token.type == TokenType.C_CODE_BLOCK:
            return self._parse_c_code_block()
        elif token.type == TokenType.ASM_BLOCK:
            return self._parse_asm_block()
        elif token.type == TokenType.ASSEMBLY_LINE:
            return self._parse_assembly_line()
        else:
            # Unknown token, might be start of assembly line
            # Collect all tokens on this line to form complete assembly
            return self._parse_unknown_as_assembly()
    
    def _is_assignment(self) -> bool:
        """Check if the current identifier is followed by an assignment operator"""
        # Save current position
        saved_position = self.current
        
        # Skip the identifier
        if self._check(TokenType.IDENTIFIER):
            self._advance()
            
            # Check if next token is assignment operator
            is_assign = self._check(TokenType.ASSIGN)
            
            # Restore position
            self.current = saved_position
            return is_assign
        
        return False
    
    def _parse_var_declaration(self) -> ASTNode:
        """Parse variable declaration: @type name = value or @extern <header>"""
        self._consume(TokenType.AT_SYMBOL, "Expected '@'")
        
        # Check if this is an extern directive
        if self._check(TokenType.EXTERN):
            return self._parse_extern_directive()
        
        # Get the type
        if not self._check_types([TokenType.INT_TYPE, TokenType.STR_TYPE, TokenType.BOOL_TYPE, 
                                 TokenType.FLOAT_TYPE, TokenType.BUFFER_TYPE]):
            raise ParseError("Expected type after '@'", self._peek())
        
        type_token = self._advance()
        var_type = type_token.value
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expected variable name")
        name = name_token.value
        
        # Check for array/buffer size notation: name[size]
        size = None
        if self._check(TokenType.LEFT_BRACKET):
            self._advance()  # consume '['
            size_token = self._consume(TokenType.NUMBER, "Expected array size")
            size = int(size_token.value)
            self._consume(TokenType.RIGHT_BRACKET, "Expected ']'")
        
        # Expect = sign
        self._consume(TokenType.ASSIGN, "Expected '=' after variable name")
        
        # Rest of the line is the value
        value_parts = []
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            token = self._advance()
            if token.type != TokenType.EOF:
                value_parts.append(token.value)
        
        value = ' '.join(value_parts).strip()
        
        # Set default values based on type if no value provided
        if not value:
            if var_type == "int":
                value = "0"
            elif var_type in ["str", "string"]:
                value = '""'
            elif var_type == "bool":
                value = "false"
            elif var_type == "float":
                value = "0.0"
            elif var_type == "buffer":
                value = ""  # No value needed for buffers
        
        return VarDeclarationNode(name, value, var_type, size)
    
    def _parse_extern_directive(self) -> ExternDirectiveNode:
        """Parse extern directive: @extern <header>"""
        self._consume(TokenType.EXTERN, "Expected 'extern'")
        
        # Check for angle brackets or just a regular identifier/string
        header_name = ""
        
        if self._check(TokenType.LESS_THAN):
            self._advance()  # consume '<'
            # Collect everything until '>'
            header_parts = []
            while not self._check(TokenType.GREATER_THAN) and not self._is_at_end():
                token = self._advance()
                if token.type != TokenType.EOF:
                    header_parts.append(token.value)
            self._consume(TokenType.GREATER_THAN, "Expected '>'")
            header_name = ''.join(header_parts)
        elif self._check(TokenType.STRING):
            # String literal header
            string_token = self._advance()
            header_name = string_token.value.strip('"\'')
        else:
            # Regular identifier (like stdio.h)
            # Collect until newline as header name
            header_parts = []
            while not self._check(TokenType.NEWLINE) and not self._is_at_end():
                token = self._advance()
                if token.type != TokenType.EOF:
                    header_parts.append(token.value)
            header_name = ''.join(header_parts).strip()
        
        return ExternDirectiveNode(header_name)
    
    def _parse_assignment(self) -> AssignmentNode:
        """Parse variable assignment: variable = expression"""
        name_token = self._advance()  # consume identifier
        name = name_token.value
        
        # Expect = sign
        self._consume(TokenType.ASSIGN, "Expected '=' after variable name")
        
        # Rest of the line is the expression
        value_parts = []
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            token = self._advance()
            if token.type != TokenType.EOF:
                value_parts.append(token.value)
        
        value = ' '.join(value_parts).strip()
        
        return AssignmentNode(name, value)
    
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
        
        # Accept either a number or an identifier for the count
        if self._check(TokenType.NUMBER):
            count_token = self._advance()
            count = count_token.value
        elif self._check(TokenType.IDENTIFIER):
            count_token = self._advance()
            count = count_token.value
        else:
            raise ParseError("Expected number or variable name in range()", self._peek())
        
        self._consume(TokenType.RIGHT_PAREN, "Expected ')'")
        
        # Parse body
        body = []
        while not self._check(TokenType.ENDFOR) and not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
        
        self._consume(TokenType.ENDFOR, "Expected '%endfor'")
        
        return ForNode(variable, count, body)
    
    def _parse_print_statement(self) -> PrintlnNode:
        """Parse print statement: print message"""
        self._consume(TokenType.PRINT, "Expected 'print'")
        
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
    
    def _parse_scan_statement(self) -> ScanfNode:
        """Parse scan statement: scan variable"""
        self._consume(TokenType.SCAN, "Expected 'scan'")
        
        # Parse variable name (could be identifier or type name)
        if self._check(TokenType.IDENTIFIER):
            var_token = self._advance()
        elif self._check_types([TokenType.BUFFER_TYPE, TokenType.INT_TYPE, TokenType.STR_TYPE, 
                               TokenType.BOOL_TYPE, TokenType.FLOAT_TYPE]):
            var_token = self._advance()
        else:
            raise ParseError("Expected variable name", self._peek())
        
        variable = var_token.value
        
        # For scan, we'll use %s as default format string
        format_string = '"%s"'
        
        return ScanfNode(format_string, variable)
    
    def _parse_assembly_line(self) -> AssemblyNode:
        """Parse raw assembly line"""
        token = self._advance()
        return AssemblyNode(token.value)
    
    def _parse_unknown_as_assembly(self) -> AssemblyNode:
        """Parse unknown tokens as assembly line - collect entire line"""
        # Start with the current token
        assembly_parts = []
        
        # Collect all tokens until newline or EOF
        while not self._check(TokenType.NEWLINE) and not self._is_at_end():
            token = self._advance()
            if token.type == TokenType.EOF:
                break
            assembly_parts.append(token.value)
        
        # Join all parts to form complete assembly line
        assembly_code = ' '.join(assembly_parts)
        return AssemblyNode(assembly_code)
    
    def _parse_c_code_block(self) -> CCodeBlockNode:
        """Parse C code block: _c_ ... _endc_"""
        self._consume(TokenType.C_CODE_BLOCK, "Expected '_c_'")
        
        # Collect all code between _c_ and _endc_
        c_code_parts = []
        
        while not self._is_at_end():
            if self._check(TokenType.C_CODE_END):
                self._advance()  # consume _endc_
                break
            
            # Collect tokens as C code
            token = self._advance()
            if token.type == TokenType.NEWLINE:
                c_code_parts.append('\n')
            elif token.type != TokenType.EOF:
                c_code_parts.append(token.value)
        
        c_code = ' '.join(c_code_parts).strip()
        return CCodeBlockNode(c_code)
    
    def _parse_asm_block(self) -> AsmBlockNode:
        """Parse assembly block: _asm_ ... _endasm_"""
        self._consume(TokenType.ASM_BLOCK, "Expected '_asm_'")
        
        # Collect all code between _asm_ and _endasm_
        asm_code_parts = []
        
        while not self._is_at_end():
            if self._check(TokenType.ASM_END):
                self._advance()  # consume _endasm_
                break
            
            # Collect tokens as assembly code
            token = self._advance()
            if token.type == TokenType.NEWLINE:
                asm_code_parts.append('\n')
            elif token.type != TokenType.EOF:
                asm_code_parts.append(token.value)
        
        asm_code = ' '.join(asm_code_parts).strip()
        return AsmBlockNode(asm_code)
    
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
    
    def _check_types(self, token_types: List[TokenType]) -> bool:
        """Check if current token is one of the given types"""
        if self._is_at_end():
            return False
        return self._peek().type in token_types
    
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