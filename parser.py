#!/usr/bin/env python3

from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from lexer import Token, TokenType, Lexer

# Abstract Syntax Tree Node classes
class ASTNode(ABC):
    """Base class for all AST nodes"""
    pass

@dataclass
class ProgramNode(ASTNode):
    """Root node of the program"""
    statements: List[ASTNode]

@dataclass
class IfNode(ASTNode):
    """If statement node"""
    condition: str
    if_block: List[ASTNode]
    else_block: Optional[List[ASTNode]] = None

@dataclass
class WhileNode(ASTNode):
    """While loop node"""
    condition: str
    body: List[ASTNode]

@dataclass
class DoWhileNode(ASTNode):
    """Do-while loop node"""
    body: List[ASTNode]
    condition: str

@dataclass
class ForNode(ASTNode):
    """For loop node"""
    variable: str
    count_expr: str
    body: List[ASTNode]

@dataclass
class SwitchNode(ASTNode):
    """Switch statement node"""
    expression: str
    cases: List['CaseNode']
    default_case: Optional['DefaultNode'] = None

@dataclass
class CaseNode(ASTNode):
    """Case in switch statement"""
    value: str
    body: List[ASTNode]

@dataclass
class DefaultNode(ASTNode):
    """Default case in switch statement"""
    body: List[ASTNode]

@dataclass
class FunctionNode(ASTNode):
    """Function definition node"""
    name: str
    body: List[ASTNode]

@dataclass
class CallNode(ASTNode):
    """Function call node"""
    name: str

@dataclass
class ReturnNode(ASTNode):
    """Return statement node"""
    pass

@dataclass
class VarNode(ASTNode):
    """Variable declaration node"""
    name: str
    value: Optional[str] = None
    var_type: Optional[str] = None  # 'string', 'number', 'boolean', 'array'
    size: Optional[int] = None  # For arrays
    init_values: Optional[List[Union[int, float, str]]] = None  # For array initialization

@dataclass
class ConstNode(ASTNode):
    """Constant declaration node"""
    name: str
    value: str

@dataclass
class PrintNode(ASTNode):
    """Print statement node"""
    argument: str
    newline: bool = False

@dataclass
class InputNode(ASTNode):
    """Input statement node"""
    destination: str

@dataclass
class StdlibCallNode(ASTNode):
    """Standard library function call node"""
    function: str
    arguments: List[str]

@dataclass
class ExitNode(ASTNode):
    """Exit statement node"""
    code: str = "0"

@dataclass
class BreakNode(ASTNode):
    """Break statement node"""
    pass

@dataclass
class ContinueNode(ASTNode):
    """Continue statement node"""
    pass

@dataclass
class AssemblyNode(ASTNode):
    """Raw assembly instruction node"""
    instruction: str

@dataclass
class CommentNode(ASTNode):
    """Comment node"""
    text: str

@dataclass
class SectionNode(ASTNode):
    """Section directive node"""
    directive: str

class ParseError(Exception):
    """Exception raised for parsing errors"""
    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        if self.token:
            return f"Parse error at line {self.token.line}, column {self.token.column}: {self.message}"
        return f"Parse error: {self.message}"

class Parser:
    def __init__(self):
        self.tokens: List[Token] = []
        self.current = 0
    
    def parse(self, tokens: List[Token]) -> ProgramNode:
        """Parse tokens into an AST"""
        self.tokens = tokens
        self.current = 0
        
        statements = []
        while not self._is_at_end():
            if self._peek().type == TokenType.NEWLINE:
                self._advance()
                continue
            
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        
        return ProgramNode(statements)
    
    def parse_file(self, filename: str) -> ProgramNode:
        """Parse a file into an AST"""
        lexer = Lexer()
        tokens = lexer.tokenize_file(filename)
        return self.parse(tokens)
    
    def _parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement"""
        token = self._peek()
        
        if token.type == TokenType.COMMENT:
            return self._parse_comment()
        elif token.type == TokenType.SECTION:
            return self._parse_section()
        elif token.type == TokenType.IF:
            return self._parse_if()
        elif token.type == TokenType.WHILE:
            return self._parse_while()
        elif token.type == TokenType.DO:
            return self._parse_do_while()
        elif token.type == TokenType.FOR:
            return self._parse_for()
        elif token.type == TokenType.SWITCH:
            return self._parse_switch()
        elif token.type == TokenType.FUNCTION:
            return self._parse_function()
        elif token.type == TokenType.CALL:
            return self._parse_call()
        elif token.type == TokenType.RETURN:
            return self._parse_return()
        elif token.type == TokenType.VAR:
            return self._parse_var()
        elif token.type == TokenType.CONST:
            return self._parse_const()
        elif token.type == TokenType.PRINT:
            return self._parse_print(False)
        elif token.type == TokenType.PRINTLN:
            return self._parse_print(True)
        elif token.type == TokenType.INPUT:
            return self._parse_input()
        elif token.type in [TokenType.SCANF, TokenType.STRLEN, TokenType.STRCPY, 
                           TokenType.STRCMP, TokenType.MEMSET, TokenType.MEMCPY,
                           TokenType.ATOI, TokenType.ITOA]:
            return self._parse_stdlib_call()
        elif token.type == TokenType.EXIT:
            return self._parse_exit()
        elif token.type == TokenType.BREAK:
            return self._parse_break()
        elif token.type == TokenType.CONTINUE:
            return self._parse_continue()
        elif token.type == TokenType.ASSEMBLY:
            return self._parse_assembly()
        else:
            # Skip unknown tokens
            self._advance()
            return None
    
    def _parse_comment(self) -> CommentNode:
        """Parse a comment"""
        token = self._advance()
        return CommentNode(token.value)
    
    def _parse_section(self) -> SectionNode:
        """Parse a section directive"""
        token = self._advance()
        return SectionNode(token.value)
    
    def _parse_if(self) -> IfNode:
        """Parse an if statement"""
        self._advance()  # consume 'if'
        condition = self._parse_expression()
        
        if_block = self._parse_block([TokenType.ELSE, TokenType.ENDIF])
        
        else_block = None
        if self._check(TokenType.ELSE):
            self._advance()  # consume 'else'
            else_block = self._parse_block([TokenType.ENDIF])
        
        self._consume(TokenType.ENDIF, "Expected '%endif'")
        return IfNode(condition, if_block, else_block)
    
    def _parse_while(self) -> WhileNode:
        """Parse a while loop"""
        self._advance()  # consume 'while'
        condition = self._parse_expression()
        body = self._parse_block([TokenType.ENDWHILE])
        self._consume(TokenType.ENDWHILE, "Expected '%endwhile'")
        return WhileNode(condition, body)
    
    def _parse_do_while(self) -> DoWhileNode:
        """Parse a do-while loop"""
        self._advance()  # consume 'do'
        body = self._parse_block([TokenType.WHILE])
        self._consume(TokenType.WHILE, "Expected '%while'")
        condition = self._parse_expression()
        return DoWhileNode(body, condition)
    
    def _parse_for(self) -> ForNode:
        """Parse a for loop"""
        self._advance()  # consume 'for'
        variable = self._consume(TokenType.IDENTIFIER, "Expected variable name").value
        self._consume(TokenType.IN, "Expected 'in'")
        self._consume(TokenType.RANGE, "Expected 'range'")
        count_expr = self._parse_expression()
        body = self._parse_block([TokenType.ENDFOR])
        self._consume(TokenType.ENDFOR, "Expected '%endfor'")
        return ForNode(variable, count_expr, body)
    
    def _parse_switch(self) -> SwitchNode:
        """Parse a switch statement"""
        self._advance()  # consume 'switch'
        expression = self._parse_expression()
        
        cases = []
        default_case = None
        
        while not self._check(TokenType.ENDSWITCH) and not self._is_at_end():
            if self._check(TokenType.CASE):
                cases.append(self._parse_case())
            elif self._check(TokenType.DEFAULT):
                default_case = self._parse_default()
            else:
                self._advance()
        
        self._consume(TokenType.ENDSWITCH, "Expected '%endswitch'")
        return SwitchNode(expression, cases, default_case)
    
    def _parse_case(self) -> CaseNode:
        """Parse a case statement"""
        self._advance()  # consume 'case'
        value = self._parse_expression()
        body = self._parse_block([TokenType.CASE, TokenType.DEFAULT, TokenType.ENDSWITCH])
        return CaseNode(value, body)
    
    def _parse_default(self) -> DefaultNode:
        """Parse a default case"""
        self._advance()  # consume 'default'
        body = self._parse_block([TokenType.ENDSWITCH])
        return DefaultNode(body)
    
    def _parse_function(self) -> FunctionNode:
        """Parse a function definition"""
        self._advance()  # consume 'function'
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").value
        body = self._parse_block([TokenType.ENDFUNCTION])
        self._consume(TokenType.ENDFUNCTION, "Expected '%endfunction'")
        return FunctionNode(name, body)
    
    def _parse_call(self) -> CallNode:
        """Parse a function call"""
        self._advance()  # consume 'call'
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").value
        return CallNode(name)
    
    def _parse_return(self) -> ReturnNode:
        """Parse a return statement"""
        self._advance()  # consume 'return'
        return ReturnNode()
    
    def _parse_var(self) -> VarNode:
        """Parse a variable declaration"""
        self._advance()  # consume 'var'
        name = self._consume(TokenType.IDENTIFIER, "Expected variable name").value
        
        var_type = None
        size = None
        value = None
        init_values = None
        
        # Check if it's an array declaration: var buffer[125]
        if self._check(TokenType.LBRACKET):
            self._advance()  # consume '['
            size_token = self._consume(TokenType.NUMBER, "Expected array size")
            size = int(size_token.value)
            self._consume(TokenType.RBRACKET, "Expected ']'")
            var_type = "array"
            
            # Check for array initialization: var array[10] {1, 2, 3, ...}
            if self._check(TokenType.LBRACE):
                self._advance()  # consume '{'
                init_values = []
                
                while not self._check(TokenType.RBRACE) and not self._is_at_end():
                    if self._check(TokenType.NUMBER):
                        val = self._peek().value
                        # Try to parse as int first, then float
                        try:
                            parsed_val = int(val)
                        except ValueError:
                            parsed_val = float(val)
                        init_values.append(parsed_val)
                        self._advance()
                    elif self._check(TokenType.STRING):
                        init_values.append(self._peek().value)
                        self._advance()
                    else:
                        break
                    
                    # Skip commas
                    if self._check(TokenType.COMMA):
                        self._advance()
                
                self._consume(TokenType.RBRACE, "Expected '}' after array initialization")
        
        # Check if there's a value assignment for non-arrays
        elif not self._is_at_end() and not self._check(TokenType.NEWLINE):
            # Parse the value to determine type
            value_expr = self._parse_expression()
            value = value_expr
            
            # Determine variable type based on value
            if value.startswith('"') and value.endswith('"'):
                var_type = "string"
            elif value.lower() in ['true', 'false']:
                var_type = "boolean"
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                var_type = "number"
            else:
                var_type = "identifier"  # Variable reference
        
        return VarNode(name, value, var_type, size, init_values)
    
    def _parse_const(self) -> ConstNode:
        """Parse a constant declaration"""
        self._advance()  # consume 'const'
        name = self._consume(TokenType.IDENTIFIER, "Expected constant name").value
        
        # Make the '=' optional - support both "%const NAME = VALUE" and "%const NAME VALUE"
        if not self._is_at_end() and self._peek().type == TokenType.ASSIGN:
            self._advance()  # consume '=' if present
        
        value = self._parse_expression()
        return ConstNode(name, value)
    
    def _parse_print(self, newline: bool) -> PrintNode:
        """Parse a print statement"""
        self._advance()  # consume 'print' or 'println'
        argument = self._parse_expression()
        return PrintNode(argument, newline)
    
    def _parse_input(self) -> InputNode:
        """Parse an input statement"""
        self._advance()  # consume 'input'
        destination = self._parse_expression()
        return InputNode(destination)
    
    def _parse_stdlib_call(self) -> StdlibCallNode:
        """Parse a standard library function call"""
        token = self._advance()
        function = token.value[1:]  # Remove '%' prefix
        
        arguments = []
        if not self._check(TokenType.NEWLINE) and not self._is_at_end():
            arguments.append(self._parse_expression())
            
            while self._match_value(','):
                arguments.append(self._parse_expression())
        
        return StdlibCallNode(function, arguments)
    
    def _parse_exit(self) -> ExitNode:
        """Parse an exit statement"""
        self._advance()  # consume 'exit'
        code = "0"
        if not self._check(TokenType.NEWLINE) and not self._is_at_end():
            code = self._parse_expression()
        return ExitNode(code)
    
    def _parse_break(self) -> BreakNode:
        """Parse a break statement"""
        self._advance()  # consume 'break'
        return BreakNode()
    
    def _parse_continue(self) -> ContinueNode:
        """Parse a continue statement"""
        self._advance()  # consume 'continue'
        return ContinueNode()
    
    def _parse_assembly(self) -> AssemblyNode:
        """Parse a raw assembly instruction"""
        token = self._advance()
        return AssemblyNode(token.value)
    
    def _parse_block(self, end_tokens: List[TokenType]) -> List[ASTNode]:
        """Parse a block of statements until one of the end tokens"""
        statements = []
        
        while not self._is_at_end() and not any(self._check(token_type) for token_type in end_tokens):
            if self._check(TokenType.NEWLINE):
                self._advance()
                continue
            
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        
        return statements
    
    def _parse_expression(self) -> str:
        """Parse an expression (simplified - returns as string)"""
        parts = []
        
        while (not self._check(TokenType.NEWLINE) and 
               not self._is_at_end() and 
               self._peek().type not in [TokenType.IF, TokenType.ELSE, TokenType.ENDIF,
                                       TokenType.WHILE, TokenType.ENDWHILE, TokenType.FOR,
                                       TokenType.ENDFOR, TokenType.DO, TokenType.SWITCH,
                                       TokenType.CASE, TokenType.DEFAULT, TokenType.ENDSWITCH,
                                       TokenType.FUNCTION, TokenType.ENDFUNCTION]):
            token = self._advance()
            parts.append(token.value)
        
        return ' '.join(parts).strip()
    
    def _match_value(self, value: str) -> bool:
        """Check if current token value matches and advance if so"""
        if not self._is_at_end() and self._peek().value == value:
            self._advance()
            return True
        return False
    
    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type"""
        if self._is_at_end():
            return False
        return self._peek().type == token_type
    
    def _advance(self) -> Token:
        """Consume and return current token"""
        if not self._is_at_end():
            self.current += 1
        return self._previous()
    
    def _is_at_end(self) -> bool:
        """Check if we're at end of tokens"""
        return self.current >= len(self.tokens)
    
    def _peek(self) -> Token:
        """Return current token without advancing"""
        if self._is_at_end():
            return Token(TokenType.NEWLINE, '', 0, 0)
        return self.tokens[self.current]
    
    def _previous(self) -> Token:
        """Return previous token"""
        return self.tokens[self.current - 1]
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error"""
        if self._check(token_type):
            return self._advance()
        
        current_token = self._peek() if not self._is_at_end() else None
        raise ParseError(message, current_token)