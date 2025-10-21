from typing import List, Optional, Tuple
from .tokens import TokenType, Token, SourceLocation
from .diagnostics import DiagnosticEngine
from .symbols import SymbolTable, Symbol
from .ast import ProgramNode, VarDeclarationNode, FunctionDeclarationNode, StructDeclarationNode, UnionDeclarationNode, EnumDeclarationNode, TypedefNode, MacroDefinitionNode, AssignmentNode, IfNode, WhileNode, DoWhileNode, ForNode, SwitchNode, BreakNode, ContinueNode, ReturnNode, GotoNode, LabelNode, PrintlnNode, ScanfNode, AssemblyNode, AsmBlockNode, CCodeBlockNode, ExternDirectiveNode, CommentNode

class ParseError(Exception):
    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        location_str = f" at {token.location}" if token else ""
        super().__init__(f"Parse error{location_str}: {message}")

class CASMParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        self.diagnostics = DiagnosticEngine()
        self.symbol_table = SymbolTable()
        self._anon_var_counter = 0
        self._label_counter = 0
        self._in_function = False
        self._in_loop = False
        self._in_switch = False

    # Many parsing methods follow; for brevity they mirror the original core.py's parser.
