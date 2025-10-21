from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple, Dict

# Prefer reusing the compiler AST implementation when available. Many parts of
# the `src.core` pipeline expect specific class names; the compiler AST uses
# slightly different names in some places, so we import and alias where
# appropriate. If importing fails we fall back to providing minimal local
# definitions.
try:
    # Import everything from the compiler AST and then create aliases for the
    # names the core modules expect.
    from src.compiler.ast import *  # noqa: F401,F403

    # Aliases: core imports expect names like FunctionDeclarationNode,
    # StructDeclarationNode, PrintlnNode, etc. Map them to the compiler AST
    # equivalents when present.
    if 'FunctionNode' in globals() and 'FunctionDeclarationNode' not in globals():
        FunctionDeclarationNode = FunctionNode  # type: ignore

    if 'StructNode' in globals() and 'StructDeclarationNode' not in globals():
        StructDeclarationNode = StructNode  # type: ignore
        UnionDeclarationNode = StructNode  # type: ignore

    if 'PrintNode' in globals() and 'PrintlnNode' not in globals():
        PrintlnNode = PrintNode  # type: ignore

    # AsmBlock/Assembly naming
    if 'AssemblyNode' in globals() and 'AsmBlockNode' not in globals():
        AsmBlockNode = AssemblyNode  # type: ignore

    # Provide small aliases for commonly referenced names that may be missing
    for alias, target in (
        ('TypedefNode', 'OptimizationHintNode'),
        ('MacroDefinitionNode', 'OptimizationHintNode'),
        ('ExternDirectiveNode', 'CommentNode'),
    ):
        if alias not in globals() and target in globals():
            globals()[alias] = globals()[target]

except Exception:
    # Minimal local definitions if import fails. These are intentionally small
    # and sufficient for the parser/codegen import to succeed; full
    # implementations may exist elsewhere.

    class ASTNode(ABC):
        def __init__(self, location: Optional[object] = None):
            self.location = location
            self.parent: Optional['ASTNode'] = None
            self.attributes: Dict[str, Any] = {}

        @abstractmethod
        def accept(self, visitor):
            pass

        def get_children(self) -> List['ASTNode']:
            return []

        def set_parent(self, parent: 'ASTNode'):
            self.parent = parent
            for child in self.get_children():
                if isinstance(child, ASTNode):
                    child.set_parent(self)

    @dataclass
    class ProgramNode(ASTNode):
        statements: List[ASTNode] = field(default_factory=list)

        def accept(self, visitor):
            return visitor.visit_program(self)

        def get_children(self):
            return self.statements

    @dataclass
    class VarDeclarationNode(ASTNode):
        name: str
        var_type: str
        value: Optional[str] = None
        size: Optional[int] = None
        is_const: bool = False
        is_static: bool = False
        is_extern: bool = False
        is_volatile: bool = False
        is_register: bool = False
        storage_class: Optional[str] = None

        def accept(self, visitor):
            return visitor.visit_var_declaration(self)

    @dataclass
    class FunctionDeclarationNode(ASTNode):
        name: str
        return_type: str
        parameters: List[Tuple[str, str]] = field(default_factory=list)
        body: Optional[List[ASTNode]] = None
        is_inline: bool = False
        is_static: bool = False
        is_extern: bool = False
        calling_convention: Optional[str] = None

        def accept(self, visitor):
            return visitor.visit_function_declaration(self)

        def get_children(self):
            return self.body if self.body else []

    # Lightweight stubs for other commonly-used node types so imports succeed
    @dataclass
    class StructDeclarationNode(ASTNode):
        name: str
        members: List[Tuple[str, str]] = field(default_factory=list)

        def accept(self, visitor):
            return visitor.visit_struct(self)

    @dataclass
    class UnionDeclarationNode(StructDeclarationNode):
        pass

    @dataclass
    class EnumDeclarationNode(ASTNode):
        name: str
        members: List[Tuple[str, int]] = field(default_factory=list)

        def accept(self, visitor):
            return visitor.visit_struct(self)

    @dataclass
    class TypedefNode(ASTNode):
        name: str
        target: str

        def accept(self, visitor):
            return visitor.visit_struct(self)

    @dataclass
    class MacroDefinitionNode(ASTNode):
        name: str
        body: str

        def accept(self, visitor):
            return visitor.visit_struct(self)

    @dataclass
    class AssignmentNode(ASTNode):
        target: Any
        value: Any

        def accept(self, visitor):
            return visitor.visit_assignment(self)

    @dataclass
    class IfNode(ASTNode):
        condition: Any
        if_body: List[ASTNode] = field(default_factory=list)
        elif_branches: List[Tuple[Any, List[ASTNode]]] = field(default_factory=list)
        else_body: Optional[List[ASTNode]] = None

        def accept(self, visitor):
            return visitor.visit_if(self)

    @dataclass
    class WhileNode(ASTNode):
        condition: Any
        body: List[ASTNode] = field(default_factory=list)

        def accept(self, visitor):
            return visitor.visit_while(self)

    @dataclass
    class DoWhileNode(ASTNode):
        body: List[ASTNode] = field(default_factory=list)
        condition: Any = None

        def accept(self, visitor):
            return visitor.visit_do_while(self)

    @dataclass
    class ForNode(ASTNode):
        init: Optional[ASTNode] = None
        condition: Optional[Any] = None
        increment: Optional[ASTNode] = None
        body: List[ASTNode] = field(default_factory=list)

        def accept(self, visitor):
            return visitor.visit_for(self)

    @dataclass
    class SwitchNode(ASTNode):
        expression: Any
        cases: List[Tuple[Optional[Any], List[ASTNode]]] = field(default_factory=list)

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
        value: Optional[Any] = None

        def accept(self, visitor):
            return visitor.visit_return(self)

    @dataclass
    class GotoNode(ASTNode):
        label: str

        def accept(self, visitor):
            return visitor.visit_goto(self)

    @dataclass
    class LabelNode(ASTNode):
        name: str

        def accept(self, visitor):
            return visitor.visit_label(self)

    @dataclass
    class PrintlnNode(ASTNode):
        expressions: List[Any] = field(default_factory=list)

        def accept(self, visitor):
            return visitor.visit_println(self)

    @dataclass
    class LiteralNode(ASTNode):
        value: Any

        def accept(self, visitor):
            return visitor.visit_literal(self)

    @dataclass
    class IdentifierNode(ASTNode):
        name: str

        def accept(self, visitor):
            return visitor.visit_identifier(self)

    @dataclass
    class BinaryOpNode(ASTNode):
        left: Any
        operator: str
        right: Any

        def accept(self, visitor):
            return visitor.visit_binary_op(self)

    @dataclass
    class FunctionCallNode(ASTNode):
        function: Any
        arguments: List[Any] = field(default_factory=list)

        def accept(self, visitor):
            return visitor.visit_function_call(self)

    @dataclass
    class ScanfNode(ASTNode):
        format_expr: Any
        targets: List[Any] = field(default_factory=list)

        def accept(self, visitor):
            return visitor.visit_scanf(self)

    @dataclass
    class AssemblyNode(ASTNode):
        code: str

        def accept(self, visitor):
            return visitor.visit_assembly(self)

    AsmBlockNode = AssemblyNode

    @dataclass
    class CCodeBlockNode(ASTNode):
        c_code: str

        def accept(self, visitor):
            return visitor.visit_c_code_block(self)

    @dataclass
    class ExternDirectiveNode(ASTNode):
        name: str

        def accept(self, visitor):
            return visitor.visit_comment(self)

    @dataclass
    class CommentNode(ASTNode):
        text: str

        def accept(self, visitor):
            return visitor.visit_comment(self)

    @dataclass
    class OptimizationHintNode(ASTNode):
        hint_type: str
        target: Optional[ASTNode] = None
        parameters: Dict[str, Any] = field(default_factory=dict)

        def accept(self, visitor):
            return visitor.visit_optimization_hint(self)
