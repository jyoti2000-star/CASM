from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any


class ASTNode(ABC):
    def __init__(self, location: Optional[object] = None):
        self.location = location
        self.type: Optional[object] = None  # Type annotation after semantic analysis
    
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
    return_type: object
    parameters: List[Tuple[str, object]]
    body: List[ASTNode]
    is_inline: bool = False
    
    def accept(self, visitor):
        return visitor.visit_function(self)

@dataclass
class StructNode(ASTNode):
    name: str
    members: List[Tuple[str, object]]
    
    def accept(self, visitor):
        return visitor.visit_struct(self)

@dataclass
class VarDeclarationNode(ASTNode):
    name: str
    var_type: object
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
    literal_type: object
    
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
    target_type: object
    
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
