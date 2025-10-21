from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple, Dict

class ASTNode(ABC):
    def __init__(self, location: Optional[object] = None):
        self.location = location
        self.parent: Optional[ASTNode] = None
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

# Declarations and many node classes follow; to keep this split manageable, importers may use this
# module for AST node definitions. For brevity we include the commonly used nodes.

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

# ... other node classes (IfNode, WhileNode, etc.) omitted for brevity; full definitions exist in original core.py
