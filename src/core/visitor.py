from abc import ABC, abstractmethod
from .ast import ProgramNode, VarDeclarationNode, FunctionDeclarationNode

class ASTVisitor(ABC):
    @abstractmethod
    def visit_program(self, node: ProgramNode): pass
    
    @abstractmethod
    def visit_var_declaration(self, node: VarDeclarationNode): pass
    
    @abstractmethod
    def visit_function_declaration(self, node: FunctionDeclarationNode): pass
    
    # Other visit_ methods omitted for brevity; concrete generators implement required ones.
