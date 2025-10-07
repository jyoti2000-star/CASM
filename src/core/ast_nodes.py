#!/usr/bin/env python3

"""
AST Node definitions for CASM
Clean, focused abstract syntax tree nodes
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
from dataclasses import dataclass

class ASTNode(ABC):
    """Base class for all AST nodes"""
    
    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor for the visitor pattern"""
        pass

@dataclass
class ProgramNode(ASTNode):
    """Root program node containing all statements"""
    statements: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_program(self)

@dataclass
class VarDeclarationNode(ASTNode):
    """Variable declaration: %var name value"""
    name: str
    value: str
    
    def accept(self, visitor):
        return visitor.visit_var_declaration(self)

@dataclass
class IfNode(ASTNode):
    """If statement: %if condition ... %endif"""
    condition: str
    if_body: List[ASTNode]
    else_body: Optional[List[ASTNode]] = None
    
    def accept(self, visitor):
        return visitor.visit_if(self)

@dataclass
class WhileNode(ASTNode):
    """While loop: %while condition ... %endwhile"""
    condition: str
    body: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_while(self)

@dataclass
class ForNode(ASTNode):
    """For loop: %for var in range(count) ... %endfor"""
    variable: str
    count: str
    body: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_for(self)

@dataclass
class PrintlnNode(ASTNode):
    """Print statement: %println message"""
    message: str
    
    def accept(self, visitor):
        return visitor.visit_println(self)

@dataclass
class ScanfNode(ASTNode):
    """Input statement: %scanf format variable"""
    format_string: str
    variable: str
    
    def accept(self, visitor):
        return visitor.visit_scanf(self)

@dataclass
class AssemblyNode(ASTNode):
    """Raw assembly code line"""
    code: str
    
    def accept(self, visitor):
        return visitor.visit_assembly(self)

@dataclass
class CommentNode(ASTNode):
    """Comment line"""
    text: str
    
    def accept(self, visitor):
        return visitor.visit_comment(self)

@dataclass
class CCodeBlockNode(ASTNode):
    """Embedded C code block"""
    c_code: str
    
    def accept(self, visitor):
        return visitor.visit_c_code_block(self)

@dataclass
class ExternDirectiveNode(ASTNode):
    """External header directive"""
    header_name: str
    
    def accept(self, visitor):
        return visitor.visit_extern_directive(self)

# Visitor interface for AST traversal
class ASTVisitor(ABC):
    """Visitor interface for traversing AST"""
    
    @abstractmethod
    def visit_program(self, node: ProgramNode):
        pass
    
    @abstractmethod
    def visit_var_declaration(self, node: VarDeclarationNode):
        pass
    
    @abstractmethod
    def visit_if(self, node: IfNode):
        pass
    
    @abstractmethod
    def visit_while(self, node: WhileNode):
        pass
    
    @abstractmethod
    def visit_for(self, node: ForNode):
        pass
    
    @abstractmethod
    def visit_println(self, node: PrintlnNode):
        pass
    
    @abstractmethod
    def visit_scanf(self, node: ScanfNode):
        pass
    
    @abstractmethod
    def visit_assembly(self, node: AssemblyNode):
        pass
    
    @abstractmethod
    def visit_comment(self, node: CommentNode):
        pass
    
    @abstractmethod
    def visit_c_code_block(self, node: CCodeBlockNode):
        pass
    
    @abstractmethod
    def visit_extern_directive(self, node: ExternDirectiveNode):
        pass