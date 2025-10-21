from abc import ABC, abstractmethod
from typing import List
from .ast import ProgramNode, ASTNode, BinaryOpNode, UnaryOpNode, LiteralNode, ReturnNode, BreakNode, ContinueNode


class OptimizationPass(ABC):
    """Base class for optimization passes"""
    
    @abstractmethod
    def optimize(self, ast: ProgramNode) -> ProgramNode:
        pass

class ConstantFoldingPass(OptimizationPass):
    """Fold constant expressions at compile time"""
    
    def optimize(self, ast: ProgramNode) -> ProgramNode:
        return self._visit_program(ast)
    
    def _visit_program(self, node: ProgramNode) -> ProgramNode:
        node.statements = [self._visit_node(stmt) for stmt in node.statements]
        return node
    
    def _visit_node(self, node: ASTNode) -> ASTNode:
        if isinstance(node, BinaryOpNode):
            return self._fold_binary_op(node)
        elif isinstance(node, UnaryOpNode):
            return self._fold_unary_op(node)
        # Add more node types as needed
        return node
    
    def _fold_binary_op(self, node: BinaryOpNode) -> ASTNode:
        left = self._visit_node(node.left)
        right = self._visit_node(node.right)
        
        if isinstance(left, LiteralNode) and isinstance(right, LiteralNode):
            try:
                if node.operator == '+':
                    result = left.value + right.value
                elif node.operator == '-':
                    result = left.value - right.value
                elif node.operator == '*':
                    result = left.value * right.value
                elif node.operator == '/':
                    if right.value != 0:
                        result = left.value / right.value
                    else:
                        return node
                elif node.operator == '%':
                    if right.value != 0:
                        result = left.value % right.value
                    else:
                        return node
                else:
                    return node
                
                return LiteralNode(
                    value=result,
                    literal_type=left.literal_type,
                    location=node.location
                )
            except Exception:
                return node
        
        node.left = left
        node.right = right
        return node
    
    def _fold_unary_op(self, node: UnaryOpNode) -> ASTNode:
        operand = self._visit_node(node.operand)
        
        if isinstance(operand, LiteralNode):
            try:
                if node.operator == '-':
                    result = -operand.value
                elif node.operator == '+':
                    result = +operand.value
                elif node.operator == '!':
                    result = not operand.value
                elif node.operator == '~':
                    result = ~operand.value
                else:
                    return node
                
                return LiteralNode(
                    value=result,
                    literal_type=operand.literal_type,
                    location=node.location
                )
            except Exception:
                return node
        
        node.operand = operand
        return node

class DeadCodeEliminationPass(OptimizationPass):
    """Remove unreachable code"""
    
    def optimize(self, ast: ProgramNode) -> ProgramNode:
        ast.statements = self._eliminate_dead_code(ast.statements)
        return ast
    
    def _eliminate_dead_code(self, statements: List[ASTNode]) -> List[ASTNode]:
        result = []
        reachable = True
        
        for stmt in statements:
            if not reachable:
                break
            
            result.append(stmt)
            
            if isinstance(stmt, ReturnNode):
                reachable = False
            elif isinstance(stmt, BreakNode) or isinstance(stmt, ContinueNode):
                reachable = False
        
        return result

class Optimizer:
    """Main optimizer that runs multiple passes"""
    
    def __init__(self, optimization_level: int = 2):
        self.optimization_level = optimization_level
        self.passes: List[OptimizationPass] = []
        
        if optimization_level >= 1:
            self.passes.append(ConstantFoldingPass())
        
        if optimization_level >= 2:
            self.passes.append(DeadCodeEliminationPass())
    
    def optimize(self, ast: ProgramNode) -> ProgramNode:
        """Run all optimization passes"""
        for pass_obj in self.passes:
            ast = pass_obj.optimize(ast)
        return ast
