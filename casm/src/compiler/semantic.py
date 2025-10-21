from abc import ABC
from typing import Optional, List, Any
from .symbols import Symbol, SymbolTable
from .types import TypeKind, Type
from .ast import *


class SemanticError(Exception):
    """Semantic analysis error"""
    def __init__(self, message: str, location: Optional[object] = None):
        self.message = message
        self.location = location
        super().__init__(f"Semantic error at {location}: {message}" if location else message)


class SemanticAnalyzer(ABC):
    """Performs semantic analysis and type checking"""
    
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[SemanticError] = []
        self.warnings: List[str] = []
        self.current_function: Optional[Symbol] = None
        self.loop_depth = 0
        
    def analyze(self, ast: ProgramNode) -> bool:
        """Analyze AST and return True if no errors"""
        try:
            self._analyze_program(ast)
            return len(self.errors) == 0
        except SemanticError as e:
            self.errors.append(e)
            return False
    
    def _analyze_program(self, node: ProgramNode):
        """Analyze program node"""
        for stmt in node.statements:
            self._analyze_node(stmt)
    
    def _analyze_node(self, node: ASTNode):
        """Dispatch to appropriate analyzer"""
        if isinstance(node, VarDeclarationNode):
            self._analyze_var_declaration(node)
        elif isinstance(node, FunctionNode):
            self._analyze_function(node)
        elif isinstance(node, AssignmentNode):
            self._analyze_assignment(node)
        elif isinstance(node, IfNode):
            self._analyze_if(node)
        elif isinstance(node, WhileNode):
            self._analyze_while(node)
        elif isinstance(node, ForNode):
            self._analyze_for(node)
        elif isinstance(node, ReturnNode):
            self._analyze_return(node)
        elif isinstance(node, ExpressionNode):
            self._analyze_expression(node)
    
    def _analyze_var_declaration(self, node: VarDeclarationNode):
        """Analyze variable declaration"""
        # Check if already defined in current scope
        existing = self.symbol_table.lookup_current_scope(node.name)
        if existing:
            self.errors.append(SemanticError(
                f"Variable '{node.name}' already defined in current scope",
                node.location
            ))
            return
        
        # Check initial value type if present
        if node.initial_value:
            expr_type = self._analyze_expression(node.initial_value)
            if expr_type and not node.var_type.is_compatible_with(expr_type):
                self.errors.append(SemanticError(
                    f"Type mismatch: cannot assign {expr_type} to {node.var_type}",
                    node.location
                ))
        
        # Add to symbol table
        symbol = Symbol(
            name=node.name,
            type=node.var_type,
            scope_level=self.symbol_table.current_scope_level,
            location=node.location,
            is_extern=node.is_extern,
            is_static=node.is_static
        )
        self.symbol_table.define(symbol)
        node.type = node.var_type
    
    def _analyze_function(self, node: FunctionNode):
        """Analyze function definition"""
        # Create function type
        param_types = [param_type for _, param_type in node.parameters]
        func_type = Type(
            kind=TypeKind.FUNCTION,
            name=node.name,
            param_types=param_types,
            return_type=node.return_type
        )
        
        # Add function to symbol table
        symbol = Symbol(
            name=node.name,
            type=func_type,
            scope_level=0,  # Functions are always global
            location=node.location,
            is_function=True
        )
        
        if not self.symbol_table.define(symbol):
            self.errors.append(SemanticError(
                f"Function '{node.name}' already defined",
                node.location
            ))
            return
        
        # Enter function scope
        old_function = self.current_function
        self.current_function = symbol
        self.symbol_table.enter_scope()
        
        # Add parameters to scope
        for param_name, param_type in node.parameters:
            param_symbol = Symbol(
                name=param_name,
                type=param_type,
                scope_level=self.symbol_table.current_scope_level,
                is_parameter=True
            )
            self.symbol_table.define(param_symbol)
        
        # Analyze function body
        for stmt in node.body:
            self._analyze_node(stmt)
        
        # Exit function scope
        self.symbol_table.exit_scope()
        self.current_function = old_function
    
    def _analyze_assignment(self, node: AssignmentNode):
        """Analyze assignment"""
        target_type = self._analyze_expression(node.target)
        value_type = self._analyze_expression(node.value)
        
        if target_type and value_type:
            if not target_type.is_compatible_with(value_type):
                self.errors.append(SemanticError(
                    f"Type mismatch: cannot assign {value_type} to {target_type}",
                    node.location
                ))
    
    def _analyze_if(self, node: IfNode):
        """Analyze if statement"""
        cond_type = self._analyze_expression(node.condition)
        
        if cond_type and cond_type.name != 'bool':
            self.warnings.append(f"Condition should be boolean, got {cond_type}")
        
        self.symbol_table.enter_scope()
        for stmt in node.if_body:
            self._analyze_node(stmt)
        self.symbol_table.exit_scope()
        
        for elif_cond, elif_body in node.elif_branches:
            self._analyze_expression(elif_cond)
            self.symbol_table.enter_scope()
            for stmt in elif_body:
                self._analyze_node(stmt)
            self.symbol_table.exit_scope()
        
        if node.else_body:
            self.symbol_table.enter_scope()
            for stmt in node.else_body:
                self._analyze_node(stmt)
            self.symbol_table.exit_scope()
    
    def _analyze_while(self, node: WhileNode):
        """Analyze while loop"""
        self._analyze_expression(node.condition)
        self.loop_depth += 1
        self.symbol_table.enter_scope()
        
        for stmt in node.body:
            self._analyze_node(stmt)
        
        self.symbol_table.exit_scope()
        self.loop_depth -= 1
    
    def _analyze_for(self, node: ForNode):
        """Analyze for loop"""
        self.symbol_table.enter_scope()
        
        if node.init:
            self._analyze_node(node.init)
        if node.condition:
            self._analyze_expression(node.condition)
        if node.increment:
            self._analyze_node(node.increment)
        
        self.loop_depth += 1
        for stmt in node.body:
            self._analyze_node(stmt)
        self.loop_depth -= 1
        
        self.symbol_table.exit_scope()
    
    def _analyze_return(self, node: ReturnNode):
        """Analyze return statement"""
        if not self.current_function:
            self.errors.append(SemanticError(
                "Return statement outside function",
                node.location
            ))
            return
        
        return_type = self.current_function.type.return_type
        
        if node.value:
            value_type = self._analyze_expression(node.value)
            if value_type and not return_type.is_compatible_with(value_type):
                self.errors.append(SemanticError(
                    f"Return type mismatch: expected {return_type}, got {value_type}",
                    node.location
                ))
        elif return_type.kind != TypeKind.VOID:
            self.errors.append(SemanticError(
                f"Function must return a value of type {return_type}",
                node.location
            ))
    
    def _analyze_expression(self, node: ExpressionNode) -> Optional[Type]:
        """Analyze expression and return its type"""
        if isinstance(node, LiteralNode):
            node.type = node.literal_type
            return node.literal_type
        
        elif isinstance(node, IdentifierNode):
            symbol = self.symbol_table.lookup(node.name)
            if not symbol:
                self.errors.append(SemanticError(
                    f"Undefined variable '{node.name}'",
                    node.location
                ))
                return None
            node.type = symbol.type
            return symbol.type
        
        elif isinstance(node, BinaryOpNode):
            left_type = self._analyze_expression(node.left)
            right_type = self._analyze_expression(node.right)
            
            if left_type and right_type:
                # Type checking for operators
                if node.operator in ['+', '-', '*', '/', '%']:
                    if left_type._is_numeric() and right_type._is_numeric():
                        # Result type promotion
                        node.type = left_type if left_type.size >= right_type.size else right_type
                        return node.type
                    else:
                        self.errors.append(SemanticError(
                            f"Arithmetic operators require numeric types",
                            node.location
                        ))
                
                elif node.operator in ['==', '!=', '<', '>', '<=', '>=']:
                    if left_type.is_compatible_with(right_type):
                        node.type = BOOL_TYPE
                        return BOOL_TYPE
                    else:
                        self.errors.append(SemanticError(
                            f"Cannot compare {left_type} with {right_type}",
                            node.location
                        ))
                
                elif node.operator in ['&&', '||']:
                    node.type = BOOL_TYPE
                    return BOOL_TYPE
            
            return None
        
        elif isinstance(node, UnaryOpNode):
            operand_type = self._analyze_expression(node.operand)
            
            if operand_type:
                if node.operator in ['++', '--']:
                    if operand_type._is_numeric():
                        node.type = operand_type
                        return operand_type
                elif node.operator == '!':
                    node.type = BOOL_TYPE
                    return BOOL_TYPE
                elif node.operator in ['+', '-', '~']:
                    if operand_type._is_numeric():
                        node.type = operand_type
                        return operand_type
            
            return None
        
        elif isinstance(node, ArrayAccessNode):
            array_type = self._analyze_expression(node.array)
            index_type = self._analyze_expression(node.index)
            
            if array_type and array_type.kind == TypeKind.ARRAY:
                node.type = array_type.base_type
                return array_type.base_type
            elif array_type and array_type.kind == TypeKind.POINTER:
                node.type = array_type.base_type
                return array_type.base_type
            else:
                self.errors.append(SemanticError(
                    f"Cannot index non-array type {array_type}",
                    node.location
                ))
            
            return None
        
        elif isinstance(node, FunctionCallNode):
            func_type = self._analyze_expression(node.function)
            
            if func_type and func_type.kind == TypeKind.FUNCTION:
                # Check argument count
                if len(node.arguments) != len(func_type.param_types):
                    self.errors.append(SemanticError(
                        f"Function expects {len(func_type.param_types)} arguments, got {len(node.arguments)}",
                        node.location
                    ))
                
                # Check argument types
                for i, (arg, param_type) in enumerate(zip(node.arguments, func_type.param_types)):
                    arg_type = self._analyze_expression(arg)
                    if arg_type and not param_type.is_compatible_with(arg_type):
                        self.errors.append(SemanticError(
                            f"Argument {i+1} type mismatch: expected {param_type}, got {arg_type}",
                            node.location
                        ))
                
                node.type = func_type.return_type
                return func_type.return_type
            
            return None
        
        elif isinstance(node, CastNode):
            self._analyze_expression(node.expression)
            node.type = node.target_type
            return node.target_type
        
        elif isinstance(node, TernaryNode):
            self._analyze_expression(node.condition)
            true_type = self._analyze_expression(node.true_expr)
            false_type = self._analyze_expression(node.false_expr)
            
            if true_type and false_type:
                if true_type.is_compatible_with(false_type):
                    node.type = true_type
                    return true_type
                else:
                    self.errors.append(SemanticError(
                        f"Ternary branches have incompatible types: {true_type} and {false_type}",
                        node.location
                    ))
            
            return None
        
        return None
