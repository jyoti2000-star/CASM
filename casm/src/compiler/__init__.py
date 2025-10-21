"""Compiler package re-exporting submodules from former compiler.py"""
from .tokens import TokenType, Token, SourceLocation
from .types import TypeKind, Type, TypeFactory, INT_TYPE, INT8_TYPE, INT16_TYPE, INT32_TYPE, INT64_TYPE, UINT_TYPE, FLOAT_TYPE, DOUBLE_TYPE, CHAR_TYPE, BOOL_TYPE, VOID_TYPE
from .symbols import Symbol, SymbolTable
from .ast import *
from .lexer import CASMLexer
from .semantic import SemanticAnalyzer, SemanticError
from .optimizer import Optimizer, OptimizationPass, ConstantFoldingPass, DeadCodeEliminationPass
from .codegen import CodeGenerationContext, ASTVisitor

__all__ = [
    'TokenType', 'Token', 'SourceLocation',
    'TypeKind', 'Type', 'TypeFactory',
    'INT_TYPE','INT8_TYPE','INT16_TYPE','INT32_TYPE','INT64_TYPE','UINT_TYPE','FLOAT_TYPE','DOUBLE_TYPE','CHAR_TYPE','BOOL_TYPE','VOID_TYPE',
    'Symbol','SymbolTable',
    'CASMLexer',
    'SemanticAnalyzer','SemanticError',
    'Optimizer','OptimizationPass','ConstantFoldingPass','DeadCodeEliminationPass',
    'CodeGenerationContext','ASTVisitor'
]
