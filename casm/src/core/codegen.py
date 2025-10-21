import re
from typing import List
from .ast import ProgramNode, VarDeclarationNode, FunctionDeclarationNode, StructDeclarationNode, UnionDeclarationNode, EnumDeclarationNode, TypedefNode, MacroDefinitionNode, AssignmentNode, IfNode, WhileNode, DoWhileNode, ForNode, SwitchNode, BreakNode, ContinueNode, ReturnNode, GotoNode, LabelNode, PrintlnNode, ScanfNode, AssemblyNode, AsmBlockNode, CommentNode, CCodeBlockNode, ExternDirectiveNode, OptimizationHintNode
from .diagnostics import DiagnosticEngine
from .visitor import ASTVisitor

class AssemblyCodeGenerator(ASTVisitor):
    def __init__(self):
        self.data_section = []
        self.bss_section = []
        self.text_section = []
        self.rodata_section = []
        self.string_labels = {}
        self.variable_labels = {}
        self.variable_info = {}
        self.function_labels = {}
        self.label_counter = 0
        self.var_counter = 0
        self.str_counter = 0
        self.current_function = None
        self.diagnostics = DiagnosticEngine()

    # Visitor methods copied from original core.py follow; omitted here for brevity
