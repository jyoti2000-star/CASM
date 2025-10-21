from typing import Set, List

class CodeGenerationContext:
    """Context for code generation"""
    
    def __init__(self):
        self.temp_counter = 0
        self.label_counter = 0
        self.string_counter = 0
        self.register_pool = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11']
        self.used_registers: Set[str] = set()
        self.stack_offset = 0
        
    def allocate_temp(self) -> str:
        self.temp_counter += 1
        return f"T{self.temp_counter}"
    
    def allocate_label(self, prefix: str = "L") -> str:
        self.label_counter += 1
        return f"{prefix}{self.label_counter}"
    
    def allocate_string_label(self) -> str:
        self.string_counter += 1
        return f"STR{self.string_counter}"
    
    def allocate_register(self) -> Optional[str]:
        for reg in self.register_pool:
            if reg not in self.used_registers:
                self.used_registers.add(reg)
                return reg
        return None
    
    def free_register(self, reg: str):
        self.used_registers.discard(reg)

# Visitor interface for code generation
from abc import ABC, abstractmethod
class ASTVisitor(ABC):
    @abstractmethod
    def visit_program(self, node): pass
    @abstractmethod
    def visit_function(self, node): pass
    @abstractmethod
    def visit_struct(self, node): pass
    @abstractmethod
    def visit_var_declaration(self, node): pass
    @abstractmethod
    def visit_assignment(self, node): pass
    @abstractmethod
    def visit_binary_op(self, node): pass
    @abstractmethod
    def visit_unary_op(self, node): pass
    @abstractmethod
    def visit_literal(self, node): pass
    @abstractmethod
    def visit_identifier(self, node): pass
    @abstractmethod
    def visit_array_access(self, node): pass
    @abstractmethod
    def visit_member_access(self, node): pass
    @abstractmethod
    def visit_function_call(self, node): pass
    @abstractmethod
    def visit_cast(self, node): pass
    @abstractmethod
    def visit_ternary(self, node): pass
    @abstractmethod
    def visit_if(self, node): pass
    @abstractmethod
    def visit_while(self, node): pass
    @abstractmethod
    def visit_do_while(self, node): pass
    @abstractmethod
    def visit_for(self, node): pass
    @abstractmethod
    def visit_switch(self, node): pass
    @abstractmethod
    def visit_break(self, node): pass
    @abstractmethod
    def visit_continue(self, node): pass
    @abstractmethod
    def visit_return(self, node): pass
    @abstractmethod
    def visit_print(self, node): pass
    @abstractmethod
    def visit_scanf(self, node): pass
    @abstractmethod
    def visit_assembly(self, node): pass
    @abstractmethod
    def visit_comment(self, node): pass
    @abstractmethod
    def visit_c_code_block(self, node): pass
    @abstractmethod
    def visit_optimization_hint(self, node): pass
