from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from .tokens import SourceLocation

@dataclass
class Symbol:
    name: str
    type_name: str
    scope_level: int
    location: Optional[SourceLocation] = None
    is_defined: bool = True
    is_used: bool = False
    is_extern: bool = False
    is_static: bool = False
    is_const: bool = False
    is_volatile: bool = False
    is_register: bool = False
    asm_label: Optional[str] = None
    value: Optional[Any] = None
    size: Optional[int] = None
    offset: int = 0
    is_function: bool = False
    parameters: List[Tuple[str, str]] = field(default_factory=list)
    return_type: Optional[str] = None
    references: List[SourceLocation] = field(default_factory=list)
    
    def add_reference(self, location: SourceLocation):
        self.is_used = True
        self.references.append(location)

class SymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]
        self.current_scope_level = 0
        self.global_scope = self.scopes[0]
    
    def enter_scope(self):
        self.current_scope_level += 1
        self.scopes.append({})
    
    def exit_scope(self):
        if self.current_scope_level > 0:
            self.scopes.pop()
            self.current_scope_level -= 1
    
    def define(self, symbol: Symbol) -> bool:
        symbol.scope_level = self.current_scope_level
        current = self.scopes[self.current_scope_level]
        if symbol.name in current:
            return False
        current[symbol.name] = symbol
        return True
    
    def lookup(self, name: str) -> Optional[Symbol]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def lookup_current_scope(self, name: str) -> Optional[Symbol]:
        return self.scopes[self.current_scope_level].get(name)
    
    def get_all_symbols(self) -> List[Symbol]:
        symbols = []
        for scope in self.scopes:
            symbols.extend(scope.values())
        return symbols
    
    def get_unused_symbols(self) -> List[Symbol]:
        return [s for s in self.get_all_symbols() if not s.is_used and not s.is_extern]
