from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class SourceLocation:
    line: int
    column: int
    file: str = "<stdin>"
    raw_line: str = ""


@dataclass
class Symbol:
    """Enhanced symbol representation"""
    name: str
    type: object
    scope_level: int
    is_defined: bool = True
    is_extern: bool = False
    is_static: bool = False
    is_register: bool = False
    location: Optional[SourceLocation] = None
    asm_label: Optional[str] = None
    value: Optional[Any] = None
    is_function: bool = False
    is_parameter: bool = False
    offset: int = 0  # Stack offset or struct offset


class SymbolTable:
    """Advanced symbol table with lexical scoping"""
    
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]  # Stack of scopes
        self.current_scope_level = 0
        self.global_scope = self.scopes[0]
        
    def enter_scope(self):
        """Enter a new lexical scope"""
        self.current_scope_level += 1
        self.scopes.append({})
        
    def exit_scope(self):
        """Exit current scope"""
        if self.current_scope_level > 0:
            self.scopes.pop()
            self.current_scope_level -= 1
    
    def define(self, symbol: Symbol) -> bool:
        """Define a symbol in current scope"""
        symbol.scope_level = self.current_scope_level
        current = self.scopes[self.current_scope_level]
        
        if symbol.name in current:
            return False  # Already defined in this scope
        
        current[symbol.name] = symbol
        return True
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Lookup symbol in all visible scopes"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def lookup_current_scope(self, name: str) -> Optional[Symbol]:
        """Lookup symbol only in current scope"""
        return self.scopes[self.current_scope_level].get(name)
    
    def get_all_symbols(self) -> List[Symbol]:
        """Get all symbols from all scopes"""
        symbols = []
        for scope in self.scopes:
            symbols.extend(scope.values())
        return symbols
