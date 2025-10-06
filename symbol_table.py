#!/usr/bin/env python3

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class SymbolType(Enum):
    """Types of symbols in the symbol table"""
    VARIABLE = "variable"
    FUNCTION = "function"
    CONSTANT = "constant"
    ARRAY = "array"
    LABEL = "label"
    MACRO = "macro"
    STRUCT = "struct"
    ENUM = "enum"

class DataType(Enum):
    """Data types supported by the compiler"""
    BYTE = "byte"      # 8-bit
    WORD = "word"      # 16-bit
    DWORD = "dword"    # 32-bit
    QWORD = "qword"    # 64-bit
    FLOAT = "float"    # 32-bit float
    DOUBLE = "double"  # 64-bit double
    STRING = "string"  # null-terminated string
    POINTER = "pointer" # memory address
    BOOLEAN = "boolean" # true/false
    VOID = "void"      # no type
    AUTO = "auto"      # type inference

class Scope(Enum):
    """Symbol scope levels"""
    GLOBAL = "global"
    FUNCTION = "function"
    BLOCK = "block"
    LOOP = "loop"

@dataclass
class Symbol:
    """Represents a symbol in the symbol table"""
    name: str
    symbol_type: SymbolType
    data_type: DataType
    scope: Scope
    size: int = 1
    offset: int = 0
    is_initialized: bool = False
    is_used: bool = False
    is_constant: bool = False
    value: Optional[Union[str, int, float, bool]] = None
    array_size: Optional[int] = None
    function_params: Optional[List['Symbol']] = None
    function_return_type: Optional[DataType] = None
    memory_location: Optional[str] = None
    line_declared: int = 0
    line_last_used: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Symbol({self.name}, {self.symbol_type.value}, {self.data_type.value}, {self.scope.value})"

class SymbolTable:
    """Advanced symbol table with scoping support"""
    
    def __init__(self):
        self.symbols: Dict[str, List[Symbol]] = {}
        self.current_scope: Scope = Scope.GLOBAL
        self.scope_stack: List[Scope] = [Scope.GLOBAL]
        self.scope_depth: int = 0
        self.memory_offset: int = 0
        self.function_stack: List[str] = []
        self.type_definitions: Dict[str, DataType] = {}
        
        # Built-in types
        self._register_builtin_types()
    
    def _register_builtin_types(self):
        """Register built-in data types"""
        builtin_types = {
            'int8': DataType.BYTE,
            'int16': DataType.WORD,
            'int32': DataType.DWORD,
            'int64': DataType.QWORD,
            'uint8': DataType.BYTE,
            'uint16': DataType.WORD,
            'uint32': DataType.DWORD,
            'uint64': DataType.QWORD,
            'char': DataType.BYTE,
            'short': DataType.WORD,
            'int': DataType.DWORD,
            'long': DataType.QWORD,
            'float': DataType.FLOAT,
            'double': DataType.DOUBLE,
            'string': DataType.STRING,
            'bool': DataType.BOOLEAN,
            'void': DataType.VOID,
        }
        self.type_definitions.update(builtin_types)
    
    def enter_scope(self, scope_type: Scope, function_name: str = None):
        """Enter a new scope"""
        self.scope_stack.append(scope_type)
        self.current_scope = scope_type
        self.scope_depth += 1
        
        if scope_type == Scope.FUNCTION and function_name:
            self.function_stack.append(function_name)
    
    def exit_scope(self):
        """Exit current scope"""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]
            self.scope_depth -= 1
            
            if self.function_stack and self.current_scope != Scope.FUNCTION:
                self.function_stack.pop()
    
    def declare_symbol(self, name: str, symbol_type: SymbolType, 
                      data_type: DataType = DataType.AUTO,
                      value: Any = None, array_size: int = None,
                      line_number: int = 0) -> Symbol:
        """Declare a new symbol"""
        
        # Check for redeclaration in current scope
        if self.is_declared_in_current_scope(name):
            raise SymbolError(f"Symbol '{name}' already declared in current scope")
        
        # Create symbol
        symbol = Symbol(
            name=name,
            symbol_type=symbol_type,
            data_type=data_type,
            scope=self.current_scope,
            value=value,
            array_size=array_size,
            line_declared=line_number,
            is_initialized=value is not None
        )
        
        # Calculate size and memory location
        symbol.size = self._calculate_symbol_size(symbol)
        symbol.memory_location = self._allocate_memory(symbol)
        
        # Add to symbol table
        if name not in self.symbols:
            self.symbols[name] = []
        self.symbols[name].append(symbol)
        
        return symbol
    
    def lookup_symbol(self, name: str) -> Optional[Symbol]:
        """Look up a symbol, considering scope hierarchy"""
        if name not in self.symbols:
            return None
        
        # Search from innermost to outermost scope
        for scope in reversed(self.scope_stack):
            for symbol in reversed(self.symbols[name]):
                if symbol.scope == scope:
                    symbol.is_used = True
                    return symbol
        
        # If not found in any scope, return the most recent global symbol
        for symbol in reversed(self.symbols[name]):
            if symbol.scope == Scope.GLOBAL:
                symbol.is_used = True
                return symbol
        
        return None
    
    def is_declared_in_current_scope(self, name: str) -> bool:
        """Check if symbol is declared in current scope"""
        if name not in self.symbols:
            return False
        
        for symbol in self.symbols[name]:
            if symbol.scope == self.current_scope:
                return True
        return False
    
    def get_symbols_in_scope(self, scope: Scope) -> List[Symbol]:
        """Get all symbols in a specific scope"""
        result = []
        for symbol_list in self.symbols.values():
            for symbol in symbol_list:
                if symbol.scope == scope:
                    result.append(symbol)
        return result
    
    def get_unused_symbols(self) -> List[Symbol]:
        """Get all unused symbols (for warnings)"""
        unused = []
        for symbol_list in self.symbols.values():
            for symbol in symbol_list:
                if not symbol.is_used and symbol.symbol_type == SymbolType.VARIABLE:
                    unused.append(symbol)
        return unused
    
    def get_uninitialized_symbols(self) -> List[Symbol]:
        """Get all uninitialized symbols (for warnings)"""
        uninitialized = []
        for symbol_list in self.symbols.values():
            for symbol in symbol_list:
                if not symbol.is_initialized and symbol.symbol_type == SymbolType.VARIABLE:
                    uninitialized.append(symbol)
        return uninitialized
    
    def update_symbol_usage(self, name: str, line_number: int):
        """Update symbol usage information"""
        symbol = self.lookup_symbol(name)
        if symbol:
            symbol.is_used = True
            symbol.line_last_used = line_number
    
    def _calculate_symbol_size(self, symbol: Symbol) -> int:
        """Calculate the memory size needed for a symbol"""
        base_size = {
            DataType.BYTE: 1,
            DataType.WORD: 2,
            DataType.DWORD: 4,
            DataType.QWORD: 8,
            DataType.FLOAT: 4,
            DataType.DOUBLE: 8,
            DataType.POINTER: 8,
            DataType.BOOLEAN: 1,
        }.get(symbol.data_type, 8)  # Default to 8 bytes
        
        if symbol.symbol_type == SymbolType.ARRAY and symbol.array_size:
            return base_size * symbol.array_size
        
        if symbol.data_type == DataType.STRING and symbol.value:
            return len(str(symbol.value)) + 1  # +1 for null terminator
        
        return base_size
    
    def _allocate_memory(self, symbol: Symbol) -> str:
        """Allocate memory location for symbol"""
        if symbol.scope == Scope.GLOBAL:
            return f"_global_{symbol.name}"
        elif symbol.scope == Scope.FUNCTION:
            return f"rbp-{self.memory_offset + symbol.size}"
        else:
            return f"_local_{symbol.name}_{self.scope_depth}"
    
    def generate_symbol_report(self) -> str:
        """Generate a comprehensive symbol table report"""
        report = []
        report.append("=" * 80)
        report.append("SYMBOL TABLE REPORT")
        report.append("=" * 80)
        
        # Group by scope
        for scope in Scope:
            symbols = self.get_symbols_in_scope(scope)
            if symbols:
                report.append(f"\n{scope.value.upper()} SCOPE:")
                report.append("-" * 40)
                for symbol in symbols:
                    usage = "USED" if symbol.is_used else "UNUSED"
                    init = "INIT" if symbol.is_initialized else "UNINIT"
                    report.append(f"  {symbol.name:<20} {symbol.symbol_type.value:<10} "
                                f"{symbol.data_type.value:<10} {usage:<8} {init}")
        
        # Warnings section
        unused = self.get_unused_symbols()
        uninitialized = self.get_uninitialized_symbols()
        
        if unused or uninitialized:
            report.append("\nWARNINGS:")
            report.append("-" * 40)
            
            if unused:
                report.append("Unused variables:")
                for symbol in unused:
                    report.append(f"  - {symbol.name} (line {symbol.line_declared})")
            
            if uninitialized:
                report.append("Uninitialized variables:")
                for symbol in uninitialized:
                    report.append(f"  - {symbol.name} (line {symbol.line_declared})")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def export_to_json(self) -> Dict[str, Any]:
        """Export symbol table to JSON-serializable format"""
        export_data = {
            "symbols": {},
            "current_scope": self.current_scope.value,
            "scope_depth": self.scope_depth,
            "memory_offset": self.memory_offset
        }
        
        for name, symbol_list in self.symbols.items():
            export_data["symbols"][name] = []
            for symbol in symbol_list:
                symbol_data = {
                    "name": symbol.name,
                    "symbol_type": symbol.symbol_type.value,
                    "data_type": symbol.data_type.value,
                    "scope": symbol.scope.value,
                    "size": symbol.size,
                    "offset": symbol.offset,
                    "is_initialized": symbol.is_initialized,
                    "is_used": symbol.is_used,
                    "is_constant": symbol.is_constant,
                    "value": symbol.value,
                    "array_size": symbol.array_size,
                    "memory_location": symbol.memory_location,
                    "line_declared": symbol.line_declared,
                    "line_last_used": symbol.line_last_used,
                    "attributes": symbol.attributes
                }
                export_data["symbols"][name].append(symbol_data)
        
        return export_data

class SymbolError(Exception):
    """Exception raised for symbol table errors"""
    def __init__(self, message: str, symbol: Optional[Symbol] = None):
        self.message = message
        self.symbol = symbol
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        if self.symbol:
            return f"Symbol error for '{self.symbol.name}': {self.message}"
        return f"Symbol error: {self.message}"

class TypeChecker:
    """Type checking utilities"""
    
    @staticmethod
    def is_compatible(from_type: DataType, to_type: DataType) -> bool:
        """Check if types are compatible for assignment/operation"""
        if from_type == to_type:
            return True
        
        # Numeric type compatibility
        numeric_types = {DataType.BYTE, DataType.WORD, DataType.DWORD, DataType.QWORD}
        if from_type in numeric_types and to_type in numeric_types:
            return True
        
        # Float compatibility
        if from_type == DataType.FLOAT and to_type == DataType.DOUBLE:
            return True
        if from_type == DataType.DOUBLE and to_type == DataType.FLOAT:
            return True  # With potential precision loss
        
        # Pointer compatibility
        if from_type == DataType.POINTER or to_type == DataType.POINTER:
            return True  # Pointers can be cast
        
        # Auto type can be assigned from anything
        if to_type == DataType.AUTO:
            return True
        
        return False
    
    @staticmethod
    def get_result_type(left_type: DataType, right_type: DataType, operation: str) -> DataType:
        """Get the result type of a binary operation"""
        # Auto type propagation
        if left_type == DataType.AUTO:
            return right_type
        if right_type == DataType.AUTO:
            return left_type
        
        # Same types
        if left_type == right_type:
            return left_type
        
        # Numeric promotion rules
        numeric_hierarchy = [DataType.BYTE, DataType.WORD, DataType.DWORD, DataType.QWORD]
        if left_type in numeric_hierarchy and right_type in numeric_hierarchy:
            return max(left_type, right_type, key=lambda x: numeric_hierarchy.index(x))
        
        # Float promotion
        if left_type in [DataType.FLOAT, DataType.DOUBLE] or right_type in [DataType.FLOAT, DataType.DOUBLE]:
            if left_type == DataType.DOUBLE or right_type == DataType.DOUBLE:
                return DataType.DOUBLE
            return DataType.FLOAT
        
        # Default to larger type
        return DataType.QWORD

# Global symbol table instance
global_symbol_table = SymbolTable()