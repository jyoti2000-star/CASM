from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict

class TypeKind(Enum):
    PRIMITIVE = auto()
    POINTER = auto()
    ARRAY = auto()
    STRUCT = auto()
    UNION = auto()
    FUNCTION = auto()
    VOID = auto()


@dataclass
class Type:
    kind: TypeKind
    name: str
    size: int = 0
    alignment: int = 0
    is_const: bool = False
    is_volatile: bool = False
    is_signed: bool = True
    
    # For complex types
    base_type: Optional['Type'] = None  # For pointers/arrays
    members: Dict[str, 'Type'] = field(default_factory=dict)  # For structs/unions
    param_types: List['Type'] = field(default_factory=list)  # For functions
    return_type: Optional['Type'] = None  # For functions
    array_size: Optional[int] = None
    
    def __str__(self):
        qualifiers = []
        if self.is_const:
            qualifiers.append("const")
        if self.is_volatile:
            qualifiers.append("volatile")
        
        qual_str = " ".join(qualifiers) + " " if qualifiers else ""
        
        if self.kind == TypeKind.POINTER:
            return f"{qual_str}{self.base_type}*"
        elif self.kind == TypeKind.ARRAY:
            return f"{qual_str}{self.base_type}[{self.array_size}]"
        else:
            return f"{qual_str}{self.name}"
    
    def is_compatible_with(self, other: 'Type') -> bool:
        """Check type compatibility"""
        if self.kind != other.kind:
            return False
        if self.kind == TypeKind.PRIMITIVE:
            return self.name == other.name or self._is_numeric() and other._is_numeric()
        if self.kind == TypeKind.POINTER:
            return self.base_type.is_compatible_with(other.base_type)
        return self.name == other.name
    
    def _is_numeric(self) -> bool:
        return self.name in {'int', 'int8', 'int16', 'int32', 'int64', 
                            'uint', 'uint8', 'uint16', 'uint32', 'uint64',
                            'float', 'double', 'char'}


class TypeFactory:
    _types_cache = {}
    
    @classmethod
    def get_primitive(cls, name: str, size: int, signed: bool = True) -> Type:
        key = (name, size, signed)
        if key not in cls._types_cache:
            cls._types_cache[key] = Type(
                kind=TypeKind.PRIMITIVE,
                name=name,
                size=size,
                alignment=size,
                is_signed=signed
            )
        return cls._types_cache[key]
    
    @classmethod
    def get_pointer(cls, base: Type) -> Type:
        return Type(
            kind=TypeKind.POINTER,
            name=f"{base.name}*",
            size=8,  # 64-bit pointer
            alignment=8,
            base_type=base
        )
    
    @classmethod
    def get_array(cls, base: Type, size: int) -> Type:
        return Type(
            kind=TypeKind.ARRAY,
            name=f"{base.name}[{size}]",
            size=base.size * size,
            alignment=base.alignment,
            base_type=base,
            array_size=size
        )

# Predefined types
INT_TYPE = TypeFactory.get_primitive("int", 4)
INT8_TYPE = TypeFactory.get_primitive("int8", 1)
INT16_TYPE = TypeFactory.get_primitive("int16", 2)
INT32_TYPE = TypeFactory.get_primitive("int32", 4)
INT64_TYPE = TypeFactory.get_primitive("int64", 8)
UINT_TYPE = TypeFactory.get_primitive("uint", 4, signed=False)
FLOAT_TYPE = TypeFactory.get_primitive("float", 4)
DOUBLE_TYPE = TypeFactory.get_primitive("double", 8)
CHAR_TYPE = TypeFactory.get_primitive("char", 1)
BOOL_TYPE = TypeFactory.get_primitive("bool", 1)
VOID_TYPE = Type(kind=TypeKind.VOID, name="void", size=0, alignment=1)
