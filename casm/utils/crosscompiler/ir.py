from dataclasses import dataclass, field
from typing import Any, List, Optional, Set
from .enums import IROpcode

@dataclass
class IROperand:
    type: str  # "reg", "imm", "mem", "label"
    value: Any
    size: int = 32
    
    def __str__(self):
        if self.type == "reg":
            return f"v{self.value}"
        elif self.type == "imm":
            return f"#{self.value}"
        elif self.type == "mem":
            return f"[{self.value}]"
        elif self.type == "label":
            return f"@{self.value}"
        return str(self.value)

@dataclass
class IRInstruction:
    opcode: IROpcode
    operands: List[IROperand]
    condition: Optional[str] = None
    size: int = 32
    flags_set: Set[str] = field(default_factory=set)
    flags_used: Set[str] = field(default_factory=set)
    comment: str = ""
    
    def __str__(self):
        ops = ", ".join(str(op) for op in self.operands)
        cond = f".{self.condition}" if self.condition else ""
        return f"{self.opcode.name.lower()}{cond} {ops}"
