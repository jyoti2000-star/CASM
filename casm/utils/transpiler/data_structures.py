from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from .enums import SectionType, InstructionType

@dataclass
class Variable:
    name: str
    type_: str
    size: int
    offset: int = 0
    is_global: bool = False
    is_const: bool = False
    is_volatile: bool = False
    alignment: int = 8
    section: SectionType = SectionType.DATA
    initial_value: Optional[Any] = None
    usage_count: int = 0
    last_write: Optional[int] = None
    last_read: Optional[int] = None

@dataclass
class Function:
    name: str
    params: List[Tuple[str, str]]
    return_type: str = "void"
    local_size: int = 0
    is_global: bool = False
    is_inline: bool = False
    is_pure: bool = False
    is_noreturn: bool = False
    preserves_regs: List[str] = field(default_factory=list)
    clobbers_regs: List[str] = field(default_factory=list)
    stack_alignment: int = 16
    uses_simd: bool = False
    is_leaf: bool = True
    call_count: int = 0
    max_recursion_depth: int = 0
    hot_path: bool = False
    optimization_hints: Dict[str, Any] = field(default_factory=dict)
    entry_block: Optional[str] = None
    exit_blocks: Set[str] = field(default_factory=set)
    complexity: int = 0

@dataclass
class Macro:
    name: str
    params: List[str]
    body: List[str]
    is_variadic: bool = False
    doc: str = ""
    expansion_count: int = 0

@dataclass
class Loop:
    start_label: str
    end_label: str
    continue_label: Optional[str] = None
    counter_reg: Optional[str] = None
    trip_count: Optional[int] = None
    is_vectorizable: bool = False
    unroll_factor: int = 1
    increment: Optional[str] = None
    invariant_code: List[str] = field(default_factory=list)
    nesting_level: int = 0
    contains_calls: bool = False

@dataclass
class BasicBlock:
    label: str
    instructions: List[str] = field(default_factory=list)
    predecessors: Set[str] = field(default_factory=set)
    successors: Set[str] = field(default_factory=set)
    dominators: Set[str] = field(default_factory=set)
    dominated_by: Set[str] = field(default_factory=set)
    dom_frontier: Set[str] = field(default_factory=set)
    live_in: Set[str] = field(default_factory=set)
    live_out: Set[str] = field(default_factory=set)
    gen: Set[str] = field(default_factory=set)
    kill: Set[str] = field(default_factory=set)
    is_loop_header: bool = False
    is_loop_exit: bool = False
    execution_frequency: int = 0

@dataclass
class Instruction:
    mnemonic: str
    operands: List[str]
    line_number: int
    original_line: str
    type: InstructionType = InstructionType.SCALAR
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    flags_affected: Set[str] = field(default_factory=set)
    latency: int = 1
    throughput: float = 1.0
    can_eliminate: bool = False
    is_branch: bool = False
    is_call: bool = False
    is_return: bool = False
