from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Set, Any, Dict
from .enums import OptimizationLevel, VariableScope, CallingConvention, CompilerType

@dataclass
class CVariable:
    """C variable metadata"""
    name: str
    type_: str
    value: Optional[str] = None
    scope: VariableScope = VariableScope.LOCAL
    is_const: bool = False
    is_volatile: bool = False
    is_static: bool = False
    array_size: Optional[int] = None
    pointer_depth: int = 0
    struct_name: Optional[str] = None
    alignment: Optional[int] = None
    section: Optional[str] = None
    asm_label: Optional[str] = None
    usage_count: int = 0

@dataclass
class CFunction:
    """C function metadata"""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]
    body: str
    is_inline: bool = False
    is_static: bool = False
    is_extern: bool = False
    attributes: List[str] = field(default_factory=list)
    calling_conv: Optional[CallingConvention] = None
    asm_label: Optional[str] = None
    optimization_hints: Dict[str, Any] = field(default_factory=dict)
    line_number: int = 0
    complexity: int = 0

@dataclass
class CHeader:
    """Header file metadata"""
    name: str
    is_system: bool
    path: Optional[Path] = None
    content_hash: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)

@dataclass
class CCodeBlock:
    """C code block with assembly markers"""
    block_id: str
    source_code: str
    compiled_code: Optional[str] = None
    start_marker: str = ""
    end_marker: str = ""
    variables_used: Set[str] = field(default_factory=set)
    functions_called: Set[str] = field(default_factory=set)
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    compile_time: float = 0.0
    object_size: int = 0

@dataclass
class CompilationResult:
    """Compilation result metadata"""
    success: bool
    assembly_code: str = ""
    object_file: Optional[Path] = None
    stderr: str = ""
    stdout: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    compile_time: float = 0.0
    compiler_used: Optional[CompilerType] = None

@dataclass
class CompilerConfig:
    """Compiler configuration"""
    compiler: CompilerType
    optimization: OptimizationLevel
    warning_level: int
    debug_info: bool
    position_independent: bool
    sanitizers: List[str]
    custom_flags: List[str]
    defines: Dict[str, str]
    include_paths: List[Path]
    standard: str  # e.g., "c11", "c99"
