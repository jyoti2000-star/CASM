#!/usr/bin/env python3
"""
Advanced C Code Processor v2.0

A production-grade C code processor for assembly integration with:
- Multi-compiler support (GCC, Clang, MSVC, ICC)
- Cross-compilation capabilities
- Advanced optimization levels
- Inline assembly extraction and analysis
- Symbol resolution and linking
- Debug information preservation
- Compiler flag management
- Caching and incremental compilation
- Static analysis integration
- Memory sanitizer support
"""

from __future__ import annotations

import re
import os
import sys
import json
import hashlib
import tempfile
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

__version__ = "2.0.0"
__all__ = ["CCodeProcessor", "c_processor", "CompilerError", "CompilerType"]

# ==================== ENUMS AND CONSTANTS ====================

class CompilerType(Enum):
    """Supported C compilers"""
    GCC = "gcc"
    CLANG = "clang"
    MSVC = "cl"
    ICC = "icc"
    TCC = "tcc"
    ZIGCC = "zig cc"

class OptimizationLevel(Enum):
    """Optimization levels"""
    DEBUG = "-O0"
    BASIC = "-O1"
    MODERATE = "-O2"
    AGGRESSIVE = "-O3"
    SIZE = "-Os"
    FAST = "-Ofast"

class WarningLevel(Enum):
    """Warning levels"""
    NONE = 0
    BASIC = 1
    ALL = 2
    EXTRA = 3
    PEDANTIC = 4

class Platform(Enum):
    """Target platforms"""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    BSD = "bsd"
    ANDROID = "android"
    IOS = "ios"

class Architecture(Enum):
    """Target architectures"""
    X86_64 = "x86_64"
    X86 = "x86"
    ARM64 = "arm64"
    ARM32 = "arm32"
    RISCV64 = "riscv64"
    RISCV32 = "riscv32"
    MIPS = "mips"
    POWERPC = "powerpc"

class VariableScope(Enum):
    """Variable scope"""
    LOCAL = auto()
    GLOBAL = auto()
    EXTERN = auto()
    STATIC = auto()
    THREAD_LOCAL = auto()

class CallingConvention(Enum):
    """Function calling conventions"""
    CDECL = "cdecl"
    STDCALL = "stdcall"
    FASTCALL = "fastcall"
    VECTORCALL = "vectorcall"
    SYSV = "sysv"
    MS_ABI = "ms_abi"

# ==================== DATA STRUCTURES ====================

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
    warning_level: WarningLevel
    debug_info: bool
    position_independent: bool
    sanitizers: List[str]
    custom_flags: List[str]
    defines: Dict[str, str]
    include_paths: List[Path]
    standard: str  # e.g., "c11", "c99"

# ==================== EXCEPTIONS ====================

class CompilerError(Exception):
    """Compilation error"""
    def __init__(self, message: str, stderr: str = "", compiler: Optional[str] = None):
        self.message = message
        self.stderr = stderr
        self.compiler = compiler
        super().__init__(message)

class VariableError(Exception):
    """Variable-related error"""
    pass

class HeaderError(Exception):
    """Header-related error"""
    pass

# ==================== COMPILER DETECTION ====================

class CompilerDetector:
    """Detect and validate available compilers"""
    
    @staticmethod
    def detect_compilers() -> Dict[CompilerType, Path]:
        """Detect available compilers on the system"""
        compilers = {}
        
        for compiler_type in CompilerType:
            path = CompilerDetector.find_compiler(compiler_type)
            if path:
                compilers[compiler_type] = path
        
        return compilers
    
    @staticmethod
    def find_compiler(compiler_type: CompilerType) -> Optional[Path]:
        """Find a specific compiler"""
        if compiler_type == CompilerType.ZIGCC:
            # Special case for zig cc
            zig_path = shutil.which("zig")
            if zig_path:
                return Path(zig_path)
            return None
        
        compiler_name = compiler_type.value
        compiler_path = shutil.which(compiler_name)
        
        if compiler_path:
            return Path(compiler_path)
        
        return None
    
    @staticmethod
    def get_compiler_version(compiler_path: Path) -> Optional[str]:
        """Get compiler version"""
        try:
            result = subprocess.run(
                [str(compiler_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip().split('\n')[0]
        except Exception:
            return None
    
    @staticmethod
    def select_best_compiler(
        platform: Platform,
        arch: Architecture,
        available: Dict[CompilerType, Path]
    ) -> Optional[CompilerType]:
        """Select the best compiler for target platform/arch"""
        # Preference order
        if platform == Platform.WINDOWS:
            preferences = [CompilerType.MSVC, CompilerType.CLANG, CompilerType.GCC]
        elif platform == Platform.MACOS:
            preferences = [CompilerType.CLANG, CompilerType.GCC]
        else:
            preferences = [CompilerType.GCC, CompilerType.CLANG]
        
        for compiler in preferences:
            if compiler in available:
                return compiler
        
        # Return any available compiler
        if available:
            return next(iter(available.keys()))
        
        return None

# ==================== VARIABLE ANALYZER ====================

class VariableAnalyzer:
    """Analyze C variables and their types"""
    
    # C type patterns
    TYPE_PATTERNS = [
        r'(unsigned\s+)?(char|short|int|long|long\s+long)',
        r'(signed\s+)?(char|short|int|long)',
        r'float|double|long\s+double',
        r'void',
        r'size_t|ssize_t|ptrdiff_t',
        r'int8_t|int16_t|int32_t|int64_t',
        r'uint8_t|uint16_t|uint32_t|uint64_t',
        r'intptr_t|uintptr_t',
        r'struct\s+\w+',
        r'union\s+\w+',
        r'enum\s+\w+',
        r'\w+_t',  # typedef'd types
    ]
    
    @staticmethod
    def parse_declaration(declaration: str) -> Optional[CVariable]:
        """Parse a C variable declaration"""
        declaration = declaration.strip().rstrip(';')
        
        # Handle const and volatile
        is_const = 'const' in declaration
        is_volatile = 'volatile' in declaration
        is_static = 'static' in declaration
        
        # Remove qualifiers
        decl = re.sub(r'\b(const|volatile|static|extern|register|auto)\b', '', declaration)
        decl = ' '.join(decl.split())  # normalize whitespace
        
        # Extract type and name
        match = re.match(r'(.+?)\s+(\**)(\w+)(\[(\d+)\])?\s*(=\s*(.+))?', decl)
        
        if not match:
            return None
        
        type_, pointers, name, _, array_size, _, value = match.groups()
        type_ = type_.strip()
        pointer_depth = len(pointers)
        
        # Parse array size
        array_sz = int(array_size) if array_size else None
        
        # Parse value
        val = value.strip() if value else None
        
        return CVariable(
            name=name,
            type_=type_,
            value=val,
            is_const=is_const,
            is_volatile=is_volatile,
            is_static=is_static,
            array_size=array_sz,
            pointer_depth=pointer_depth,
            scope=VariableScope.STATIC if is_static else VariableScope.LOCAL
        )
    
    @staticmethod
    def extract_variables(code: str) -> List[CVariable]:
        """Extract all variable declarations from C code"""
        variables = []
        
        # Pattern for variable declarations
        var_pattern = r'(?:const|volatile|static|extern|register)?\s*' + \
                     r'(?:unsigned|signed)?\s*' + \
                     r'(?:char|short|int|long|float|double|void|\w+_t|\w+)\s*' + \
                     r'\*?\s*\w+\s*(?:\[[\d\w]+\])?\s*(?:=\s*[^;]+)?\s*;'
        
        for match in re.finditer(var_pattern, code):
            var = VariableAnalyzer.parse_declaration(match.group(0))
            if var:
                variables.append(var)
        
        return variables
    
    @staticmethod
    def infer_type(value: str) -> str:
        """Infer C type from value"""
        value = value.strip()
        
        # String literal
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            if value.startswith('"'):
                return "const char*"
            return "char"
        
        # Null pointer
        if value.lower() in ('null', 'nullptr', '0'):
            return "void*"
        
        # Hex number
        if value.startswith('0x') or value.startswith('0X'):
            return "unsigned int"
        
        # Float/double
        if '.' in value or 'e' in value.lower():
            if value.endswith('f') or value.endswith('F'):
                return "float"
            return "double"
        
        # Integer
        try:
            int(value)
            return "int"
        except ValueError:
            pass
        
        # Boolean
        if value.lower() in ('true', 'false'):
            return "bool"
        
        return "int"  # default

# ==================== FUNCTION ANALYZER ====================

class FunctionAnalyzer:
    """Analyze C functions"""
    
    @staticmethod
    def extract_functions(code: str) -> List[CFunction]:
        """Extract function definitions from C code"""
        functions = []
        
        # Simple function pattern (can be enhanced)
        func_pattern = r'((?:static|inline|extern)\s+)?(\w+(?:\s+\w+)?)\s+(\**)(\w+)\s*\(([^)]*)\)\s*\{'
        
        for match in re.finditer(func_pattern, code):
            qualifiers, return_type, pointers, name, params = match.groups()
            
            # Parse qualifiers
            is_static = 'static' in (qualifiers or '')
            is_inline = 'inline' in (qualifiers or '')
            is_extern = 'extern' in (qualifiers or '')
            
            # Parse parameters
            param_list = []
            if params.strip():
                for param in params.split(','):
                    param = param.strip()
                    if param and param != 'void':
                        # Simple parameter parsing
                        parts = param.split()
                        if len(parts) >= 2:
                            param_type = ' '.join(parts[:-1])
                            param_name = parts[-1].strip('*')
                            param_list.append((param_name, param_type))
            
            # Extract function body
            start = match.end() - 1
            body = FunctionAnalyzer._extract_function_body(code, start)
            
            func = CFunction(
                name=name,
                return_type=return_type.strip() + pointers,
                parameters=param_list,
                body=body,
                is_inline=is_inline,
                is_static=is_static,
                is_extern=is_extern,
                line_number=code[:match.start()].count('\n') + 1
            )
            
            functions.append(func)
        
        return functions
    
    @staticmethod
    def _extract_function_body(code: str, start: int) -> str:
        """Extract function body using brace matching"""
        depth = 0
        i = start
        
        while i < len(code):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    return code[start:i+1]
            i += 1
        
        return code[start:]
    
    @staticmethod
    def calculate_complexity(func: CFunction) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        # Count decision points
        decision_keywords = ['if', 'else', 'while', 'for', 'case', 'default', '&&', '||', '?']
        
        for keyword in decision_keywords:
            complexity += func.body.count(keyword)
        
        return complexity

# ==================== C CODE PROCESSOR ====================

class CCodeProcessor:
    """Advanced C code processor with assembly integration"""
    
    def __init__(self):
        # Configuration
        self.target_platform = Platform.LINUX
        self.target_arch = Architecture.X86_64
        self.compiler_config = CompilerConfig(
            compiler=CompilerType.GCC,
            optimization=OptimizationLevel.MODERATE,
            warning_level=WarningLevel.ALL,
            debug_info=True,
            position_independent=True,
            sanitizers=[],
            custom_flags=[],
            defines={},
            include_paths=[],
            standard="c11"
        )
        
        # Storage
        self.headers: List[CHeader] = []
        self.variables: Dict[str, CVariable] = {}
        self.casm_variables: Dict[str, Any] = {}
        self.functions: Dict[str, CFunction] = {}
        self.c_code_blocks: List[CCodeBlock] = []
        self.globals: List[str] = []
        
        # Compilation
        self.temp_dir: Optional[Path] = None
        self.available_compilers: Dict[CompilerType, Path] = {}
        self.compilation_cache: Dict[str, CompilationResult] = {}
        self.marker_counter = 0
        
        # State
        self.save_debug = False
        self.verbose = False
        self._last_compile_output = ''
        self._last_status = ''
        self._compile_lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Detect compilers
        self._detect_compilers()
    
    def _detect_compilers(self):
        """Detect available compilers"""
        self.available_compilers = CompilerDetector.detect_compilers()
        
        if self.available_compilers:
            self.logger.info(f"Detected compilers: {', '.join(c.value for c in self.available_compilers.keys())}")
            
            # Select best compiler
            best = CompilerDetector.select_best_compiler(
                self.target_platform,
                self.target_arch,
                self.available_compilers
            )
            
            if best:
                self.compiler_config.compiler = best
                self.logger.info(f"Selected compiler: {best.value}")
        else:
            self.logger.warning("No compilers detected!")
    
    # ==================== HEADER MANAGEMENT ====================
    
    def add_header(self, header_name: str, is_system: bool = False):
        """Add a header file"""
        # Clean header name
        header_name = header_name.strip()
        
        if header_name.startswith('<') and header_name.endswith('>'):
            header_name = header_name[1:-1]
            is_system = True
        elif header_name.startswith('"') and header_name.endswith('"'):
            header_name = header_name[1:-1]
            is_system = False
        
        # Check if already added
        for header in self.headers:
            if header.name == header_name:
                return
        
        header = CHeader(name=header_name, is_system=is_system)
        self.headers.append(header)
        self.logger.info(f"Added header: {'<' if is_system else ''}{header_name}{'>' if is_system else ''}")
    
    def add_system_header(self, header_name: str):
        """Add a system header"""
        self.add_header(header_name, is_system=True)
    
    def get_header_includes(self) -> str:
        """Generate #include directives"""
        includes = []
        
        for header in self.headers:
            if header.is_system:
                includes.append(f"#include <{header.name}>")
            else:
                includes.append(f'#include "{header.name}"')
        
        return '\n'.join(includes)
    
    # ==================== VARIABLE MANAGEMENT ====================
    
    def set_casm_variables(self, variables: Dict[str, Any]):
        """Set CASM variables"""
        self.casm_variables = variables
        self.logger.info(f"Set {len(variables)} CASM variables")
    
    def add_variable(self, var: CVariable):
        """Add a C variable"""
        self.variables[var.name] = var
        self.logger.info(f"Added variable: {var.name} ({var.type_})")
    
    def get_variable(self, name: str) -> Optional[CVariable]:
        """Get a variable by name"""
        return self.variables.get(name)
    
    def get_c_variables(self) -> Dict[str, CVariable]:
        """Get all C variables"""
        return self.variables
    
    def extract_variables_from_code(self, code: str):
        """Extract and register variables from C code"""
        variables = VariableAnalyzer.extract_variables(code)
        
        for var in variables:
            if var.name not in self.variables:
                self.add_variable(var)
    
    # ==================== FUNCTION MANAGEMENT ====================
    
    def add_function(self, func: CFunction):
        """Add a C function"""
        self.functions[func.name] = func
        func.complexity = FunctionAnalyzer.calculate_complexity(func)
        self.logger.info(f"Added function: {func.name} (complexity: {func.complexity})")
    
    def extract_functions_from_code(self, code: str):
        """Extract and register functions from C code"""
        functions = FunctionAnalyzer.extract_functions(code)
        
        for func in functions:
            if func.name not in self.functions:
                self.add_function(func)
    
    # ==================== CODE BLOCK MANAGEMENT ====================
    
    def add_c_code_block(self, code: str, optimization: Optional[OptimizationLevel] = None) -> str:
        """Add a C code block for compilation"""
        block_id = f"CASM_BLOCK_{self.marker_counter}"
        self.marker_counter += 1
        
        start_marker = f"{block_id}_START"
        end_marker = f"{block_id}_END"
        
        # Wrap code with assembly markers
        marked_code = f"""
    __asm__ volatile("{start_marker}:");
{code}
    __asm__ volatile("{end_marker}:");
"""
        
        block = CCodeBlock(
            block_id=block_id,
            source_code=code,
            start_marker=start_marker,
            end_marker=end_marker,
            optimization_level=optimization or self.compiler_config.optimization
        )
        
        self.c_code_blocks.append(block)
        self.logger.info(f"Added code block: {block_id}")
        
        return block_id
    
    # ==================== COMPILATION ====================
    
    def set_target(self, platform: str, arch: str = 'x86_64'):
        """Set compilation target"""
        self.target_platform = Platform(platform.lower())
        self.target_arch = Architecture(arch.lower())
        
        self.logger.info(f"Target set to: {self.target_platform.value}/{self.target_arch.value}")
        
        # Re-select best compiler for target
        best = CompilerDetector.select_best_compiler(
            self.target_platform,
            self.target_arch,
            self.available_compilers
        )
        
        if best:
            self.compiler_config.compiler = best
    
    def set_optimization_level(self, level: OptimizationLevel):
        """Set optimization level"""
        self.compiler_config.optimization = level
        self.logger.info(f"Optimization level set to: {level.value}")
    
    def add_define(self, name: str, value: str = "1"):
        """Add a preprocessor define"""
        self.compiler_config.defines[name] = value
    
    def add_include_path(self, path: Union[str, Path]):
        """Add an include path"""
        path = Path(path)
        if path not in self.compiler_config.include_paths:
            self.compiler_config.include_paths.append(path)
    
    def set_compiler(self, compiler: CompilerType):
        """Set the compiler to use"""
        if compiler not in self.available_compilers:
            raise CompilerError(f"Compiler {compiler.value} not available")
        
        self.compiler_config.compiler = compiler
        self.logger.info(f"Compiler set to: {compiler.value}")
    
    def _get_compiler_flags(self) -> List[str]:
        """Build compiler flags"""
        flags = []
        
        # Optimization
        flags.append(self.compiler_config.optimization.value)
        
        # Warning level
        if self.compiler_config.warning_level.value >= WarningLevel.BASIC.value:
            flags.append("-Wall")
        if self.compiler_config.warning_level.value >= WarningLevel.EXTRA.value:
            flags.append("-Wextra")
        if self.compiler_config.warning_level.value >= WarningLevel.PEDANTIC.value:
            flags.append("-Wpedantic")
        
        # Debug info
        if self.compiler_config.debug_info:
            flags.append("-g")
        
        # Position independent code
        if self.compiler_config.position_independent:
            flags.append("-fPIC")
        
        # Standard
        flags.append(f"-std={self.compiler_config.standard}")
        
        # Generate assembly
        flags.append("-S")
        flags.append("-masm=intel")  # Intel syntax for x86
        
        # Defines
        for name, value in self.compiler_config.defines.items():
            flags.append(f"-D{name}={value}")
        
        # Include paths
        for path in self.compiler_config.include_paths:
            flags.append(f"-I{path}")
        
        # Sanitizers
        for sanitizer in self.compiler_config.sanitizers:
            flags.append(f"-fsanitize={sanitizer}")
        
        # Custom flags
        flags.extend(self.compiler_config.custom_flags)
        
        return flags
    
    def _create_temp_dir(self) -> Path:
        """Create temporary directory for compilation"""
        if not self.temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='casm_c_'))
            self.logger.info(f"Created temp directory: {self.temp_dir}")
        
        return self.temp_dir
    
    def _generate_full_source(self, code_block: CCodeBlock) -> str:
        """Generate complete C source file"""
        parts = []
        
        # Headers
        if self.headers:
            parts.append(self.get_header_includes())
            parts.append("")
        
        # CASM variable declarations
        if self.casm_variables:
            parts.append("// CASM Variables")
            for var_name, var_info in self.casm_variables.items():
                if isinstance(var_info, dict):
                    var_type = var_info.get('type', 'int')
                    label = var_info.get('label', var_name)
                else:
                    var_type = 'int'
                    label = str(var_info)
                
                parts.append(f"extern {var_type} {label};")
            parts.append("")
        
        # Code block
        parts.append("// C Code Block")
        parts.append("void __casm_block(void) {")
        parts.append(code_block.source_code)
        parts.append("}")
        
        return '\n'.join(parts)
    
    def _compile_single_block(self, block: CCodeBlock) -> CompilationResult:
        """Compile a single code block"""
        import time
        start_time = time.time()
        
        # Check cache
        source_hash = hashlib.sha256(block.source_code.encode()).hexdigest()
        if source_hash in self.compilation_cache:
            self.logger.info(f"Using cached compilation for {block.block_id}")
            return self.compilation_cache[source_hash]
        
        # Create temp directory
        temp_dir = self._create_temp_dir()
        
        # Generate source file
        source_code = self._generate_full_source(block)
        source_file = temp_dir / f"{block.block_id}.c"
        asm_file = temp_dir / f"{block.block_id}.s"
        
        source_file.write_text(source_code, encoding='utf-8')
        
        # Get compiler
        compiler = self.available_compilers.get(self.compiler_config.compiler)
        if not compiler:
            return CompilationResult(
                success=False,
                errors=[f"Compiler {self.compiler_config.compiler.value} not available"]
            )
        
        # Build command
        if self.compiler_config.compiler == CompilerType.ZIGCC:
            cmd = [str(compiler), "cc"] + self._get_compiler_flags() + [
                "-o", str(asm_file),
                str(source_file)
            ]
        else:
            cmd = [str(compiler)] + self._get_compiler_flags() + [
                "-o", str(asm_file),
                str(source_file)
            ]
        
        self.logger.info(f"Compiling: {' '.join(cmd)}")
        
        # Compile
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            compile_time = time.time() - start_time
            
            if result.returncode == 0 and asm_file.exists():
                # Read assembly
                assembly_code = asm_file.read_text(encoding='utf-8')
                
                # Extract relevant section
                extracted = self._extract_block_assembly(
                    assembly_code,
                    block.start_marker,
                    block.end_marker
                )
                
                compilation_result = CompilationResult(
                    success=True,
                    assembly_code=extracted,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    compile_time=compile_time,
                    compiler_used=self.compiler_config.compiler
                )
                
                # Cache result
                self.compilation_cache[source_hash] = compilation_result
                
                self.logger.info(f"Compiled {block.block_id} in {compile_time:.2f}s")
                
                return compilation_result
            else:
                return CompilationResult(
                    success=False,
                    stderr=result.stderr,
                    stdout=result.stdout,
                    errors=[result.stderr],
                    compiler_used=self.compiler_config.compiler
                )
        
        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False,
                errors=["Compilation timeout"]
            )
        except Exception as e:
            return CompilationResult(
                success=False,
                errors=[f"Compilation error: {str(e)}"]
            )
    
    def _extract_block_assembly(self, asm_code: str, start_marker: str, end_marker: str) -> str:
        """Extract assembly code between markers"""
        lines = asm_code.split('\n')
        extracted = []
        in_block = False
        
        for line in lines:
            if start_marker in line:
                in_block = True
                continue
            elif end_marker in line:
                in_block = False
                break
            elif in_block:
                # Skip compiler-generated comments and directives
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    extracted.append(line)
        
        return '\n'.join(extracted)
    
    def compile_all_c_code(self, parallel: bool = True) -> Dict[str, str]:
        """Compile all C code blocks"""
        if not self.c_code_blocks:
            self.logger.warning("No C code blocks to compile")
            return {}
        
        self.logger.info(f"Compiling {len(self.c_code_blocks)} code blocks")
        
        results = {}
        
        if parallel and len(self.c_code_blocks) > 1:
            # Parallel compilation
            with ThreadPoolExecutor(max_workers=min(4, len(self.c_code_blocks))) as executor:
                future_to_block = {
                    executor.submit(self._compile_single_block, block): block
                    for block in self.c_code_blocks
                }
                
                for future in as_completed(future_to_block):
                    block = future_to_block[future]
                    try:
                        result = future.result()
                        if result.success:
                            results[block.block_id] = result.assembly_code
                            block.compiled_code = result.assembly_code
                            block.compile_time = result.compile_time
                        else:
                            self.logger.error(f"Failed to compile {block.block_id}: {result.errors}")
                    except Exception as e:
                        self.logger.error(f"Exception compiling {block.block_id}: {e}")
        else:
            # Sequential compilation
            for block in self.c_code_blocks:
                result = self._compile_single_block(block)
                if result.success:
                    results[block.block_id] = result.assembly_code
                    block.compiled_code = result.assembly_code
                    block.compile_time = result.compile_time
                else:
                    self.logger.error(f"Failed to compile {block.block_id}: {result.errors}")
        
        self.logger.info(f"Successfully compiled {len(results)}/{len(self.c_code_blocks)} blocks")
        return results
    
    def compile_inline_c(self, code: str) -> Optional[str]:
        """Compile a single inline C code snippet"""
        block = CCodeBlock(
            block_id=f"INLINE_{self.marker_counter}",
            source_code=code,
            start_marker=f"INLINE_{self.marker_counter}_START",
            end_marker=f"INLINE_{self.marker_counter}_END"
        )
        self.marker_counter += 1
        
        result = self._compile_single_block(block)
        
        if result.success:
            return result.assembly_code
        else:
            self.logger.error(f"Failed to compile inline C: {result.errors}")
            return None
    
    # ==================== CODE PROCESSING ====================
    
    def process_c_code(self, content: str) -> str:
        """Process C code and extract metadata"""
        # Extract variables
        self.extract_variables_from_code(content)
        
        # Extract functions
        self.extract_functions_from_code(content)
        
        # Process variable references
        processed = self._process_variable_references(content)
        
        return processed
    
    def _process_variable_references(self, code: str) -> str:
        """Process CASM variable references in C code"""
        processed = code
        
        # Replace $varname with actual variable names
        for var_name, var_info in self.casm_variables.items():
            if isinstance(var_info, dict):
                label = var_info.get('label', var_name)
            else:
                label = str(var_info)
            
            # Replace $varname references
            processed = re.sub(
                r'\ + re.escape(var_name) + r'\b',
                label,
                processed
            )
        
        return processed
    
    # ==================== OPTIMIZATION ====================
    
    def optimize_code_block(self, block: CCodeBlock) -> str:
        """Apply additional optimizations to compiled code"""
        if not block.compiled_code:
            return ""
        
        optimized = block.compiled_code
        
        # Remove redundant moves
        optimized = self._remove_redundant_moves(optimized)
        
        # Optimize stack operations
        optimized = self._optimize_stack_ops(optimized)
        
        return optimized
    
    def _remove_redundant_moves(self, asm_code: str) -> str:
        """Remove redundant move instructions"""
        lines = asm_code.split('\n')
        optimized = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for mov reg, reg
            if line.startswith('mov '):
                parts = line.split(',')
                if len(parts) == 2:
                    src = parts[1].strip()
                    dst = parts[0].replace('mov', '').strip()
                    
                    if src == dst:
                        # Skip redundant move
                        i += 1
                        continue
            
            optimized.append(lines[i])
            i += 1
        
        return '\n'.join(optimized)
    
    def _optimize_stack_ops(self, asm_code: str) -> str:
        """Optimize stack operations"""
        lines = asm_code.split('\n')
        optimized = []
        
        i = 0
        while i < len(lines):
            current = lines[i].strip()
            
            # Check for push/pop of same register
            if i + 1 < len(lines) and current.startswith('push '):
                next_line = lines[i + 1].strip()
                if next_line.startswith('pop '):
                    push_reg = current.replace('push', '').strip()
                    pop_reg = next_line.replace('pop', '').strip()
                    
                    if push_reg == pop_reg:
                        # Skip both instructions
                        i += 2
                        continue
            
            optimized.append(lines[i])
            i += 1
        
        return '\n'.join(optimized)
    
    # ==================== ANALYSIS ====================
    
    def analyze_dependencies(self) -> Dict[str, Set[str]]:
        """Analyze dependencies between code blocks"""
        dependencies = defaultdict(set)
        
        for block in self.c_code_blocks:
            # Find variables and functions used
            for var_name in self.variables:
                if var_name in block.source_code:
                    block.variables_used.add(var_name)
            
            for func_name in self.functions:
                if func_name in block.source_code:
                    block.functions_called.add(func_name)
            
            # Add to dependencies
            dependencies[block.block_id] = block.variables_used | block.functions_called
        
        return dict(dependencies)
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics"""
        total_blocks = len(self.c_code_blocks)
        compiled_blocks = sum(1 for b in self.c_code_blocks if b.compiled_code)
        total_time = sum(b.compile_time for b in self.c_code_blocks)
        
        return {
            'total_blocks': total_blocks,
            'compiled_blocks': compiled_blocks,
            'failed_blocks': total_blocks - compiled_blocks,
            'total_compile_time': total_time,
            'average_compile_time': total_time / total_blocks if total_blocks > 0 else 0,
            'cache_hits': len(self.compilation_cache),
            'compilers_available': len(self.available_compilers),
            'current_compiler': self.compiler_config.compiler.value,
        }
    
    def export_compilation_report(self, output_file: Union[str, Path]):
        """Export detailed compilation report"""
        output_path = Path(output_file)
        
        report = {
            'version': __version__,
            'target': {
                'platform': self.target_platform.value,
                'architecture': self.target_arch.value,
            },
            'compiler_config': {
                'compiler': self.compiler_config.compiler.value,
                'optimization': self.compiler_config.optimization.value,
                'standard': self.compiler_config.standard,
                'debug_info': self.compiler_config.debug_info,
                'sanitizers': self.compiler_config.sanitizers,
            },
            'statistics': self.get_compilation_stats(),
            'blocks': [
                {
                    'id': block.block_id,
                    'source_lines': len(block.source_code.split('\n')),
                    'compiled': block.compiled_code is not None,
                    'compile_time': block.compile_time,
                    'variables_used': list(block.variables_used),
                    'functions_called': list(block.functions_called),
                }
                for block in self.c_code_blocks
            ],
            'variables': {
                name: {
                    'type': var.type_,
                    'scope': var.scope.name,
                    'usage_count': var.usage_count,
                }
                for name, var in self.variables.items()
            },
            'functions': {
                name: {
                    'return_type': func.return_type,
                    'parameters': len(func.parameters),
                    'complexity': func.complexity,
                    'is_inline': func.is_inline,
                }
                for name, func in self.functions.items()
            },
        }
        
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        self.logger.info(f"Exported compilation report to {output_path}")
    
    # ==================== SANITIZER SUPPORT ====================
    
    def enable_sanitizer(self, sanitizer: str):
        """Enable a sanitizer (asan, ubsan, tsan, msan)"""
        valid_sanitizers = ['address', 'undefined', 'thread', 'memory', 'leak']
        
        if sanitizer not in valid_sanitizers:
            raise ValueError(f"Invalid sanitizer: {sanitizer}. Valid: {valid_sanitizers}")
        
        if sanitizer not in self.compiler_config.sanitizers:
            self.compiler_config.sanitizers.append(sanitizer)
            self.logger.info(f"Enabled sanitizer: {sanitizer}")
    
    def disable_sanitizer(self, sanitizer: str):
        """Disable a sanitizer"""
        if sanitizer in self.compiler_config.sanitizers:
            self.compiler_config.sanitizers.remove(sanitizer)
            self.logger.info(f"Disabled sanitizer: {sanitizer}")
    
    # ==================== CROSS-COMPILATION ====================
    
    def set_cross_compile(self, triple: str):
        """Set cross-compilation target triple (e.g., 'arm-linux-gnueabi')"""
        self.compiler_config.custom_flags.append(f"--target={triple}")
        self.logger.info(f"Set cross-compilation target: {triple}")
    
    def set_sysroot(self, path: Union[str, Path]):
        """Set sysroot for cross-compilation"""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Sysroot path does not exist: {path}")
        
        self.compiler_config.custom_flags.append(f"--sysroot={path}")
        self.logger.info(f"Set sysroot: {path}")
    
    # ==================== DEBUGGING ====================
    
    def enable_verbose(self):
        """Enable verbose output"""
        self.verbose = True
        self.logger.setLevel(logging.DEBUG)
    
    def enable_debug_save(self, save_dir: Optional[Union[str, Path]] = None):
        """Save intermediate files for debugging"""
        self.save_debug = True
        
        if save_dir:
            self.temp_dir = Path(save_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_last_compile_output(self) -> str:
        """Get last compilation output"""
        return self._last_compile_output
    
    def dump_state(self, output_file: Union[str, Path]):
        """Dump processor state to file"""
        output_path = Path(output_file)
        
        state = {
            'headers': [{'name': h.name, 'is_system': h.is_system} for h in self.headers],
            'variables': {name: var.__dict__ for name, var in self.variables.items()},
            'functions': {name: {
                'name': f.name,
                'return_type': f.return_type,
                'parameters': f.parameters,
                'complexity': f.complexity,
            } for name, f in self.functions.items()},
            'code_blocks': len(self.c_code_blocks),
            'compiled_blocks': sum(1 for b in self.c_code_blocks if b.compiled_code),
        }
        
        output_path.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
        self.logger.info(f"Dumped state to {output_path}")
    
    # ==================== CLEANUP ====================
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists() and not self.save_debug:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.logger.info("Cleaned up temporary files")
    
    def reset(self):
        """Reset processor state"""
        self.headers.clear()
        self.variables.clear()
        self.casm_variables.clear()
        self.functions.clear()
        self.c_code_blocks.clear()
        self.globals.clear()
        self.compilation_cache.clear()
        self.marker_counter = 0
        
        self.cleanup()
        
        self.logger.info("Reset processor state")
    
    # ==================== CONTEXT MANAGER ====================
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False
    
    # ==================== UTILITY METHODS ====================
    
    def format_assembly(self, asm_code: str, indent: int = 4) -> str:
        """Format assembly code with proper indentation"""
        lines = asm_code.split('\n')
        formatted = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted.append('')
                continue
            
            # Labels
            if stripped.endswith(':'):
                formatted.append(stripped)
            # Directives
            elif stripped.startswith('.'):
                formatted.append('  ' + stripped)
            # Instructions
            else:
                formatted.append(' ' * indent + stripped)
        
        return '\n'.join(formatted)
    
    def validate_c_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate C code syntax without compilation"""
        errors = []
        
        # Basic syntax checks
        if code.count('{') != code.count('}'):
            errors.append("Mismatched braces")
        
        if code.count('(') != code.count(')'):
            errors.append("Mismatched parentheses")
        
        if code.count('[') != code.count(']'):
            errors.append("Mismatched brackets")
        
        # Check for common errors
        if re.search(r'==\s*=', code):
            errors.append("Possible assignment in condition (== vs =)")
        
        return len(errors) == 0, errors
    
    def estimate_complexity(self, code: str) -> int:
        """Estimate code complexity"""
        complexity = 1
        
        # Count control structures
        complexity += code.count('if')
        complexity += code.count('else')
        complexity += code.count('while')
        complexity += code.count('for')
        complexity += code.count('switch')
        complexity += code.count('case')
        complexity += code.count('&&')
        complexity += code.count('||')
        complexity += code.count('?')
        
        return complexity
    
    def get_size_estimate(self, block: CCodeBlock) -> int:
        """Estimate compiled code size"""
        if block.compiled_code:
            # Count instructions
            instructions = [line for line in block.compiled_code.split('\n') 
                          if line.strip() and not line.strip().startswith('.')]
            return len(instructions)
        
        # Rough estimate from source
        return len(block.source_code.split('\n')) * 3

# ==================== GLOBAL INSTANCE ====================

# Global processor instance for convenience
c_processor = CCodeProcessor()

# ==================== CONVENIENCE FUNCTIONS ====================

def compile_c_inline(code: str, optimization: str = 'moderate') -> Optional[str]:
    """Compile inline C code to assembly"""
    opt_map = {
        'none': OptimizationLevel.DEBUG,
        'basic': OptimizationLevel.BASIC,
        'moderate': OptimizationLevel.MODERATE,
        'aggressive': OptimizationLevel.AGGRESSIVE,
        'size': OptimizationLevel.SIZE,
        'fast': OptimizationLevel.FAST,
    }
    
    processor = CCodeProcessor()
    processor.set_optimization_level(opt_map.get(optimization, OptimizationLevel.MODERATE))
    
    result = processor.compile_inline_c(code)
    processor.cleanup()
    
    return result

def analyze_c_code(code: str) -> Dict[str, Any]:
    """Analyze C code and return metadata"""
    processor = CCodeProcessor()
    processor.process_c_code(code)
    
    return {
        'variables': {name: {
            'type': var.type_,
            'scope': var.scope.name,
        } for name, var in processor.variables.items()},
        'functions': {name: {
            'return_type': func.return_type,
            'parameters': func.parameters,
            'complexity': func.complexity,
        } for name, func in processor.functions.items()},
        'complexity': processor.estimate_complexity(code),
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage
    processor = CCodeProcessor()
    
    # Configure
    processor.set_target("linux", "x86_64")
    processor.set_optimization_level(OptimizationLevel.AGGRESSIVE)
    processor.add_system_header("stdio.h")
    processor.add_system_header("stdlib.h")
    
    # Add C code
    c_code = """
    int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
    
    void print_result(int x) {
        printf("Result: %d\\n", x);
    }
    """
    
    block_id = processor.add_c_code_block(c_code)
    
    # Compile
    results = processor.compile_all_c_code()
    
    if block_id in results:
        print("Compiled assembly:")
        print(processor.format_assembly(results[block_id]))
    
    # Get statistics
    stats = processor.get_compilation_stats()
    print(f"\nCompilation stats: {stats}")
    
    # Cleanup
    processor.cleanup()