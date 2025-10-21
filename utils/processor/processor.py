#!/usr/bin/env python3
from pathlib import Path
import tempfile
import hashlib
import shutil
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, Optional, Any
from .data import CHeader, CCodeBlock, CompilationResult, CompilerConfig
from .enums import CompilerType, OptimizationLevel, WarningLevel, Platform, Architecture
from .detector import CompilerDetector
from .analyzers import VariableAnalyzer, FunctionAnalyzer

__version__ = "2.0.0"


class CCodeProcessor:
    """Advanced C code processor with assembly integration"""
    
    def __init__(self):
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
        
        self.headers: list[CHeader] = []
        self.variables: Dict[str, Any] = {}
        self.casm_variables: Dict[str, Any] = {}
        self.functions: Dict[str, Any] = {}
        self.c_code_blocks: list[CCodeBlock] = []
        self.globals: list[str] = []
        
        self.temp_dir: Optional[Path] = None
        self.available_compilers: Dict[CompilerType, Path] = {}
        self.compilation_cache: Dict[str, CompilationResult] = {}
        self.marker_counter = 0
        
        self.save_debug = False
        self.verbose = False
        self._last_compile_output = ''
        self._last_status = ''
        self._compile_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self._detect_compilers()
    
    def _detect_compilers(self):
        self.available_compilers = CompilerDetector.detect_compilers()
        
        if self.available_compilers:
            self.logger.info(f"Detected compilers: {', '.join(c.value for c in self.available_compilers.keys())}")
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
    
    def add_header(self, header_name: str, is_system: bool = False):
        header_name = header_name.strip()
        if header_name.startswith('<') and header_name.endswith('>'):
            header_name = header_name[1:-1]
            is_system = True
        elif header_name.startswith('"') and header_name.endswith('"'):
            header_name = header_name[1:-1]
            is_system = False

        for header in self.headers:
            if header.name == header_name:
                return

        header = CHeader(name=header_name, is_system=is_system)
        self.headers.append(header)
        self.logger.info(f"Added header: {'<' if is_system else ''}{header_name}{'>' if is_system else ''}")
    
    def add_system_header(self, header_name: str):
        self.add_header(header_name, is_system=True)
    
    def get_header_includes(self) -> str:
        includes = []
        for header in self.headers:
            if header.is_system:
                includes.append(f"#include <{header.name}>")
            else:
                includes.append(f'#include "{header.name}"')
        return '\n'.join(includes)
    
    def set_casm_variables(self, variables: Dict[str, Any]):
        self.casm_variables = variables
        self.logger.info(f"Set {len(variables)} CASM variables")
    
    def add_variable(self, var: Any):
        self.variables[var.name] = var
        self.logger.info(f"Added variable: {var.name} ({var.type_})")
    
    def get_variable(self, name: str) -> Optional[Any]:
        return self.variables.get(name)
    
    def get_c_variables(self) -> Dict[str, Any]:
        return self.variables
    
    def extract_variables_from_code(self, code: str):
        variables = VariableAnalyzer.extract_variables(code)
        for var in variables:
            if var.name not in self.variables:
                self.add_variable(var)
    
    def add_function(self, func: Any):
        self.functions[func.name] = func
        func.complexity = FunctionAnalyzer.calculate_complexity(func)
        self.logger.info(f"Added function: {func.name} (complexity: {func.complexity})")
    
    def extract_functions_from_code(self, code: str):
        functions = FunctionAnalyzer.extract_functions(code)
        for func in functions:
            if func.name not in self.functions:
                self.add_function(func)
    
    def add_c_code_block(self, code: str, optimization: Optional[OptimizationLevel] = None) -> str:
        block_id = f"CASM_BLOCK_{self.marker_counter}"
        self.marker_counter += 1
        start_marker = f"{block_id}_START"
        end_marker = f"{block_id}_END"
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
    
    def set_target(self, platform: str, arch: str = 'x86_64'):
        self.target_platform = Platform(platform.lower())
        self.target_arch = Architecture(arch.lower())
        self.logger.info(f"Target set to: {self.target_platform.value}/{self.target_arch.value}")
        best = CompilerDetector.select_best_compiler(
            self.target_platform,
            self.target_arch,
            self.available_compilers
        )
        if best:
            self.compiler_config.compiler = best
    
    def set_optimization_level(self, level: OptimizationLevel):
        self.compiler_config.optimization = level
        self.logger.info(f"Optimization level set to: {level.value}")
    
    def add_define(self, name: str, value: str = "1"):
        self.compiler_config.defines[name] = value
    
    def add_include_path(self, path: str):
        p = Path(path)
        if p not in self.compiler_config.include_paths:
            self.compiler_config.include_paths.append(p)
    
    def set_compiler(self, compiler: CompilerType):
        if compiler not in self.available_compilers:
            raise RuntimeError(f"Compiler {compiler.value} not available")
        self.compiler_config.compiler = compiler
        self.logger.info(f"Compiler set to: {compiler.value}")
    
    def _get_compiler_flags(self) -> list[str]:
        flags = []
        flags.append(self.compiler_config.optimization.value)
        if self.compiler_config.warning_level.value >= WarningLevel.BASIC.value:
            flags.append("-Wall")
        if self.compiler_config.warning_level.value >= WarningLevel.EXTRA.value:
            flags.append("-Wextra")
        if self.compiler_config.warning_level.value >= WarningLevel.PEDANTIC.value:
            flags.append("-Wpedantic")
        if self.compiler_config.debug_info:
            flags.append("-g")
        if self.compiler_config.position_independent:
            flags.append("-fPIC")
        flags.append(f"-std={self.compiler_config.standard}")
        flags.append("-S")
        flags.append("-masm=intel")
        for name, value in self.compiler_config.defines.items():
            flags.append(f"-D{name}={value}")
        for path in self.compiler_config.include_paths:
            flags.append(f"-I{path}")
        for sanitizer in self.compiler_config.sanitizers:
            flags.append(f"-fsanitize={sanitizer}")
        flags.extend(self.compiler_config.custom_flags)
        return flags
    
    def _create_temp_dir(self) -> Path:
        if not self.temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='casm_c_'))
            self.logger.info(f"Created temp directory: {self.temp_dir}")
        return self.temp_dir
    
    def _generate_full_source(self, code_block: CCodeBlock) -> str:
        parts = []
        if self.headers:
            parts.append(self.get_header_includes())
            parts.append("")
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
        parts.append("// C Code Block")
        parts.append("void __casm_block(void) {")
        parts.append(code_block.source_code)
        parts.append("}")
        return '\n'.join(parts)
    
    def _compile_single_block(self, block: CCodeBlock) -> CompilationResult:
        import time
        start_time = time.time()
        source_hash = hashlib.sha256(block.source_code.encode()).hexdigest()
        if source_hash in self.compilation_cache:
            self.logger.info(f"Using cached compilation for {block.block_id}")
            return self.compilation_cache[source_hash]
        temp_dir = self._create_temp_dir()
        source_code = self._generate_full_source(block)
        source_file = temp_dir / f"{block.block_id}.c"
        asm_file = temp_dir / f"{block.block_id}.s"
        source_file.write_text(source_code, encoding='utf-8')
        compiler = self.available_compilers.get(self.compiler_config.compiler)
        if not compiler:
            return CompilationResult(success=False, errors=[f"Compiler {self.compiler_config.compiler.value} not available"])
        if self.compiler_config.compiler == CompilerType.ZIGCC:
            cmd = [str(compiler), "cc"] + self._get_compiler_flags() + ["-o", str(asm_file), str(source_file)]
        else:
            cmd = [str(compiler)] + self._get_compiler_flags() + ["-o", str(asm_file), str(source_file)]
        self.logger.info(f"Compiling: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            compile_time = time.time() - start_time
            if result.returncode == 0 and asm_file.exists():
                assembly_code = asm_file.read_text(encoding='utf-8')
                extracted = self._extract_block_assembly(assembly_code, block.start_marker, block.end_marker)
                compilation_result = CompilationResult(success=True, assembly_code=extracted, stdout=result.stdout, stderr=result.stderr, compile_time=compile_time, compiler_used=self.compiler_config.compiler)
                self.compilation_cache[source_hash] = compilation_result
                self.logger.info(f"Compiled {block.block_id} in {compile_time:.2f}s")
                return compilation_result
            else:
                return CompilationResult(success=False, stderr=result.stderr, stdout=result.stdout, errors=[result.stderr], compiler_used=self.compiler_config.compiler)
        except subprocess.TimeoutExpired:
            return CompilationResult(success=False, errors=["Compilation timeout"])
        except Exception as e:
            return CompilationResult(success=False, errors=[f"Compilation error: {str(e)}"])
    
    def _extract_block_assembly(self, asm_code: str, start_marker: str, end_marker: str) -> str:
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
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    extracted.append(line)
        return '\n'.join(extracted)
    
    def compile_all_c_code(self, parallel: bool = True) -> Dict[str, str]:
        if not self.c_code_blocks:
            self.logger.warning("No C code blocks to compile")
            return {}
        self.logger.info(f"Compiling {len(self.c_code_blocks)} code blocks")
        results = {}
        if parallel and len(self.c_code_blocks) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(self.c_code_blocks))) as executor:
                future_to_block = {executor.submit(self._compile_single_block, block): block for block in self.c_code_blocks}
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
        block = CCodeBlock(block_id=f"INLINE_{self.marker_counter}", source_code=code, start_marker=f"INLINE_{self.marker_counter}_START", end_marker=f"INLINE_{self.marker_counter}_END")
        self.marker_counter += 1
        result = self._compile_single_block(block)
        if result.success:
            return result.assembly_code
        else:
            self.logger.error(f"Failed to compile inline C: {result.errors}")
            return None
    
    def process_c_code(self, content: str) -> str:
        self.extract_variables_from_code(content)
        self.extract_functions_from_code(content)
        processed = self._process_variable_references(content)
        return processed
    
    def _process_variable_references(self, code: str) -> str:
        processed = code
        for var_name, var_info in self.casm_variables.items():
            if isinstance(var_info, dict):
                label = var_info.get('label', var_name)
            else:
                label = str(var_info)
            processed = processed.replace(f"${var_name}", label)
        return processed
    
    def optimize_code_block(self, block: CCodeBlock) -> str:
        if not block.compiled_code:
            return ""
        optimized = block.compiled_code
        optimized = self._remove_redundant_moves(optimized)
        optimized = self._optimize_stack_ops(optimized)
        return optimized
    
    def _remove_redundant_moves(self, asm_code: str) -> str:
        lines = asm_code.split('\n')
        optimized = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('mov '):
                parts = line.split(',')
                if len(parts) == 2:
                    src = parts[1].strip()
                    dst = parts[0].replace('mov', '').strip()
                    if src == dst:
                        i += 1
                        continue
            optimized.append(lines[i])
            i += 1
        return '\n'.join(optimized)
    
    def _optimize_stack_ops(self, asm_code: str) -> str:
        lines = asm_code.split('\n')
        optimized = []
        i = 0
        while i < len(lines):
            current = lines[i].strip()
            if i + 1 < len(lines) and current.startswith('push '):
                next_line = lines[i + 1].strip()
                if next_line.startswith('pop '):
                    push_reg = current.replace('push', '').strip()
                    pop_reg = next_line.replace('pop', '').strip()
                    if push_reg == pop_reg:
                        i += 2
                        continue
            optimized.append(lines[i])
            i += 1
        return '\n'.join(optimized)
    
    def analyze_dependencies(self) -> Dict[str, set]:
        dependencies = defaultdict(set)
        for block in self.c_code_blocks:
            for var_name in self.variables:
                if var_name in block.source_code:
                    block.variables_used.add(var_name)
            for func_name in self.functions:
                if func_name in block.source_code:
                    block.functions_called.add(func_name)
            dependencies[block.block_id] = block.variables_used | block.functions_called
        return dict(dependencies)
    
    def get_compilation_stats(self) -> Dict[str, Any]:
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
    
    def export_compilation_report(self, output_file: str):
        output_path = Path(output_file)
        import json
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
    
    def enable_sanitizer(self, sanitizer: str):
        valid_sanitizers = ['address', 'undefined', 'thread', 'memory', 'leak']
        if sanitizer not in valid_sanitizers:
            raise ValueError(f"Invalid sanitizer: {sanitizer}. Valid: {valid_sanitizers}")
        if sanitizer not in self.compiler_config.sanitizers:
            self.compiler_config.sanitizers.append(sanitizer)
            self.logger.info(f"Enabled sanitizer: {sanitizer}")
    
    def disable_sanitizer(self, sanitizer: str):
        if sanitizer in self.compiler_config.sanitizers:
            self.compiler_config.sanitizers.remove(sanitizer)
            self.logger.info(f"Disabled sanitizer: {sanitizer}")
    
    def set_cross_compile(self, triple: str):
        self.compiler_config.custom_flags.append(f"--target={triple}")
        self.logger.info(f"Set cross-compilation target: {triple}")
    
    def set_sysroot(self, path: str):
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Sysroot path does not exist: {p}")
        self.compiler_config.custom_flags.append(f"--sysroot={p}")
        self.logger.info(f"Set sysroot: {p}")
    
    def enable_verbose(self):
        self.verbose = True
        self.logger.setLevel(logging.DEBUG)
    
    def enable_debug_save(self, save_dir: Optional[str] = None):
        self.save_debug = True
        if save_dir:
            self.temp_dir = Path(save_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_last_compile_output(self) -> str:
        return self._last_compile_output
    
    def dump_state(self, output_file: str):
        output_path = Path(output_file)
        import json
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
    
    def cleanup(self):
        if self.temp_dir and self.temp_dir.exists() and not self.save_debug:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.logger.info("Cleaned up temporary files")
    
    def reset(self):
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
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def format_assembly(self, asm_code: str, indent: int = 4) -> str:
        lines = asm_code.split('\n')
        formatted = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted.append('')
                continue
            if stripped.endswith(':'):
                formatted.append(stripped)
            elif stripped.startswith('.'):
                formatted.append('  ' + stripped)
            else:
                formatted.append(' ' * indent + stripped)
        return '\n'.join(formatted)
    
    def validate_c_syntax(self, code: str):
        errors = []
        if code.count('{') != code.count('}'):
            errors.append("Mismatched braces")
        if code.count('(') != code.count(')'):
            errors.append("Mismatched parentheses")
        if code.count('[') != code.count(']'):
            errors.append("Mismatched brackets")
        if re.search(r'==\s*=', code):
            errors.append("Possible assignment in condition (== vs =)")
        return len(errors) == 0, errors
    
    def estimate_complexity(self, code: str) -> int:
        complexity = 1
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
        if block.compiled_code:
            instructions = [line for line in block.compiled_code.split('\n') if line.strip() and not line.strip().startswith('.')]
            return len(instructions)
        return len(block.source_code.split('\n')) * 3

# Global processor instance
c_processor = CCodeProcessor()

def compile_c_inline(code: str, optimization: str = 'moderate') -> Optional[str]:
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
    processor = CCodeProcessor()
    processor.process_c_code(code)
    return {
        'variables': {name: {'type': var.type_, 'scope': var.scope.name} for name, var in processor.variables.items()},
        'functions': {name: {'return_type': func.return_type, 'parameters': func.parameters, 'complexity': func.complexity} for name, func in processor.functions.items()},
        'complexity': processor.estimate_complexity(code),
    }
