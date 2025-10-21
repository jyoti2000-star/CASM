import re
import logging
from pathlib import Path
from typing import List, Optional
from .enums import Platform, Architecture, OptimizationLevel, SectionType
from .data_structures import Variable, Function, Macro, Loop, Instruction
from .calling import CallingConvention
from .syscalls import Syscalls
from .simd import SIMDTranslator
from .analyzer import InstructionAnalyzer
from .cfg import ControlFlowGraph
from .optimizers import PeepholeOptimizer, LoopOptimizer, DeadCodeEliminator, InstructionScheduler
from .exceptions import TranspileError

__version__ = "3.0.0"

class AssemblyTranspiler:
    def __init__(
        self,
        target: str = "linux",
        arch: str = "x86_64",
        opt_level: OptimizationLevel = OptimizationLevel.BASIC,
        enable_simd: bool = True,
        enable_loop_unroll: bool = True,
        unroll_factor: int = 4
    ):
        self.platform = Platform(target.lower())
        self.arch = Architecture(arch.lower())
        self.opt_level = opt_level
        self.enable_simd = enable_simd
        self.enable_loop_unroll = enable_loop_unroll
        self.unroll_factor = unroll_factor
        
        self.variables = {}
        self.functions = {}
        self.macros = {}
        self.loops = []
        
        self.data_section = []
        self.text_section = []
        self.bss_section = []
        self.rodata_section = []
        
        self.externs = set()
        self.globals = set()
        
        self.calling_conv = CallingConvention.get_convention(self.platform, self.arch)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def transpile(self, source: str) -> str:
        self.logger.info(f"Transpiling for {self.platform.value}/{self.arch.value}")
        lines = source.splitlines()
        self._parse_source(lines)
        if self.opt_level.value > 0:
            self._optimize()
        output = self._generate_output()
        self.logger.info("Transpilation complete")
        return output
    
    def _parse_source(self, lines: List[str]):
        current_section = SectionType.TEXT
        current_function = None
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(';') or stripped.startswith('#'):
                continue
            if stripped.startswith('.'):
                current_section = self._parse_directive(stripped)
                continue
            if stripped.endswith(':'):
                label = stripped[:-1]
                if current_section == SectionType.TEXT:
                    current_function = self._create_function(label)
                continue
            if current_section == SectionType.TEXT:
                self.text_section.append(line)
            elif current_section == SectionType.DATA:
                self.data_section.append(line)
            elif current_section == SectionType.BSS:
                self.bss_section.append(line)
            elif current_section == SectionType.RODATA:
                self.rodata_section.append(line)
            if stripped.upper().startswith('UPRINT '):
                expanded = self._expand_uprint_macro(stripped)
                self.text_section.extend(expanded)
    
    def _parse_directive(self, directive: str) -> SectionType:
        directive = directive.lower()
        if directive.startswith('.text'):
            return SectionType.TEXT
        elif directive.startswith('.data'):
            return SectionType.DATA
        elif directive.startswith('.bss'):
            return SectionType.BSS
        elif directive.startswith('.rodata'):
            return SectionType.RODATA
        elif directive.startswith('.extern') or directive.startswith('.global'):
            parts = directive.split()
            if len(parts) > 1:
                if directive.startswith('.extern'):
                    self.externs.add(parts[1])
                else:
                    self.globals.add(parts[1])
        return SectionType.TEXT
    
    def _create_function(self, name: str) -> Function:
        func = Function(name=name, params=[], is_global=name in self.globals)
        self.functions[name] = func
        return func
    
    def _expand_uprint_macro(self, line: str) -> List[str]:
        match = re.match(r'UPRINT\s+(\w+)', line, re.IGNORECASE)
        if not match:
            return [line]
        label = match.group(1)
        if self.platform == Platform.LINUX:
            return [
                f"    ; UPRINT {label} -> puts",
                f"    lea {self.calling_conv['int_args'][0]}, [rel {label}]",
                f"    call puts"
            ]
        elif self.platform == Platform.WINDOWS:
            return [
                f"    ; UPRINT {label} -> printf",
                f"    lea {self.calling_conv['int_args'][0]}, [rel {label}]",
                f"    call printf"
            ]
        elif self.platform == Platform.MACOS:
            return [
                f"    ; UPRINT {label} -> puts",
                f"    lea {self.calling_conv['int_args'][0]}, [rel {label}]",
                f"    call _puts"
            ]
        return [line]
    
    def _optimize(self):
        self.logger.info(f"Applying optimization level: {self.opt_level.name}")
        if self.opt_level.value >= OptimizationLevel.BASIC.value:
            self.text_section = PeepholeOptimizer.optimize(self.text_section)
        if self.opt_level.value >= OptimizationLevel.MODERATE.value:
            instructions = []
            for line_num, line in enumerate(self.text_section):
                inst = InstructionAnalyzer.parse_instruction(line, line_num)
                if inst:
                    instructions.append(inst)
            cfg = ControlFlowGraph()
            cfg.build_from_instructions(instructions)
            cfg.compute_dominators()
            dead_blocks = DeadCodeEliminator.eliminate(cfg)
            self.logger.info(f"Eliminated {len(dead_blocks)} dead blocks")
            self.loops = cfg.identify_loops()
            self.logger.info(f"Identified {len(self.loops)} loops")
        if self.opt_level.value >= OptimizationLevel.AGGRESSIVE.value:
            if self.enable_loop_unroll and self.loops:
                for loop in self.loops:
                    self.text_section = LoopOptimizer.unroll_loop(loop, self.text_section, self.unroll_factor)
                self.logger.info(f"Unrolled {len(self.loops)} loops")
            if self.enable_simd:
                for i, line in enumerate(self.text_section):
                    self.text_section[i] = SIMDTranslator.translate_sse_to_avx(line)
        if self.opt_level.value >= OptimizationLevel.EXTREME.value:
            instructions = []
            for line_num, line in enumerate(self.text_section):
                inst = InstructionAnalyzer.parse_instruction(line, line_num)
                if inst:
                    instructions.append(inst)
            scheduled = InstructionScheduler.schedule(instructions)
            self.text_section = [inst.original_line for inst in scheduled]
            self.logger.info("Applied instruction scheduling")
    
    def _generate_output(self) -> str:
        output = []
        output.append(f"; Generated by Assembly Transpiler v{__version__}")
        output.append(f"; Target: {self.platform.value}/{self.arch.value}")
        output.append(f"; Optimization: {self.opt_level.name}")
        output.append("")
        if self.platform == Platform.WINDOWS:
            output.append("default rel")
        if self.externs:
            output.append("; External symbols")
            for ext in sorted(self.externs):
                prefix = "" if self.platform == Platform.LINUX else "_"
                output.append(f"extern {prefix}{ext}")
            output.append("")
        if self.globals:
            output.append("; Global symbols")
            for glob in sorted(self.globals):
                output.append(f"global {glob}")
            output.append("")
        if self.data_section:
            output.append("section .data")
            output.extend(self.data_section)
            output.append("")
        if self.rodata_section:
            output.append("section .rodata")
            output.extend(self.rodata_section)
            output.append("")
        if self.bss_section:
            output.append("section .bss")
            output.extend(self.bss_section)
            output.append("")
        output.append("section .text")
        output.extend(self.text_section)
        return '\n'.join(output)

def transpile_text(
    text: str,
    target: str = 'linux',
    arch: str = 'x86_64',
    opt_level: str = 'basic',
    enable_simd: bool = True,
    enable_loop_unroll: bool = True,
    unroll_factor: int = 4
) -> str:
    opt_map = {
        'none': OptimizationLevel.NONE,
        'basic': OptimizationLevel.BASIC,
        'moderate': OptimizationLevel.MODERATE,
        'aggressive': OptimizationLevel.AGGRESSIVE,
        'extreme': OptimizationLevel.EXTREME
    }
    opt = opt_map.get(opt_level.lower(), OptimizationLevel.BASIC)
    transpiler = AssemblyTranspiler(
        target=target,
        arch=arch,
        opt_level=opt,
        enable_simd=enable_simd,
        enable_loop_unroll=enable_loop_unroll,
        unroll_factor=unroll_factor
    )
    return transpiler.transpile(text)

def transpile_file(
    path: str,
    target: str = 'linux',
    arch: str = 'x86_64',
    opt_level: str = 'basic',
    output: Optional[str] = None
) -> str:
    input_path = Path(path)
    if not input_path.exists():
        raise TranspileError(f"File not found: {path}")
    source = input_path.read_text(encoding='utf-8')
    result = transpile_text(source, target, arch, opt_level)
    if output:
        output_path = Path(output)
        output_path.write_text(result, encoding='utf-8')
        print(f"Output written to: {output}")
    return result

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Advanced Assembly Transpiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input', help='Input assembly file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-t', '--target', default='linux', choices=['linux', 'windows', 'macos', 'bsd'])
    parser.add_argument('-a', '--arch', default='x86_64', choices=['x86_64', 'x86', 'arm64', 'riscv64'])
    parser.add_argument('-O', '--optimize', dest='opt_level', default='basic', choices=['none', 'basic', 'moderate', 'aggressive', 'extreme'])
    parser.add_argument('--no-simd', action='store_true')
    parser.add_argument('--no-unroll', action='store_true')
    parser.add_argument('--unroll-factor', type=int, default=4)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    try:
        result = transpile_file(args.input, target=args.target, arch=args.arch, opt_level=args.opt_level, output=args.output)
        if not args.output:
            print(result)
        return 0
    except TranspileError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2

if __name__ == '__main__':
    import sys
    raise SystemExit(main())
