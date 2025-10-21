from typing import List, Dict, Any
import logging
from .enums import Architecture, IROpcode
from .ir import IRInstruction, IROperand
from .parsers import X86_64Parser, ARM64Parser, RISCVParser
from .emitters import X86_64Emitter, ARM64Emitter, RISCVEmitter

class UniversalCrossCompiler:
    def __init__(self, source_arch: Architecture, target_arch: Architecture, optimization_level: int = 2):
        self.source_arch = source_arch
        self.target_arch = target_arch
        self.optimization_level = optimization_level
        self.parser = self._create_parser(source_arch)
        self.emitter = self._create_emitter(target_arch)
        self.ir_instructions: List[IRInstruction] = []
        self.labels: Dict[str, int] = {}
        self.functions: Dict[str, tuple] = {}
        self.stats = {'source_instructions': 0, 'ir_instructions': 0, 'target_instructions': 0, 'optimizations_applied': 0}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _create_parser(self, arch: Architecture):
        parsers = {Architecture.X86_64: X86_64Parser, Architecture.ARM64: ARM64Parser, Architecture.RISCV64: RISCVParser}
        parser_class = parsers.get(arch)
        if not parser_class:
            raise ValueError(f"No parser for architecture: {arch}")
        return parser_class()

    def _create_emitter(self, arch: Architecture):
        emitters = {Architecture.X86_64: X86_64Emitter, Architecture.ARM64: ARM64Emitter, Architecture.RISCV64: RISCVEmitter}
        emitter_class = emitters.get(arch)
        if not emitter_class:
            raise ValueError(f"No emitter for architecture: {arch}")
        return emitter_class()

    def parse_source(self, source: str) -> List[IRInstruction]:
        self.ir_instructions.clear()
        self.labels.clear()
        lines = source.split('\n')
        self.stats['source_instructions'] = len([l for l in lines if l.strip()])
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                label = line[:-1]
                self.labels[label] = len(self.ir_instructions)
                continue
            ir_inst = self.parser.parse_instruction(line)
            if ir_inst:
                self.ir_instructions.append(ir_inst)
        self.stats['ir_instructions'] = len(self.ir_instructions)
        self.logger.info(f"Parsed {self.stats['source_instructions']} source instructions to {self.stats['ir_instructions']} IR instructions")
        return self.ir_instructions

    def optimize_ir(self):
        if self.optimization_level == 0:
            return
        optimizations_applied = 0
        if self.optimization_level >= 1:
            optimizations_applied += self._remove_redundant_moves()
        if self.optimization_level >= 2:
            optimizations_applied += self._constant_folding()
        if self.optimization_level >= 3:
            optimizations_applied += self._eliminate_dead_code()
        self.stats['optimizations_applied'] = optimizations_applied
        self.logger.info(f"Applied {optimizations_applied} optimizations")

    def _remove_redundant_moves(self) -> int:
        count = 0
        optimized = []
        for inst in self.ir_instructions:
            if inst.opcode == IROpcode.MOVE and len(inst.operands) == 2:
                src = inst.operands[1]
                dst = inst.operands[0]
                if src.type == "reg" and dst.type == "reg" and src.value == dst.value:
                    count += 1
                    continue
            optimized.append(inst)
        self.ir_instructions = optimized
        return count

    def _constant_folding(self) -> int:
        count = 0
        optimized = []
        for inst in self.ir_instructions:
            if inst.opcode in [IROpcode.ADD, IROpcode.SUB, IROpcode.MUL]:
                if len(inst.operands) >= 2:
                    op1 = inst.operands[1] if len(inst.operands) > 1 else None
                    op2 = inst.operands[2] if len(inst.operands) > 2 else None
                    if op1 and op2 and op1.type == "imm" and op2.type == "imm":
                        if inst.opcode == IROpcode.ADD:
                            result = op1.value + op2.value
                        elif inst.opcode == IROpcode.SUB:
                            result = op1.value - op2.value
                        elif inst.opcode == IROpcode.MUL:
                            result = op1.value * op2.value
                        else:
                            result = 0
                        new_inst = IRInstruction(opcode=IROpcode.MOVE, operands=[inst.operands[0], IROperand(type="imm", value=result)])
                        optimized.append(new_inst)
                        count += 1
                        continue
            optimized.append(inst)
        self.ir_instructions = optimized
        return count

    def _eliminate_dead_code(self) -> int:
        count = 0
        reachable = set()
        i = 0
        while i < len(self.ir_instructions):
            reachable.add(i)
            inst = self.ir_instructions[i]
            if inst.opcode == IROpcode.JMP:
                if inst.operands and inst.operands[0].type == "label":
                    label = inst.operands[0].value
                    if label in self.labels:
                        i = self.labels[label]
                        continue
            elif inst.opcode == IROpcode.RET:
                break
            i += 1
        optimized = [inst for i, inst in enumerate(self.ir_instructions) if i in reachable]
        count = len(self.ir_instructions) - len(optimized)
        self.ir_instructions = optimized
        return count

    def emit_target(self) -> str:
        output = []
        output.append(f"; Translated from {self.source_arch.value} to {self.target_arch.value}")
        output.append(f"; IR instructions: {len(self.ir_instructions)}")
        output.append("")
        label_positions = {pos: label for label, pos in self.labels.items()}
        for i, ir_inst in enumerate(self.ir_instructions):
            if i in label_positions:
                output.append(f"{label_positions[i]}:")
            asm_lines = self.emitter.emit_instruction(ir_inst)
            output.extend(asm_lines)
        result = '\n'.join(output)
        self.stats['target_instructions'] = len([l for l in output if l.strip() and not l.strip().startswith(';')])
        return result

    def translate(self, source: str) -> str:
        self.logger.info(f"Translating {self.source_arch.value} -> {self.target_arch.value}")
        self.parse_source(source)
        self.optimize_ir()
        result = self.emit_target()
        self.logger.info(f"Translation complete: {self.stats}")
        return result

    def get_statistics(self) -> Dict[str, Any]:
        return {'source_arch': self.source_arch.value, 'target_arch': self.target_arch.value, 'optimization_level': self.optimization_level, **self.stats}

    def export_ir(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f"; IR for {self.source_arch.value} -> {self.target_arch.value}\n")
            f.write(f"; {len(self.ir_instructions)} instructions\n\n")
            for i, inst in enumerate(self.ir_instructions):
                f.write(f"{i:4d}: {inst}\n")
        self.logger.info(f"Exported IR to {filename}")

def translate_assembly(source: str, source_arch: str, target_arch: str, optimization: int = 2) -> str:
    src_arch = Architecture(source_arch.lower())
    tgt_arch = Architecture(target_arch.lower())
    compiler = UniversalCrossCompiler(src_arch, tgt_arch, optimization)
    return compiler.translate(source)

def translate_file(input_file: str, output_file: str, source_arch: str, target_arch: str, optimization: int = 2):
    with open(input_file, 'r') as f:
        source = f.read()
    result = translate_assembly(source, source_arch, target_arch, optimization)
    with open(output_file, 'w') as f:
        f.write(result)
    print(f"Translated {input_file} -> {output_file}")
