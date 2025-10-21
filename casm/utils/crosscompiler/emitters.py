from abc import ABC, abstractmethod
from typing import List
from .ir import IRInstruction, IROperand
from .enums import IROpcode, Architecture
from .register_mapper import RegisterMapper

class ArchitectureEmitter(ABC):
    @abstractmethod
    def emit_instruction(self, ir_inst: IRInstruction) -> List[str]:
        pass
    
    @abstractmethod
    def emit_prologue(self, func_name: str, stack_size: int) -> List[str]:
        pass
    
    @abstractmethod
    def emit_epilogue(self) -> List[str]:
        pass

class X86_64Emitter(ArchitectureEmitter):
    OPCODE_MAP = {
        IROpcode.MOVE: 'mov',
        IROpcode.ADD: 'add',
        IROpcode.SUB: 'sub',
        IROpcode.MUL: 'imul',
        IROpcode.DIV: 'idiv',
        IROpcode.AND: 'and',
        IROpcode.OR: 'or',
        IROpcode.XOR: 'xor',
        IROpcode.NOT: 'not',
        IROpcode.SHL: 'shl',
        IROpcode.SHR: 'shr',
        IROpcode.SAR: 'asr',
        IROpcode.CMP: 'cmp',
        IROpcode.TEST: 'tst',
        IROpcode.JMP: 'b',
        IROpcode.JE: 'beq',
        IROpcode.JNE: 'bne',
        IROpcode.JL: 'blt',
        IROpcode.JLE: 'ble',
        IROpcode.JG: 'bgt',
        IROpcode.JGE: 'bge',
        IROpcode.CALL: 'bl',
        IROpcode.RET: 'ret',
        IROpcode.LOAD: 'ldr',
        IROpcode.STORE: 'str',
        IROpcode.NOP: 'nop',
        IROpcode.SYSCALL: 'svc',
    }
    
    def emit_instruction(self, ir_inst: IRInstruction) -> List[str]:
        mnemonic = self.OPCODE_MAP.get(ir_inst.opcode)
        if not mnemonic:
            return [f"; Unknown IR opcode: {ir_inst.opcode}"]
        operands = []
        for op in ir_inst.operands:
            operands.append(self._emit_operand(op))
        if operands:
            return [f"    {mnemonic} {', '.join(operands)}"]
        else:
            return [f"    {mnemonic}"]
    
    def _emit_operand(self, op: IROperand) -> str:
        if op.type == "reg":
            return RegisterMapper.from_virtual(Architecture.ARM64, str(op.value))
        elif op.type == "imm":
            return str(op.value)
        elif op.type == "mem":
            return f"[{op.value}]"
        elif op.type == "label":
            return str(op.value)
        return str(op.value)
    
    def emit_prologue(self, func_name: str, stack_size: int) -> List[str]:
        aligned_size = (stack_size + 15) & ~15
        return [
            f"{func_name}:",
            "    stp x29, x30, [sp, #-16]!",
            "    mov x29, sp",
            f"    sub sp, sp, #{aligned_size}" if aligned_size > 0 else "",
        ]
    
    def emit_epilogue(self) -> List[str]:
        return [
            "    mov sp, x29",
            "    ldp x29, x30, [sp], #16",
            "    ret",
        ]

class RISCVEmitter(ArchitectureEmitter):
    OPCODE_MAP = {
        IROpcode.MOVE: 'mv',
        IROpcode.ADD: 'add',
        IROpcode.SUB: 'sub',
        IROpcode.MUL: 'mul',
        IROpcode.DIV: 'div',
        IROpcode.AND: 'and',
        IROpcode.OR: 'or',
        IROpcode.XOR: 'xor',
        IROpcode.NOT: 'not',
        IROpcode.SHL: 'sll',
        IROpcode.SHR: 'srl',
        IROpcode.SAR: 'sra',
        IROpcode.JMP: 'j',
        IROpcode.JE: 'beq',
        IROpcode.JNE: 'bne',
        IROpcode.JL: 'blt',
        IROpcode.JGE: 'bge',
        IROpcode.CALL: 'jal',
        IROpcode.RET: 'ret',
        IROpcode.LOAD: 'ld',
        IROpcode.STORE: 'sd',
        IROpcode.NOP: 'nop',
        IROpcode.SYSCALL: 'ecall',
    }
    
    def emit_instruction(self, ir_inst: IRInstruction) -> List[str]:
        mnemonic = self.OPCODE_MAP.get(ir_inst.opcode)
        if not mnemonic:
            return [f"# Unknown IR opcode: {ir_inst.opcode}"]
        if ir_inst.opcode == IROpcode.CMP:
            operands = [self._emit_operand(op) for op in ir_inst.operands]
            return [f"    sub zero, {operands[0]}, {operands[1]}"]
        operands = []
        for op in ir_inst.operands:
            operands.append(self._emit_operand(op))
        if operands:
            return [f"    {mnemonic} {', '.join(operands)}"]
        else:
            return [f"    {mnemonic}"]
    
    def _emit_operand(self, op: IROperand) -> str:
        if op.type == "reg":
            return RegisterMapper.from_virtual(Architecture.RISCV64, str(op.value))
        elif op.type == "imm":
            return str(op.value)
        elif op.type == "mem":
            if '+' in str(op.value):
                parts = str(op.value).split('+')
                return f"{parts[1]}({parts[0]})"
            return f"0({op.value})"
        elif op.type == "label":
            return str(op.value)
        return str(op.value)
    
    def emit_prologue(self, func_name: str, stack_size: int) -> List[str]:
        aligned_size = (stack_size + 15) & ~15
        return [
            f"{func_name}:",
            f"    addi sp, sp, -{aligned_size + 16}",
            f"    sd ra, {aligned_size + 8}(sp)",
            f"    sd s0, {aligned_size}(sp)",
            f"    addi s0, sp, {aligned_size + 16}",
        ]
    
    def emit_epilogue(self) -> List[str]:
        return [
            "    ld ra, 8(sp)",
            "    ld s0, 0(sp)",
            "    addi sp, sp, 16",
            "    ret",
        ]
