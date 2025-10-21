import re
from typing import Optional, List
from abc import ABC, abstractmethod
from .ir import IROperand, IRInstruction
from .enums import IROpcode, Architecture
from .register_mapper import RegisterMapper

class ArchitectureParser(ABC):
    @abstractmethod
    def parse_instruction(self, line: str) -> Optional[IRInstruction]:
        pass
    
    @abstractmethod
    def get_syntax_type(self) -> str:
        pass

class X86_64Parser(ArchitectureParser):
    INSTRUCTION_MAP = {
        'mov': IROpcode.MOVE,
        'movq': IROpcode.MOVE,
        'movl': IROpcode.MOVE,
        'add': IROpcode.ADD,
        'sub': IROpcode.SUB,
        'imul': IROpcode.MUL,
        'idiv': IROpcode.DIV,
        'and': IROpcode.AND,
        'or': IROpcode.OR,
        'xor': IROpcode.XOR,
        'not': IROpcode.NOT,
        'shl': IROpcode.SHL,
        'shr': IROpcode.SHR,
        'sar': IROpcode.SAR,
        'cmp': IROpcode.CMP,
        'test': IROpcode.TEST,
        'jmp': IROpcode.JMP,
        'je': IROpcode.JE,
        'jne': IROpcode.JNE,
        'jz': IROpcode.JZ,
        'jnz': IROpcode.JNZ,
        'jl': IROpcode.JL,
        'jle': IROpcode.JLE,
        'jg': IROpcode.JG,
        'jge': IROpcode.JGE,
        'call': IROpcode.CALL,
        'ret': IROpcode.RET,
        'push': IROpcode.PUSH,
        'pop': IROpcode.POP,
        'nop': IROpcode.NOP,
        'syscall': IROpcode.SYSCALL,
        'lea': IROpcode.LOAD,
    }
    
    def parse_instruction(self, line: str) -> Optional[IRInstruction]:
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('#'):
            return None
        if line.endswith(':'):
            return None
        parts = re.split(r'\s+', line, maxsplit=1)
        if not parts:
            return None
        mnemonic = parts[0].lower()
        operands_str = parts[1] if len(parts) > 1 else ""
        opcode = self.INSTRUCTION_MAP.get(mnemonic)
        if not opcode:
            return None
        operands = []
        if operands_str:
            for op_str in operands_str.split(','):
                op_str = op_str.strip()
                operand = self._parse_operand(op_str)
                if operand:
                    operands.append(operand)
        return IRInstruction(opcode=opcode, operands=operands)
    
    def _parse_operand(self, op_str: str) -> Optional[IROperand]:
        op_str = op_str.strip()
        if op_str in ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                      'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']:
            vreg = RegisterMapper.to_virtual(Architecture.X86_64, op_str)
            return IROperand(type="reg", value=vreg, size=64)
        if op_str.isdigit() or (op_str.startswith('-') and op_str[1:].isdigit()):
            return IROperand(type="imm", value=int(op_str), size=32)
        if op_str.startswith('0x'):
            return IROperand(type="imm", value=int(op_str, 16), size=32)
        if op_str.startswith('[') and op_str.endswith(']'):
            mem_expr = op_str[1:-1]
            return IROperand(type="mem", value=mem_expr, size=64)
        return IROperand(type="label", value=op_str, size=0)
    
    def get_syntax_type(self) -> str:
        return "intel"

class ARM64Parser(ArchitectureParser):
    INSTRUCTION_MAP = {
        'mov': IROpcode.MOVE,
        'movz': IROpcode.MOVE,
        'movk': IROpcode.MOVE,
        'add': IROpcode.ADD,
        'sub': IROpcode.SUB,
        'mul': IROpcode.MUL,
        'sdiv': IROpcode.DIV,
        'and': IROpcode.AND,
        'orr': IROpcode.OR,
        'eor': IROpcode.XOR,
        'mvn': IROpcode.NOT,
        'lsl': IROpcode.SHL,
        'lsr': IROpcode.SHR,
        'asr': IROpcode.SAR,
        'cmp': IROpcode.CMP,
        'tst': IROpcode.TEST,
        'b': IROpcode.JMP,
        'beq': IROpcode.JE,
        'bne': IROpcode.JNE,
        'blt': IROpcode.JL,
        'ble': IROpcode.JLE,
        'bgt': IROpcode.JG,
        'bge': IROpcode.JGE,
        'bl': IROpcode.CALL,
        'blr': IROpcode.CALL,
        'ret': IROpcode.RET,
        'str': IROpcode.STORE,
        'ldr': IROpcode.LOAD,
        'stp': IROpcode.PUSH,
        'ldp': IROpcode.POP,
        'nop': IROpcode.NOP,
        'svc': IROpcode.SYSCALL,
    }
    
    def parse_instruction(self, line: str) -> Optional[IRInstruction]:
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('//'):
            return None
        if line.endswith(':'):
            return None
        parts = re.split(r'\s+', line, maxsplit=1)
        if not parts:
            return None
        mnemonic = parts[0].lower()
        operands_str = parts[1] if len(parts) > 1 else ""
        opcode = self.INSTRUCTION_MAP.get(mnemonic)
        if not opcode:
            return None
        operands = []
        if operands_str:
            for op_str in operands_str.split(','):
                op_str = op_str.strip()
                operand = self._parse_operand(op_str)
                if operand:
                    operands.append(operand)
        return IRInstruction(opcode=opcode, operands=operands)
    
    def _parse_operand(self, op_str: str) -> Optional[IROperand]:
        op_str = op_str.strip()
        if op_str.startswith('x') or op_str.startswith('w'):
            vreg = RegisterMapper.to_virtual(Architecture.ARM64, op_str)
            size = 64 if op_str.startswith('x') else 32
            return IROperand(type="reg", value=vreg, size=size)
        if op_str in ['sp', 'lr', 'fp']:
            vreg = RegisterMapper.to_virtual(Architecture.ARM64, op_str)
            return IROperand(type="reg", value=vreg, size=64)
        if op_str.startswith('#'):
            value = op_str[1:]
            if value.startswith('0x'):
                return IROperand(type="imm", value=int(value, 16), size=32)
            return IROperand(type="imm", value=int(value), size=32)
        if op_str.startswith('[') and op_str.endswith(']'):
            mem_expr = op_str[1:-1]
            return IROperand(type="mem", value=mem_expr, size=64)
        return IROperand(type="label", value=op_str, size=0)
    
    def get_syntax_type(self) -> str:
        return "arm"

class RISCVParser(ArchitectureParser):
    INSTRUCTION_MAP = {
        'mv': IROpcode.MOVE,
        'li': IROpcode.MOVE,
        'add': IROpcode.ADD,
        'addi': IROpcode.ADD,
        'sub': IROpcode.SUB,
        'mul': IROpcode.MUL,
        'div': IROpcode.DIV,
        'and': IROpcode.AND,
        'andi': IROpcode.AND,
        'or': IROpcode.OR,
        'ori': IROpcode.OR,
        'xor': IROpcode.XOR,
        'xori': IROpcode.XOR,
        'not': IROpcode.NOT,
        'sll': IROpcode.SHL,
        'srl': IROpcode.SHR,
        'sra': IROpcode.SAR,
        'beq': IROpcode.JE,
        'bne': IROpcode.JNE,
        'blt': IROpcode.JL,
        'bge': IROpcode.JGE,
        'j': IROpcode.JMP,
        'jal': IROpcode.CALL,
        'jalr': IROpcode.CALL,
        'ret': IROpcode.RET,
        'lw': IROpcode.LOAD,
        'ld': IROpcode.LOAD,
        'sw': IROpcode.STORE,
        'sd': IROpcode.STORE,
        'nop': IROpcode.NOP,
        'ecall': IROpcode.SYSCALL,
    }
    
    def parse_instruction(self, line: str) -> Optional[IRInstruction]:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith(';'):
            return None
        if line.endswith(':'):
            return None
        parts = re.split(r'\s+', line, maxsplit=1)
        if not parts:
            return None
        mnemonic = parts[0].lower()
        operands_str = parts[1] if len(parts) > 1 else ""
        opcode = self.INSTRUCTION_MAP.get(mnemonic)
        if not opcode:
            return None
        operands = []
        if operands_str:
            for op_str in operands_str.split(','):
                op_str = op_str.strip()
                operand = self._parse_operand(op_str)
                if operand:
                    operands.append(operand)
        return IRInstruction(opcode=opcode, operands=operands)
    
    def _parse_operand(self, op_str: str) -> Optional[IROperand]:
        op_str = op_str.strip()
        if op_str.startswith('x') or op_str.startswith('a') or op_str.startswith('t'):
            vreg = RegisterMapper.to_virtual(Architecture.RISCV64, op_str)
            return IROperand(type="reg", value=vreg, size=64)
        if op_str in ['sp', 'ra', 'fp', 'zero']:
            vreg = RegisterMapper.to_virtual(Architecture.RISCV64, op_str)
            return IROperand(type="reg", value=vreg, size=64)
        if op_str.isdigit() or (op_str.startswith('-') and op_str[1:].isdigit()):
            return IROperand(type="imm", value=int(op_str), size=32)
        match = re.match(r'(-?\d+)\((\w+)\)', op_str)
        if match:
            offset, reg = match.groups()
            return IROperand(type="mem", value=f"{reg}+{offset}", size=64)
        return IROperand(type="label", value=op_str, size=0)
    
    def get_syntax_type(self) -> str:
        return "riscv"
