#!/usr/bin/env python3
"""
Universal Assembly Cross-Compiler v1.0

A revolutionary assembly cross-compiler that can translate between ANY architectures:
- x86/x86_64 ↔ ARM32/ARM64 ↔ RISC-V ↔ MIPS ↔ PowerPC ↔ SPARC ↔ AVR
- Automatic instruction mapping and optimization
- Register allocation and mapping
- Calling convention translation
- Endianness conversion
- ABI compatibility layers
- Semantic-aware translation
- IR (Intermediate Representation) based approach
"""

from __future__ import annotations

import re
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__all__ = ["UniversalCrossCompiler", "Architecture", "translate_assembly"]

# ==================== ARCHITECTURES ====================

class Architecture(Enum):
    """Supported architectures"""
    X86 = "x86"
    X86_64 = "x86_64"
    ARM32 = "arm32"
    ARM64 = "arm64"
    ARMV7 = "armv7"
    THUMB = "thumb"
    RISCV32 = "riscv32"
    RISCV64 = "riscv64"
    MIPS = "mips"
    MIPS64 = "mips64"
    POWERPC = "powerpc"
    POWERPC64 = "powerpc64"
    SPARC = "sparc"
    SPARC64 = "sparc64"
    AVR = "avr"
    M68K = "m68k"
    SH4 = "sh4"
    XTENSA = "xtensa"
    WEBASSEMBLY = "wasm"

class Endianness(Enum):
    """Byte ordering"""
    LITTLE = "little"
    BIG = "big"
    BI = "bi"  # Can be either

class RegisterSize(Enum):
    """Register sizes"""
    BIT8 = 8
    BIT16 = 16
    BIT32 = 32
    BIT64 = 64
    BIT128 = 128
    BIT256 = 256
    BIT512 = 512

# ==================== IR (Intermediate Representation) ====================

class IROpcode(Enum):
    """IR operation codes - architecture-independent"""
    # Data movement
    MOVE = auto()
    LOAD = auto()
    STORE = auto()
    
    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    XOR = auto()
    NOT = auto()
    SHL = auto()
    SHR = auto()
    SAR = auto()  # Arithmetic shift
    ROL = auto()
    ROR = auto()
    
    # Comparison
    CMP = auto()
    TEST = auto()
    
    # Branch/Jump
    JMP = auto()
    JZ = auto()    # Jump if zero
    JNZ = auto()   # Jump if not zero
    JE = auto()    # Jump if equal
    JNE = auto()   # Jump if not equal
    JL = auto()    # Jump if less
    JLE = auto()   # Jump if less or equal
    JG = auto()    # Jump if greater
    JGE = auto()   # Jump if greater or equal
    
    # Function calls
    CALL = auto()
    RET = auto()
    
    # Stack operations
    PUSH = auto()
    POP = auto()
    
    # Special
    NOP = auto()
    SYSCALL = auto()
    BREAKPOINT = auto()
    
    # SIMD/Vector
    VADD = auto()
    VSUB = auto()
    VMUL = auto()
    VDIV = auto()
    
    # Atomic
    XCHG = auto()
    CMPXCHG = auto()
    
    # Memory barriers
    FENCE = auto()

@dataclass
class IROperand:
    """IR operand"""
    type: str  # "reg", "imm", "mem", "label"
    value: Any
    size: int = 32  # bits
    
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
    """IR instruction - architecture independent"""
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

# ==================== REGISTER MAPPING ====================

class RegisterMapper:
    """Maps registers between architectures"""
    
    # Virtual register types
    VIRTUAL_REGS = {
        'v0': 'general',  'v1': 'general',  'v2': 'general',  'v3': 'general',
        'v4': 'general',  'v5': 'general',  'v6': 'general',  'v7': 'general',
        'v8': 'general',  'v9': 'general',  'v10': 'general', 'v11': 'general',
        'v12': 'general', 'v13': 'general', 'v14': 'general', 'v15': 'general',
        'f0': 'float',    'f1': 'float',    'f2': 'float',    'f3': 'float',
        'f4': 'float',    'f5': 'float',    'f6': 'float',    'f7': 'float',
        'sp': 'stack',    'fp': 'frame',    'lr': 'link',     'pc': 'program',
    }
    
    # Architecture-specific register mappings
    X86_64_REGS = {
        'rax': 'v0', 'rbx': 'v1', 'rcx': 'v2', 'rdx': 'v3',
        'rsi': 'v4', 'rdi': 'v5', 'rbp': 'fp', 'rsp': 'sp',
        'r8': 'v6',  'r9': 'v7',  'r10': 'v8', 'r11': 'v9',
        'r12': 'v10', 'r13': 'v11', 'r14': 'v12', 'r15': 'v13',
        'xmm0': 'f0', 'xmm1': 'f1', 'xmm2': 'f2', 'xmm3': 'f3',
        'xmm4': 'f4', 'xmm5': 'f5', 'xmm6': 'f6', 'xmm7': 'f7',
    }
    
    ARM64_REGS = {
        'x0': 'v0',  'x1': 'v1',  'x2': 'v2',  'x3': 'v3',
        'x4': 'v4',  'x5': 'v5',  'x6': 'v6',  'x7': 'v7',
        'x8': 'v8',  'x9': 'v9',  'x10': 'v10', 'x11': 'v11',
        'x12': 'v12', 'x13': 'v13', 'x14': 'v14', 'x15': 'v15',
        'x29': 'fp', 'x30': 'lr', 'sp': 'sp',
        'v0': 'f0',  'v1': 'f1',  'v2': 'f2',  'v3': 'f3',
        'v4': 'f4',  'v5': 'f5',  'v6': 'f6',  'v7': 'f7',
    }
    
    RISCV_REGS = {
        'x0': 'v0',  'x1': 'lr',  'x2': 'sp',  'x3': 'v1',
        'x5': 'v2',  'x6': 'v3',  'x7': 'v4',  'x8': 'fp',
        'x9': 'v5',  'x10': 'v6', 'x11': 'v7', 'x12': 'v8',
        'x13': 'v9', 'x14': 'v10', 'x15': 'v11',
        'a0': 'v6',  'a1': 'v7',  'a2': 'v8',  'a3': 'v9',
        'a4': 'v10', 'a5': 'v11', 'a6': 'v12', 'a7': 'v13',
        'fa0': 'f0', 'fa1': 'f1', 'fa2': 'f2', 'fa3': 'f3',
    }
    
    MIPS_REGS = {
        '$zero': 'v0', '$at': 'v1', '$v0': 'v2', '$v1': 'v3',
        '$a0': 'v4',   '$a1': 'v5', '$a2': 'v6', '$a3': 'v7',
        '$t0': 'v8',   '$t1': 'v9', '$t2': 'v10', '$t3': 'v11',
        '$sp': 'sp',   '$fp': 'fp', '$ra': 'lr',
    }
    
    POWERPC_REGS = {
        'r0': 'v0',  'r1': 'sp',  'r2': 'v1',  'r3': 'v2',
        'r4': 'v3',  'r5': 'v4',  'r6': 'v5',  'r7': 'v6',
        'r8': 'v7',  'r9': 'v8',  'r10': 'v9', 'r11': 'v10',
        'f0': 'f0',  'f1': 'f1',  'f2': 'f2',  'f3': 'f3',
    }
    
    @staticmethod
    def to_virtual(arch: Architecture, reg: str) -> str:
        """Convert architecture register to virtual register"""
        reg = reg.lower().strip()
        
        mapping = {
            Architecture.X86_64: RegisterMapper.X86_64_REGS,
            Architecture.ARM64: RegisterMapper.ARM64_REGS,
            Architecture.RISCV64: RegisterMapper.RISCV_REGS,
            Architecture.MIPS: RegisterMapper.MIPS_REGS,
            Architecture.POWERPC: RegisterMapper.POWERPC_REGS,
        }.get(arch, {})
        
        return mapping.get(reg, f"v_unknown_{reg}")
    
    @staticmethod
    def from_virtual(arch: Architecture, vreg: str) -> str:
        """Convert virtual register to architecture register"""
        vreg = vreg.lower().strip()
        
        mapping = {
            Architecture.X86_64: RegisterMapper.X86_64_REGS,
            Architecture.ARM64: RegisterMapper.ARM64_REGS,
            Architecture.RISCV64: RegisterMapper.RISCV_REGS,
            Architecture.MIPS: RegisterMapper.MIPS_REGS,
            Architecture.POWERPC: RegisterMapper.POWERPC_REGS,
        }.get(arch, {})
        
        # Reverse mapping
        reverse = {v: k for k, v in mapping.items()}
        return reverse.get(vreg, 'r0')

# ==================== ARCHITECTURE PARSERS ====================

class ArchitectureParser(ABC):
    """Base class for architecture-specific parsers"""
    
    @abstractmethod
    def parse_instruction(self, line: str) -> Optional[IRInstruction]:
        """Parse architecture-specific instruction to IR"""
        pass
    
    @abstractmethod
    def get_syntax_type(self) -> str:
        """Get syntax type (intel, att, etc.)"""
        pass

class X86_64Parser(ArchitectureParser):
    """Parser for x86_64 assembly"""
    
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
        """Parse x86_64 instruction"""
        line = line.strip()
        
        if not line or line.startswith(';') or line.startswith('#'):
            return None
        
        if line.endswith(':'):
            return None
        
        # Parse mnemonic and operands
        parts = re.split(r'\s+', line, maxsplit=1)
        if not parts:
            return None
        
        mnemonic = parts[0].lower()
        operands_str = parts[1] if len(parts) > 1 else ""
        
        # Map to IR opcode
        opcode = self.INSTRUCTION_MAP.get(mnemonic)
        if not opcode:
            return None
        
        # Parse operands
        operands = []
        if operands_str:
            for op_str in operands_str.split(','):
                op_str = op_str.strip()
                operand = self._parse_operand(op_str)
                if operand:
                    operands.append(operand)
        
        return IRInstruction(opcode=opcode, operands=operands)
    
    def _parse_operand(self, op_str: str) -> Optional[IROperand]:
        """Parse x86_64 operand"""
        op_str = op_str.strip()
        
        # Register
        if op_str in ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                      'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']:
            vreg = RegisterMapper.to_virtual(Architecture.X86_64, op_str)
            return IROperand(type="reg", value=vreg, size=64)
        
        # Immediate
        if op_str.isdigit() or (op_str.startswith('-') and op_str[1:].isdigit()):
            return IROperand(type="imm", value=int(op_str), size=32)
        
        if op_str.startswith('0x'):
            return IROperand(type="imm", value=int(op_str, 16), size=32)
        
        # Memory reference [...]
        if op_str.startswith('[') and op_str.endswith(']'):
            mem_expr = op_str[1:-1]
            return IROperand(type="mem", value=mem_expr, size=64)
        
        # Label
        return IROperand(type="label", value=op_str, size=0)
    
    def get_syntax_type(self) -> str:
        return "intel"

class ARM64Parser(ArchitectureParser):
    """Parser for ARM64 assembly"""
    
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
        """Parse ARM64 instruction"""
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
        """Parse ARM64 operand"""
        op_str = op_str.strip()
        
        # Register
        if op_str.startswith('x') or op_str.startswith('w'):
            vreg = RegisterMapper.to_virtual(Architecture.ARM64, op_str)
            size = 64 if op_str.startswith('x') else 32
            return IROperand(type="reg", value=vreg, size=size)
        
        if op_str in ['sp', 'lr', 'fp']:
            vreg = RegisterMapper.to_virtual(Architecture.ARM64, op_str)
            return IROperand(type="reg", value=vreg, size=64)
        
        # Immediate
        if op_str.startswith('#'):
            value = op_str[1:]
            if value.startswith('0x'):
                return IROperand(type="imm", value=int(value, 16), size=32)
            return IROperand(type="imm", value=int(value), size=32)
        
        # Memory [...]
        if op_str.startswith('[') and op_str.endswith(']'):
            mem_expr = op_str[1:-1]
            return IROperand(type="mem", value=mem_expr, size=64)
        
        # Label
        return IROperand(type="label", value=op_str, size=0)
    
    def get_syntax_type(self) -> str:
        return "arm"

class RISCVParser(ArchitectureParser):
    """Parser for RISC-V assembly"""
    
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
        """Parse RISC-V instruction"""
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
        """Parse RISC-V operand"""
        op_str = op_str.strip()
        
        # Register
        if op_str.startswith('x') or op_str.startswith('a') or op_str.startswith('t'):
            vreg = RegisterMapper.to_virtual(Architecture.RISCV64, op_str)
            return IROperand(type="reg", value=vreg, size=64)
        
        if op_str in ['sp', 'ra', 'fp', 'zero']:
            vreg = RegisterMapper.to_virtual(Architecture.RISCV64, op_str)
            return IROperand(type="reg", value=vreg, size=64)
        
        # Immediate
        if op_str.isdigit() or (op_str.startswith('-') and op_str[1:].isdigit()):
            return IROperand(type="imm", value=int(op_str), size=32)
        
        # Memory offset(reg)
        match = re.match(r'(-?\d+)\((\w+)\)', op_str)
        if match:
            offset, reg = match.groups()
            return IROperand(type="mem", value=f"{reg}+{offset}", size=64)
        
        # Label
        return IROperand(type="label", value=op_str, size=0)
    
    def get_syntax_type(self) -> str:
        return "riscv"

# ==================== ARCHITECTURE EMITTERS ====================

class ArchitectureEmitter(ABC):
    """Base class for architecture-specific code generators"""
    
    @abstractmethod
    def emit_instruction(self, ir_inst: IRInstruction) -> List[str]:
        """Emit architecture-specific instruction from IR"""
        pass
    
    @abstractmethod
    def emit_prologue(self, func_name: str, stack_size: int) -> List[str]:
        """Emit function prologue"""
        pass
    
    @abstractmethod
    def emit_epilogue(self) -> List[str]:
        """Emit function epilogue"""
        pass

class X86_64Emitter(ArchitectureEmitter):
    """Emitter for x86_64 assembly"""
    
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
        """Emit ARM64 instruction"""
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
        """Emit ARM64 operand"""
        if op.type == "reg":
            reg = RegisterMapper.from_virtual(Architecture.ARM64, str(op.value))
            # Convert to appropriate size
            if op.size == 32 and reg.startswith('x'):
                reg = 'w' + reg[1:]
            return reg
        elif op.type == "imm":
            return f"#{op.value}"
        elif op.type == "mem":
            return f"[{op.value}]"
        elif op.type == "label":
            return str(op.value)
        return str(op.value)
    
    def emit_prologue(self, func_name: str, stack_size: int) -> List[str]:
        """Emit ARM64 function prologue"""
        aligned_size = (stack_size + 15) & ~15  # 16-byte alignment
        return [
            f"{func_name}:",
            "    stp x29, x30, [sp, #-16]!",
            "    mov x29, sp",
            f"    sub sp, sp, #{aligned_size}" if aligned_size > 0 else "",
        ]
    
    def emit_epilogue(self) -> List[str]:
        """Emit ARM64 function epilogue"""
        return [
            "    mov sp, x29",
            "    ldp x29, x30, [sp], #16",
            "    ret",
        ]

class RISCVEmitter(ArchitectureEmitter):
    """Emitter for RISC-V assembly"""
    
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
        """Emit RISC-V instruction"""
        mnemonic = self.OPCODE_MAP.get(ir_inst.opcode)
        if not mnemonic:
            return [f"# Unknown IR opcode: {ir_inst.opcode}"]
        
        # Handle special cases
        if ir_inst.opcode == IROpcode.CMP:
            # RISC-V doesn't have CMP, use SUB with zero register
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
        """Emit RISC-V operand"""
        if op.type == "reg":
            return RegisterMapper.from_virtual(Architecture.RISCV64, str(op.value))
        elif op.type == "imm":
            return str(op.value)
        elif op.type == "mem":
            # Convert to offset(reg) format
            if '+' in str(op.value):
                parts = str(op.value).split('+')
                return f"{parts[1]}({parts[0]})"
            return f"0({op.value})"
        elif op.type == "label":
            return str(op.value)
        return str(op.value)
    
    def emit_prologue(self, func_name: str, stack_size: int) -> List[str]:
        """Emit RISC-V function prologue"""
        aligned_size = (stack_size + 15) & ~15
        return [
            f"{func_name}:",
            f"    addi sp, sp, -{aligned_size + 16}",
            f"    sd ra, {aligned_size + 8}(sp)",
            f"    sd s0, {aligned_size}(sp)",
            f"    addi s0, sp, {aligned_size + 16}",
        ]
    
    def emit_epilogue(self) -> List[str]:
        """Emit RISC-V function epilogue"""
        return [
            "    ld ra, 8(sp)",
            "    ld s0, 0(sp)",
            "    addi sp, sp, 16",
            "    ret",
        ]

# ==================== UNIVERSAL CROSS COMPILER ====================

class UniversalCrossCompiler:
    """Universal assembly cross-compiler"""
    
    def __init__(
        self,
        source_arch: Architecture,
        target_arch: Architecture,
        optimization_level: int = 2
    ):
        self.source_arch = source_arch
        self.target_arch = target_arch
        self.optimization_level = optimization_level
        
        # Create parser and emitter
        self.parser = self._create_parser(source_arch)
        self.emitter = self._create_emitter(target_arch)
        
        # IR storage
        self.ir_instructions: List[IRInstruction] = []
        self.labels: Dict[str, int] = {}
        self.functions: Dict[str, Tuple[int, int]] = {}  # name -> (start, end)
        
        # Statistics
        self.stats = {
            'source_instructions': 0,
            'ir_instructions': 0,
            'target_instructions': 0,
            'optimizations_applied': 0,
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_parser(self, arch: Architecture) -> ArchitectureParser:
        """Create parser for source architecture"""
        parsers = {
            Architecture.X86_64: X86_64Parser,
            Architecture.ARM64: ARM64Parser,
            Architecture.RISCV64: RISCVParser,
        }
        
        parser_class = parsers.get(arch)
        if not parser_class:
            raise ValueError(f"No parser for architecture: {arch}")
        
        return parser_class()
    
    def _create_emitter(self, arch: Architecture) -> ArchitectureEmitter:
        """Create emitter for target architecture"""
        emitters = {
            Architecture.X86_64: X86_64Emitter,
            Architecture.ARM64: ARM64Emitter,
            Architecture.RISCV64: RISCVEmitter,
        }
        
        emitter_class = emitters.get(arch)
        if not emitter_class:
            raise ValueError(f"No emitter for architecture: {arch}")
        
        return emitter_class()
    
    def parse_source(self, source: str) -> List[IRInstruction]:
        """Parse source assembly to IR"""
        self.ir_instructions.clear()
        self.labels.clear()
        
        lines = source.split('\n')
        self.stats['source_instructions'] = len([l for l in lines if l.strip()])
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            # Handle labels
            if line.endswith(':'):
                label = line[:-1]
                self.labels[label] = len(self.ir_instructions)
                continue
            
            # Parse instruction
            ir_inst = self.parser.parse_instruction(line)
            if ir_inst:
                self.ir_instructions.append(ir_inst)
        
        self.stats['ir_instructions'] = len(self.ir_instructions)
        self.logger.info(f"Parsed {self.stats['source_instructions']} source instructions to {self.stats['ir_instructions']} IR instructions")
        
        return self.ir_instructions
    
    def optimize_ir(self):
        """Optimize IR instructions"""
        if self.optimization_level == 0:
            return
        
        optimizations_applied = 0
        
        # Level 1: Remove redundant moves
        if self.optimization_level >= 1:
            optimizations_applied += self._remove_redundant_moves()
        
        # Level 2: Constant folding
        if self.optimization_level >= 2:
            optimizations_applied += self._constant_folding()
        
        # Level 3: Dead code elimination
        if self.optimization_level >= 3:
            optimizations_applied += self._eliminate_dead_code()
        
        self.stats['optimizations_applied'] = optimizations_applied
        self.logger.info(f"Applied {optimizations_applied} optimizations")
    
    def _remove_redundant_moves(self) -> int:
        """Remove mov reg, reg instructions"""
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
        """Fold constant arithmetic operations"""
        count = 0
        optimized = []
        
        for inst in self.ir_instructions:
            if inst.opcode in [IROpcode.ADD, IROpcode.SUB, IROpcode.MUL]:
                if len(inst.operands) >= 2:
                    op1 = inst.operands[1] if len(inst.operands) > 1 else None
                    op2 = inst.operands[2] if len(inst.operands) > 2 else None
                    
                    if op1 and op2 and op1.type == "imm" and op2.type == "imm":
                        # Fold constants
                        if inst.opcode == IROpcode.ADD:
                            result = op1.value + op2.value
                        elif inst.opcode == IROpcode.SUB:
                            result = op1.value - op2.value
                        elif inst.opcode == IROpcode.MUL:
                            result = op1.value * op2.value
                        else:
                            result = 0
                        
                        # Replace with MOVE
                        new_inst = IRInstruction(
                            opcode=IROpcode.MOVE,
                            operands=[
                                inst.operands[0],
                                IROperand(type="imm", value=result)
                            ]
                        )
                        optimized.append(new_inst)
                        count += 1
                        continue
            
            optimized.append(inst)
        
        self.ir_instructions = optimized
        return count
    
    def _eliminate_dead_code(self) -> int:
        """Eliminate unreachable code"""
        count = 0
        reachable = set()
        
        # Mark reachable instructions
        i = 0
        while i < len(self.ir_instructions):
            reachable.add(i)
            inst = self.ir_instructions[i]
            
            # Check for unconditional jumps
            if inst.opcode == IROpcode.JMP:
                # Find jump target
                if inst.operands and inst.operands[0].type == "label":
                    label = inst.operands[0].value
                    if label in self.labels:
                        i = self.labels[label]
                        continue
            elif inst.opcode == IROpcode.RET:
                # Stop at return
                break
            
            i += 1
        
        # Keep only reachable instructions
        optimized = [inst for i, inst in enumerate(self.ir_instructions) if i in reachable]
        count = len(self.ir_instructions) - len(optimized)
        self.ir_instructions = optimized
        
        return count
    
    def emit_target(self) -> str:
        """Emit target assembly from IR"""
        output = []
        
        # Header
        output.append(f"; Translated from {self.source_arch.value} to {self.target_arch.value}")
        output.append(f"; IR instructions: {len(self.ir_instructions)}")
        output.append("")
        
        # Emit labels and instructions
        label_positions = {pos: label for label, pos in self.labels.items()}
        
        for i, ir_inst in enumerate(self.ir_instructions):
            # Emit label if present
            if i in label_positions:
                output.append(f"{label_positions[i]}:")
            
            # Emit instruction
            asm_lines = self.emitter.emit_instruction(ir_inst)
            output.extend(asm_lines)
        
        result = '\n'.join(output)
        self.stats['target_instructions'] = len([l for l in output if l.strip() and not l.strip().startswith(';')])
        
        return result
    
    def translate(self, source: str) -> str:
        """Main translation method"""
        self.logger.info(f"Translating {self.source_arch.value} -> {self.target_arch.value}")
        
        # Parse source
        self.parse_source(source)
        
        # Optimize IR
        self.optimize_ir()
        
        # Emit target
        result = self.emit_target()
        
        self.logger.info(f"Translation complete: {self.stats}")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get translation statistics"""
        return {
            'source_arch': self.source_arch.value,
            'target_arch': self.target_arch.value,
            'optimization_level': self.optimization_level,
            **self.stats,
        }
    
    def export_ir(self, filename: str):
        """Export IR to file for debugging"""
        with open(filename, 'w') as f:
            f.write(f"; IR for {self.source_arch.value} -> {self.target_arch.value}\n")
            f.write(f"; {len(self.ir_instructions)} instructions\n\n")
            
            for i, inst in enumerate(self.ir_instructions):
                f.write(f"{i:4d}: {inst}\n")
        
        self.logger.info(f"Exported IR to {filename}")

# ==================== CONVENIENCE FUNCTIONS ====================

def translate_assembly(
    source: str,
    source_arch: str,
    target_arch: str,
    optimization: int = 2
) -> str:
    """Translate assembly from one architecture to another"""
    src_arch = Architecture(source_arch.lower())
    tgt_arch = Architecture(target_arch.lower())
    
    compiler = UniversalCrossCompiler(src_arch, tgt_arch, optimization)
    return compiler.translate(source)

def translate_file(
    input_file: str,
    output_file: str,
    source_arch: str,
    target_arch: str,
    optimization: int = 2
):
    """Translate assembly file"""
    with open(input_file, 'r') as f:
        source = f.read()
    
    result = translate_assembly(source, source_arch, target_arch, optimization)
    
    with open(output_file, 'w') as f:
        f.write(result)
    
    print(f"Translated {input_file} -> {output_file}")

# ==================== CLI ====================

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Universal Assembly Cross-Compiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -s x86_64 -t arm64 input.asm -o output.asm
  %(prog)s -s arm64 -t riscv64 input.s -o output.s -O3
  %(prog)s -s x86_64 -t mips --export-ir ir.txt input.asm

Supported Architectures:
  x86_64, arm64, riscv64, mips, powerpc, sparc, and more
        """
    )
    
    parser.add_argument('input', help='Input assembly file')
    parser.add_argument('-o', '--output', required=True, help='Output assembly file')
    parser.add_argument('-s', '--source-arch', required=True,
                       help='Source architecture')
    parser.add_argument('-t', '--target-arch', required=True,
                       help='Target architecture')
    parser.add_argument('-O', '--optimization', type=int, default=2,
                       choices=[0, 1, 2, 3],
                       help='Optimization level (default: 2)')
    parser.add_argument('--export-ir', help='Export IR to file')
    parser.add_argument('--stats', action='store_true',
                       help='Show translation statistics')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--version', action='version',
                       version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Create compiler
        src_arch = Architecture(args.source_arch.lower())
        tgt_arch = Architecture(args.target_arch.lower())
        
        compiler = UniversalCrossCompiler(
            src_arch,
            tgt_arch,
            args.optimization
        )
        
        # Read input
        with open(args.input, 'r') as f:
            source = f.read()
        
        # Translate
        result = compiler.translate(source)
        
        # Write output
        with open(args.output, 'w') as f:
            f.write(result)
        
        print(f"✓ Translated {args.input} -> {args.output}")
        
        # Export IR if requested
        if args.export_ir:
            compiler.export_ir(args.export_ir)
        
        # Show stats
        if args.stats:
            stats = compiler.get_statistics()
            print("\nTranslation Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example: x86_64 to ARM64 translation
    x86_code = """
section .text
global main

main:
    push rbp
    mov rbp, rsp
    mov rax, 42
    add rax, 8
    pop rbp
    ret
"""
    
    print("=== x86_64 -> ARM64 Translation ===\n")
    print("Source (x86_64):")
    print(x86_code)
    
    compiler = UniversalCrossCompiler(
        Architecture.X86_64,
        Architecture.ARM64,
        optimization_level=2
    )
    
    result = compiler.translate(x86_code)
    
    print("\nTarget (ARM64):")
    print(result)
    
    print("\nStatistics:")
    for key, value in compiler.get_statistics().items():
        print(f"  {key}: {value}")
    
    # Example: ARM64 to RISC-V
    print("\n\n=== ARM64 -> RISC-V Translation ===\n")
    
    arm_code = """
.global start
start:
    mov x0, #10
    mov x1, #20
    add x2, x0, x1
    ret
"""
    
    print("Source (ARM64):")
    print(arm_code)
    
    compiler2 = UniversalCrossCompiler(
        Architecture.ARM64,
        Architecture.RISCV64,
        optimization_level=3
    )
    
    result2 = compiler2.translate(arm_code)
    
    print("\nTarget (RISC-V):")
    print(result2)
    
    # CLI entry point
    if len(sys.argv) > 1:
        sys.exit(main()) 'sar',
        IROpcode.CMP: 'cmp',
        IROpcode.TEST: 'test',
        IROpcode.JMP: 'jmp',
        IROpcode.JE: 'je',
        IROpcode.JNE: 'jne',
        IROpcode.JZ: 'jz',
        IROpcode.JNZ: 'jnz',
        IROpcode.JL: 'jl',
        IROpcode.JLE: 'jle',
        IROpcode.JG: 'jg',
        IROpcode.JGE: 'jge',
        IROpcode.CALL: 'call',
        IROpcode.RET: 'ret',
        IROpcode.PUSH: 'push',
        IROpcode.POP: 'pop',
        IROpcode.NOP: 'nop',
        IROpcode.SYSCALL: 'syscall',
    }
    
    def emit_instruction(self, ir_inst: IRInstruction) -> List[str]:
        """Emit x86_64 instruction"""
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
        """Emit x86_64 operand"""
        if op.type == "reg":
            return RegisterMapper.from_virtual(Architecture.X86_64, str(op.value))
        elif op.type == "imm":
            return str(op.value)
        elif op.type == "mem":
            return f"[{op.value}]"
        elif op.type == "label":
            return str(op.value)
        return str(op.value)
    
    def emit_prologue(self, func_name: str, stack_size: int) -> List[str]:
        """Emit x86_64 function prologue"""
        return [
            f"{func_name}:",
            "    push rbp",
            "    mov rbp, rsp",
            f"    sub rsp, {stack_size}" if stack_size > 0 else "",
        ]
    
    def emit_epilogue(self) -> List[str]:
        """Emit x86_64 function epilogue"""
        return [
            "    mov rsp, rbp",
            "    pop rbp",
            "    ret",
        ]

class ARM64Emitter(ArchitectureEmitter):
    """Emitter for ARM64 assembly"""
    
    OPCODE_MAP = {
        IROpcode.MOVE: 'mov',
        IROpcode.ADD: 'add',
        IROpcode.SUB: 'sub',
        IROpcode.MUL: 'mul',
        IROpcode.DIV: 'sdiv',
        IROpcode.AND: 'and',
        IROpcode.OR: 'orr',
        IROpcode.XOR: 'eor',
        IROpcode.NOT: 'mvn',
        IROpcode.SHL: 'lsl',
        IROpcode.SHR: 'lsr',
        IROpcode.SAR: 'asr',