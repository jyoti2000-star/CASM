from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any

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
    LITTLE = "little"
    BIG = "big"
    BI = "bi"

class RegisterSize(Enum):
    BIT8 = 8
    BIT16 = 16
    BIT32 = 32
    BIT64 = 64
    BIT128 = 128
    BIT256 = 256
    BIT512 = 512

class IROpcode(Enum):
    MOVE = auto()
    LOAD = auto()
    STORE = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    NOT = auto()
    SHL = auto()
    SHR = auto()
    SAR = auto()
    ROL = auto()
    ROR = auto()
    CMP = auto()
    TEST = auto()
    JMP = auto()
    JZ = auto()
    JNZ = auto()
    JE = auto()
    JNE = auto()
    JL = auto()
    JLE = auto()
    JG = auto()
    JGE = auto()
    CALL = auto()
    RET = auto()
    PUSH = auto()
    POP = auto()
    NOP = auto()
    SYSCALL = auto()
    BREAKPOINT = auto()
    VADD = auto()
    VSUB = auto()
    VMUL = auto()
    VDIV = auto()
    XCHG = auto()
    CMPXCHG = auto()
    FENCE = auto()
