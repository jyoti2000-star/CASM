from enum import Enum, auto

class Platform(Enum):
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    BSD = "bsd"
    FREEBSD = "freebsd"
    ANDROID = "android"

class Architecture(Enum):
    X86_64 = "x86_64"
    X86 = "x86"
    ARM64 = "arm64"
    ARM32 = "arm32"
    RISCV64 = "riscv64"
    RISCV32 = "riscv32"
    MIPS = "mips"
    POWERPC = "powerpc"

class OptimizationLevel(Enum):
    NONE = 0
    BASIC = 1
    MODERATE = 2
    AGGRESSIVE = 3
    EXTREME = 4

class SectionType(Enum):
    TEXT = "text"
    DATA = "data"
    BSS = "bss"
    RODATA = "rodata"
    TLS = "tls"
    INIT = "init"
    FINI = "fini"
    DEBUG = "debug"
    COMMENT = "comment"

class InstructionType(Enum):
    SCALAR = auto()
    VECTOR = auto()
    ATOMIC = auto()
    MEMORY = auto()
    CONTROL = auto()
    CRYPTO = auto()
    ARITHMETIC = auto()
    LOGICAL = auto()
    COMPARISON = auto()
    STACK = auto()

class RegisterClass(Enum):
    INTEGER = auto()
    FLOAT = auto()
    VECTOR = auto()
    SPECIAL = auto()
    FLAGS = auto()

class DataType(Enum):
    BYTE = 1
    WORD = 2
    DWORD = 4
    QWORD = 8
    OWORD = 16
    YWORD = 32
    ZWORD = 64
