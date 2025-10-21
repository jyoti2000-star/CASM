from enum import Enum, auto
from dataclasses import dataclass

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
