from .enums import Architecture
from typing import Set

class RegisterMapper:
    VIRTUAL_REGS = {
        'v0': 'general',  'v1': 'general',  'v2': 'general',  'v3': 'general',
        'v4': 'general',  'v5': 'general',  'v6': 'general',  'v7': 'general',
        'v8': 'general',  'v9': 'general',  'v10': 'general', 'v11': 'general',
        'v12': 'general', 'v13': 'general', 'v14': 'general', 'v15': 'general',
        'f0': 'float',    'f1': 'float',    'f2': 'float',    'f3': 'float',
        'f4': 'float',    'f5': 'float',    'f6': 'float',    'f7': 'float',
        'sp': 'stack',    'fp': 'frame',    'lr': 'link',     'pc': 'program',
    }
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
        vreg = vreg.lower().strip()
        mapping = {
            Architecture.X86_64: RegisterMapper.X86_64_REGS,
            Architecture.ARM64: RegisterMapper.ARM64_REGS,
            Architecture.RISCV64: RegisterMapper.RISCV_REGS,
            Architecture.MIPS: RegisterMapper.MIPS_REGS,
            Architecture.POWERPC: RegisterMapper.POWERPC_REGS,
        }.get(arch, {})
        reverse = {v: k for k, v in mapping.items()}
        return reverse.get(vreg, 'r0')
