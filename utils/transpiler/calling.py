from typing import Dict, Any, Optional
from .enums import Platform, Architecture
from .exceptions import TranspileError

class CallingConvention:
    SYSV_INT_ARGS = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
    SYSV_FLOAT_ARGS = ['xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7']
    SYSV_PRESERVED = ['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15']
    SYSV_SCRATCH = ['rax', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11']
    
    WIN64_INT_ARGS = ['rcx', 'rdx', 'r8', 'r9']
    WIN64_FLOAT_ARGS = ['xmm0', 'xmm1', 'xmm2', 'xmm3']
    WIN64_PRESERVED = ['rbx', 'rbp', 'rdi', 'rsi', 'rsp', 'r12', 'r13', 'r14', 'r15', 'xmm6', 'xmm7', 'xmm8', 'xmm9', 'xmm10', 'xmm11', 'xmm12', 'xmm13', 'xmm14', 'xmm15']
    WIN64_SCRATCH = ['rax', 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11']
    WIN64_SHADOW_SPACE = 32
    
    ARM64_INT_ARGS = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    ARM64_FLOAT_ARGS = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    ARM64_PRESERVED = ['x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30']
    ARM64_SCRATCH = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']
    
    RISCV_INT_ARGS = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
    RISCV_FLOAT_ARGS = ['fa0', 'fa1', 'fa2', 'fa3', 'fa4', 'fa5', 'fa6', 'fa7']
    RISCV_PRESERVED = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
    
    @staticmethod
    def get_convention(platform: Platform, arch: Architecture) -> Dict[str, Any]:
        if arch == Architecture.X86_64:
            if platform == Platform.WINDOWS:
                return {
                    'int_args': CallingConvention.WIN64_INT_ARGS,
                    'float_args': CallingConvention.WIN64_FLOAT_ARGS,
                    'preserved': CallingConvention.WIN64_PRESERVED,
                    'scratch': CallingConvention.WIN64_SCRATCH,
                    'shadow_space': CallingConvention.WIN64_SHADOW_SPACE,
                    'stack_alignment': 16
                }
            else:
                return {
                    'int_args': CallingConvention.SYSV_INT_ARGS,
                    'float_args': CallingConvention.SYSV_FLOAT_ARGS,
                    'preserved': CallingConvention.SYSV_PRESERVED,
                    'scratch': CallingConvention.SYSV_SCRATCH,
                    'shadow_space': 0,
                    'stack_alignment': 16
                }
        elif arch == Architecture.ARM64:
            return {
                'int_args': CallingConvention.ARM64_INT_ARGS,
                'float_args': CallingConvention.ARM64_FLOAT_ARGS,
                'preserved': CallingConvention.ARM64_PRESERVED,
                'scratch': CallingConvention.ARM64_SCRATCH,
                'shadow_space': 0,
                'stack_alignment': 16
            }
        elif arch == Architecture.RISCV64:
            return {
                'int_args': CallingConvention.RISCV_INT_ARGS,
                'float_args': CallingConvention.RISCV_FLOAT_ARGS,
                'preserved': CallingConvention.RISCV_PRESERVED,
                'shadow_space': 0,
                'stack_alignment': 16
            }
        else:
            raise TranspileError(f"Unsupported architecture: {arch}")
    
    @staticmethod
    def get_arg_register(platform: Platform, arch: Architecture, index: int, is_float: bool = False) -> Optional[str]:
        conv = CallingConvention.get_convention(platform, arch)
        args = conv['float_args'] if is_float else conv['int_args']
        return args[index] if index < len(args) else None
    
    @staticmethod
    def get_return_register(arch: Architecture, is_float: bool = False) -> str:
        if arch == Architecture.X86_64:
            return 'xmm0' if is_float else 'rax'
        elif arch == Architecture.ARM64:
            return 'v0' if is_float else 'x0'
        elif arch == Architecture.RISCV64:
            return 'fa0' if is_float else 'a0'
        return 'rax'
    
    @staticmethod
    def needs_shadow_space(platform: Platform) -> bool:
        return platform == Platform.WINDOWS
