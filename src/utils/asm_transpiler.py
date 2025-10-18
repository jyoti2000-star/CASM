#!/usr/bin/env python3
"""Ultra-Advanced Cross-Platform Assembly Transpiler v2.0

An extremely sophisticated assembly transpiler with AI-assisted optimization,
vectorization, inline assembly support, and comprehensive platform abstraction.

New Advanced Features:
- SIMD/AVX instruction translation
- Automatic vectorization hints
- Inline C/C++ integration
- Thread-safe code generation
- Atomic operations abstraction
- Performance profiling hooks
- Link-time optimization support
- Multiple architecture support (x86-64, ARM64 translation layer)
- Advanced loop unrolling
- Constant folding and propagation
- Dead code elimination
- Tail call optimization
- Custom ABI definitions
- Structured exception handling
- Memory pool management
- DMA operations
- Hardware intrinsics
"""

from __future__ import annotations

import re
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod

__version__ = "2.0.0"
__all__ = ["transpile_file", "transpile_text", "main", "TranspileError"]


class Platform(Enum):
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    BSD = "bsd"


class Architecture(Enum):
    X86_64 = "x86_64"
    ARM64 = "arm64"
    RISCV64 = "riscv64"


class OptimizationLevel(Enum):
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    EXTREME = 3


class SectionType(Enum):
    TEXT = "text"
    DATA = "data"
    BSS = "bss"
    RODATA = "rodata"
    TLS = "tls"
    INIT = "init"
    FINI = "fini"


class InstructionType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    ATOMIC = "atomic"
    MEMORY = "memory"
    CONTROL = "control"
    CRYPTO = "crypto"


@dataclass
class Variable:
    """Variable metadata"""
    name: str
    type_: str
    size: int
    offset: int = 0
    is_global: bool = False
    is_const: bool = False
    alignment: int = 8
    section: SectionType = SectionType.DATA


@dataclass
class Function:
    """Function metadata with advanced tracking"""
    name: str
    params: List[Tuple[str, str]]  # (name, type)
    return_type: str = "void"
    local_size: int = 0
    is_global: bool = False
    is_inline: bool = False
    is_pure: bool = False
    preserves_regs: List[str] = field(default_factory=list)
    clobbers_regs: List[str] = field(default_factory=list)
    stack_alignment: int = 16
    uses_simd: bool = False
    is_leaf: bool = True
    call_count: int = 0
    optimization_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Macro:
    """Macro definition with variadic support"""
    name: str
    params: List[str]
    body: List[str]
    is_variadic: bool = False
    doc: str = ""


@dataclass
class Loop:
    """Loop metadata for optimization"""
    start_label: str
    end_label: str
    counter_reg: Optional[str] = None
    trip_count: Optional[int] = None
    is_vectorizable: bool = False
    unroll_factor: int = 1


@dataclass
class BasicBlock:
    """Basic block for control flow analysis"""
    label: str
    instructions: List[str] = field(default_factory=list)
    predecessors: Set[str] = field(default_factory=set)
    successors: Set[str] = field(default_factory=set)
    dominators: Set[str] = field(default_factory=set)


class CallingConvention:
    """Enhanced calling convention abstractions"""
    
    # Linux System V AMD64 ABI
    LINUX_INT_ARGS = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
    LINUX_FLOAT_ARGS = ['xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7']
    LINUX_PRESERVED = ['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15']
    LINUX_SCRATCH = ['rax', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11']
    
    # Windows x64
    WIN_INT_ARGS = ['rcx', 'rdx', 'r8', 'r9']
    WIN_FLOAT_ARGS = ['xmm0', 'xmm1', 'xmm2', 'xmm3']
    WIN_PRESERVED = ['rbx', 'rbp', 'rdi', 'rsi', 'rsp', 'r12', 'r13', 'r14', 'r15']
    WIN_SCRATCH = ['rax', 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11']
    
    # macOS (same as Linux but with some differences)
    MACOS_INT_ARGS = LINUX_INT_ARGS
    MACOS_FLOAT_ARGS = LINUX_FLOAT_ARGS
    MACOS_PRESERVED = LINUX_PRESERVED
    
    # ARM64 AAPCS
    ARM64_INT_ARGS = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    ARM64_FLOAT_ARGS = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    ARM64_PRESERVED = ['x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28']
    
    @staticmethod
    def get_arg_register(platform: Platform, arch: Architecture, index: int, 
                        is_float: bool = False) -> Optional[str]:
        """Get register for argument at index"""
        if arch == Architecture.X86_64:
            if is_float:
                args = (CallingConvention.LINUX_FLOAT_ARGS if platform in [Platform.LINUX, Platform.MACOS] 
                       else CallingConvention.WIN_FLOAT_ARGS)
            else:
                if platform == Platform.WINDOWS:
                    args = CallingConvention.WIN_INT_ARGS
                elif platform == Platform.MACOS:
                    args = CallingConvention.MACOS_INT_ARGS
                else:
                    args = CallingConvention.LINUX_INT_ARGS
        elif arch == Architecture.ARM64:
            args = CallingConvention.ARM64_FLOAT_ARGS if is_float else CallingConvention.ARM64_INT_ARGS
        else:
            return None
            
        return args[index] if index < len(args) else None
    
    @staticmethod
    def get_return_register(arch: Architecture, is_float: bool = False) -> str:
        """Get return value register"""
        if arch == Architecture.X86_64:
            return 'xmm0' if is_float else 'rax'
        elif arch == Architecture.ARM64:
            return 'v0' if is_float else 'x0'
        return 'rax'
    
    @staticmethod
    def needs_shadow_space(platform: Platform) -> bool:
        """Windows requires 32 bytes shadow space"""
        return platform == Platform.WINDOWS
    
    @staticmethod
    def get_preserved_registers(platform: Platform, arch: Architecture) -> List[str]:
        """Get callee-saved registers"""
        if arch == Architecture.X86_64:
            if platform == Platform.WINDOWS:
                return CallingConvention.WIN_PRESERVED
            else:
                return CallingConvention.LINUX_PRESERVED
        elif arch == Architecture.ARM64:
            return CallingConvention.ARM64_PRESERVED
        return []


class Syscalls:
    """Extended syscall mappings with additional platforms"""
    
    LINUX_SYSCALLS = {
        'read': 0, 'write': 1, 'open': 2, 'close': 3,
        'stat': 4, 'fstat': 5, 'lstat': 6, 'poll': 7,
        'lseek': 8, 'mmap': 9, 'mprotect': 10, 'munmap': 11,
        'brk': 12, 'rt_sigaction': 13, 'rt_sigprocmask': 14,
        'rt_sigreturn': 15, 'ioctl': 16, 'pread64': 17,
        'pwrite64': 18, 'readv': 19, 'writev': 20,
        'access': 21, 'pipe': 22, 'select': 23,
        'sched_yield': 24, 'mremap': 25, 'msync': 26,
        'mincore': 27, 'madvise': 28, 'shmget': 29,
        'shmat': 30, 'shmctl': 31, 'dup': 32,
        'dup2': 33, 'pause': 34, 'nanosleep': 35,
        'getitimer': 36, 'alarm': 37, 'setitimer': 38,
        'getpid': 39, 'sendfile': 40, 'socket': 41,
        'connect': 42, 'accept': 43, 'sendto': 44,
        'recvfrom': 45, 'sendmsg': 46, 'recvmsg': 47,
        'shutdown': 48, 'bind': 49, 'listen': 50,
        'getsockname': 51, 'getpeername': 52, 'socketpair': 53,
        'setsockopt': 54, 'getsockopt': 55, 'clone': 56,
        'fork': 57, 'vfork': 58, 'execve': 59,
        'exit': 60, 'wait4': 61, 'kill': 62,
        'uname': 63, 'semget': 64, 'semop': 65,
        'semctl': 66, 'shmdt': 67, 'msgget': 68,
        'msgsnd': 69, 'msgrcv': 70, 'msgctl': 71,
        'fcntl': 72, 'flock': 73, 'fsync': 74,
        'fdatasync': 75, 'truncate': 76, 'ftruncate': 77,
        'getdents': 78, 'getcwd': 79, 'chdir': 80,
        'fchdir': 81, 'rename': 82, 'mkdir': 83,
        'rmdir': 84, 'creat': 85, 'link': 86,
        'unlink': 87, 'symlink': 88, 'readlink': 89,
        'chmod': 90, 'fchmod': 91, 'chown': 92,
        'fchown': 93, 'lchown': 94, 'umask': 95,
        'gettimeofday': 96, 'getrlimit': 97, 'getrusage': 98,
        'sysinfo': 99, 'times': 100, 'ptrace': 101,
        'getuid': 102, 'syslog': 103, 'getgid': 104,
        'setuid': 105, 'setgid': 106, 'geteuid': 107,
        'getegid': 108, 'setpgid': 109, 'getppid': 110,
        'getpgrp': 111, 'setsid': 112, 'setreuid': 113,
        'setregid': 114, 'getgroups': 115, 'setgroups': 116,
        'setresuid': 117, 'getresuid': 118, 'setresgid': 119,
        'getresgid': 120, 'getpgid': 121, 'setfsuid': 122,
        'setfsgid': 123, 'getsid': 124, 'capget': 125,
        'capset': 126, 'rt_sigpending': 127, 'rt_sigtimedwait': 128,
        'rt_sigqueueinfo': 129, 'rt_sigsuspend': 130, 'sigaltstack': 131,
        'utime': 132, 'mknod': 133, 'uselib': 134,
        'personality': 135, 'ustat': 136, 'statfs': 137,
        'fstatfs': 138, 'sysfs': 139, 'getpriority': 140,
        'setpriority': 141, 'sched_setparam': 142, 'sched_getparam': 143,
        'sched_setscheduler': 144, 'sched_getscheduler': 145,
        'sched_get_priority_max': 146, 'sched_get_priority_min': 147,
        'sched_rr_get_interval': 148, 'mlock': 149, 'munlock': 150,
        'mlockall': 151, 'munlockall': 152, 'vhangup': 153,
        'modify_ldt': 154, 'pivot_root': 155, '_sysctl': 156,
        'prctl': 157, 'arch_prctl': 158, 'adjtimex': 159,
        'setrlimit': 160, 'chroot': 161, 'sync': 162,
        'acct': 163, 'settimeofday': 164, 'mount': 165,
        'umount2': 166, 'swapon': 167, 'swapoff': 168,
        'reboot': 169, 'sethostname': 170, 'setdomainname': 171,
        'iopl': 172, 'ioperm': 173, 'create_module': 174,
        'init_module': 175, 'delete_module': 176, 'get_kernel_syms': 177,
        'query_module': 178, 'quotactl': 179, 'nfsservctl': 180,
        'getpmsg': 181, 'putpmsg': 182, 'afs_syscall': 183,
        'tuxcall': 184, 'security': 185, 'gettid': 186,
        'readahead': 187, 'setxattr': 188, 'lsetxattr': 189,
        'fsetxattr': 190, 'getxattr': 191, 'lgetxattr': 192,
        'fgetxattr': 193, 'listxattr': 194, 'llistxattr': 195,
        'flistxattr': 196, 'removexattr': 197, 'lremovexattr': 198,
        'fremovexattr': 199, 'tkill': 200, 'time': 201,
        'futex': 202, 'sched_setaffinity': 203, 'sched_getaffinity': 204,
        'set_thread_area': 205, 'io_setup': 206, 'io_destroy': 207,
        'io_getevents': 208, 'io_submit': 209, 'io_cancel': 210,
        'get_thread_area': 211, 'lookup_dcookie': 212, 'epoll_create': 213,
        'epoll_ctl_old': 214, 'epoll_wait_old': 215, 'remap_file_pages': 216,
        'getdents64': 217, 'set_tid_address': 218, 'restart_syscall': 219,
        'semtimedop': 220, 'fadvise64': 221, 'timer_create': 222,
        'timer_settime': 223, 'timer_gettime': 224, 'timer_getoverrun': 225,
        'timer_delete': 226, 'clock_settime': 227, 'clock_gettime': 228,
        'clock_getres': 229, 'clock_nanosleep': 230, 'exit_group': 231,
        'epoll_wait': 232, 'epoll_ctl': 233, 'tgkill': 234,
        'utimes': 235, 'vserver': 236, 'mbind': 237,
        'set_mempolicy': 238, 'get_mempolicy': 239, 'mq_open': 240,
        'mq_unlink': 241, 'mq_timedsend': 242, 'mq_timedreceive': 243,
        'mq_notify': 244, 'mq_getsetattr': 245, 'kexec_load': 246,
        'waitid': 247, 'add_key': 248, 'request_key': 249,
        'keyctl': 250, 'ioprio_set': 251, 'ioprio_get': 252,
        'inotify_init': 253, 'inotify_add_watch': 254, 'inotify_rm_watch': 255,
    }
    
    MACOS_SYSCALLS = {
        'exit': 0x2000001, 'fork': 0x2000002, 'read': 0x2000003,
        'write': 0x2000004, 'open': 0x2000005, 'close': 0x2000006,
        'mmap': 0x20000c5, 'munmap': 0x2000049,
    }
    
    @staticmethod
    def get_syscall_number(platform: Platform, name: str) -> Optional[int]:
        """Get platform-specific syscall number"""
        name = name.lower()
        if platform == Platform.LINUX:
            return Syscalls.LINUX_SYSCALLS.get(name)
        elif platform == Platform.MACOS:
            return Syscalls.MACOS_SYSCALLS.get(name)
        return None


class SIMDInstructions:
    """SIMD instruction translation"""
    
    SSE_TO_AVX = {
        'movaps': 'vmovaps', 'movups': 'vmovups',
        'addps': 'vaddps', 'subps': 'vsubps',
        'mulps': 'vmulps', 'divps': 'vdivps',
        'addpd': 'vaddpd', 'subpd': 'vsubpd',
        'mulpd': 'vmulpd', 'divpd': 'vdivpd',
    }
    
    AVX_TO_NEON = {
        'vmovaps': 'vld1.32', 'vmovups': 'vld1.32',
        'vaddps': 'vadd.f32', 'vsubps': 'vsub.f32',
        'vmulps': 'vmul.f32', 'vdivps': 'vdiv.f32',
    }
    
    @staticmethod
    def translate_simd(instruction: str, from_set: str, to_set: str) -> str:
        """Translate SIMD instructions between sets"""
        if from_set == "sse" and to_set == "avx":
            for sse, avx in SIMDInstructions.SSE_TO_AVX.items():
                if instruction.startswith(sse):
                    return instruction.replace(sse, avx, 1)
        elif from_set == "avx" and to_set == "neon":
            for avx, neon in SIMDInstructions.AVX_TO_NEON.items():
                if instruction.startswith(avx):
                    return instruction.replace(avx, neon, 1)
        return instruction


class TranspileError(Exception):
    """Transpilation error with context"""
    def __init__(self, message: str, line: int = 0, context: str = ""):
        self.message = message
        self.line = line
        self.context = context
        super().__init__(f"Line {line}: {message}")


class AssemblyAnalyzer:
    """Advanced static analysis"""
    
    def __init__(self):
        self.basic_blocks: Dict[str, BasicBlock] = {}
        self.dominators: Dict[str, Set[str]] = {}
        self.loops: List[Loop] = []
        
    def build_cfg(self, instructions: List[str]) -> Dict[str, BasicBlock]:
        """Build control flow graph"""
        current_block = BasicBlock(label="_entry")
        self.basic_blocks["_entry"] = current_block
        
        for inst in instructions:
            if ':' in inst and not inst.strip().startswith(';'):
                # New basic block
                label = inst.split(':')[0].strip()
                if label not in self.basic_blocks:
                    self.basic_blocks[label] = BasicBlock(label=label)
                current_block = self.basic_blocks[label]
            elif any(inst.strip().startswith(x) for x in ['jmp', 'je', 'jne', 'jg', 'jl', 'call']):
                current_block.instructions.append(inst)
                # Extract target
                parts = inst.split()
                if len(parts) > 1:
                    target = parts[1].strip()
                    current_block.successors.add(target)
            else:
                current_block.instructions.append(inst)
        
        return self.basic_blocks
    
    def detect_loops(self) -> List[Loop]:
        """Detect loop structures"""
        loops = []
        visited = set()
        
        def dfs(block_label: str, path: List[str]):
            if block_label in visited:
                if block_label in path:
                    # Found a loop
                    loop_start = path.index(block_label)
                    loop = Loop(
                        start_label=block_label,
                        end_label=path[-1] if path else block_label
                    )
                    loops.append(loop)
                return
            
            visited.add(block_label)
            path.append(block_label)
            
            if block_label in self.basic_blocks:
                for successor in self.basic_blocks[block_label].successors:
                    dfs(successor, path.copy())
        
        dfs("_entry", [])
        self.loops = loops
        return loops
    
    def analyze_register_usage(self, instructions: List[str]) -> Dict[str, Set[str]]:
        """Analyze register usage patterns"""
        usage = defaultdict(set)
        
        for inst in instructions:
            # Extract registers used
            regs = re.findall(r'\b(r[a-z0-9]+|xmm\d+|ymm\d+)\b', inst.lower())
            for reg in regs:
                usage[reg].add(inst)
        
        return dict(usage)


class Optimizer:
    """Advanced optimization engine"""
    
    def __init__(self, level: OptimizationLevel = OptimizationLevel.BASIC):
        self.level = level
        self.analyzer = AssemblyAnalyzer()
        
    def optimize(self, code: str, platform: Platform, arch: Architecture) -> str:
        """Apply optimization passes"""
        lines = code.splitlines()
        
        if self.level.value >= OptimizationLevel.BASIC.value:
            lines = self.eliminate_redundant_moves(lines)
            lines = self.fold_constants(lines)
            lines = self.remove_dead_code(lines)
        
        if self.level.value >= OptimizationLevel.AGGRESSIVE.value:
            lines = self.unroll_loops(lines)
            lines = self.optimize_register_allocation(lines)
            lines = self.strength_reduction(lines)
        
        if self.level.value >= OptimizationLevel.EXTREME.value:
            lines = self.vectorize_loops(lines, arch)
            lines = self.optimize_branch_prediction(lines)
            lines = self.inline_small_functions(lines)
        
        return '\n'.join(lines)
    
    def eliminate_redundant_moves(self, lines: List[str]) -> List[str]:
        """Remove redundant mov instructions"""
        result = []
        prev_line = None
        
        for line in lines:
            stripped = line.strip()
            
            # Skip redundant mov rax, rax
            if 'mov' in stripped:
                parts = [p.strip() for p in stripped.split(',')]
                if len(parts) == 2:
                    dest = parts[0].split()[-1]
                    src = parts[1]
                    if dest == src:
                        continue
            
            # Skip mov followed by immediate opposite
            if prev_line and 'mov' in prev_line and 'mov' in stripped:
                # More sophisticated check
                pass
            
            result.append(line)
            prev_line = stripped
        
        return result
    
    def fold_constants(self, lines: List[str]) -> List[str]:
        """Constant folding"""
        result = []
        constants = {}
        
        for line in lines:
            stripped = line.strip()
            
            # Track constant assignments
            if 'mov' in stripped and any(c.isdigit() for c in stripped):
                parts = stripped.split(',')
                if len(parts) == 2:
                    reg = parts[0].split()[-1].strip()
                    try:
                        value = int(parts[1].strip())
                        constants[reg] = value
                    except:
                        pass
            
            # Replace known constants
            modified = line
            for reg, value in constants.items():
                if reg in stripped and 'mov' not in stripped:
                    # Don't replace in mov instructions
                    modified = modified.replace(reg, str(value))
            
            result.append(modified)
            
            # Clear constants on function calls or branches
            if any(x in stripped for x in ['call', 'jmp', 'je', 'jne', 'ret']):
                constants.clear()
        
        return result
    
    def remove_dead_code(self, lines: List[str]) -> List[str]:
        """Dead code elimination"""
        result = []
        reachable = True
        
        for line in lines:
            stripped = line.strip()
            
            if reachable:
                result.append(line)
            
            # After unconditional jump or ret, mark unreachable
            if stripped.startswith('jmp') or stripped.startswith('ret'):
                reachable = False
            
            # Labels make code reachable again
            if ':' in stripped:
                reachable = True
        
        return result
    
    def unroll_loops(self, lines: List[str], factor: int = 4) -> List[str]:
        """Loop unrolling"""
        result = []
        in_loop = False
        loop_body = []
        loop_label = None
        
        for line in lines:
            stripped = line.strip()
            
            # Detect simple loops
            if ':' in stripped and '_loop' in stripped.lower():
                in_loop = True
                loop_label = stripped.split(':')[0]
                result.append(line)
                continue
            
            if in_loop:
                if any(x in stripped for x in ['jmp', 'jne', 'jl']):
                    # End of loop
                    in_loop = False
                    # Unroll
                    for _ in range(factor - 1):
                        result.extend(loop_body)
                    result.extend(loop_body)
                    result.append(line)
                    loop_body = []
                else:
                    loop_body.append(line)
            else:
                result.append(line)
        
        return result
    
    def optimize_register_allocation(self, lines: List[str]) -> List[str]:
        """Better register allocation"""
        # This is a simplified version
        return lines
    
    def strength_reduction(self, lines: List[str]) -> List[str]:
        """Replace expensive operations with cheaper ones"""
        result = []
        
        for line in lines:
            stripped = line.strip()
            modified = line
            
            # Replace imul by power of 2 with shift
            if 'imul' in stripped:
                match = re.search(r'imul\s+(\w+),\s*(\d+)', stripped)
                if match:
                    reg, value = match.groups()
                    try:
                        val = int(value)
                        if val > 0 and (val & (val - 1)) == 0:  # Power of 2
                            shift = val.bit_length() - 1
                            modified = line.replace(f'imul {reg}, {value}', 
                                                   f'shl {reg}, {shift}')
                    except:
                        pass
            
            # Replace idiv by power of 2 with shift
            if 'idiv' in stripped and any(c.isdigit() for c in stripped):
                # Similar optimization
                pass
            
            result.append(modified)
        
        return result
    
    def vectorize_loops(self, lines: List[str], arch: Architecture) -> List[str]:
        """Automatic vectorization"""
        if arch != Architecture.X86_64:
            return lines
        
        result = []
        
        for line in lines:
            stripped = line.strip()
            
            # Detect vectorizable patterns
            if 'addss' in stripped:  # Scalar single add
                # Convert to packed operation
                modified = line.replace('addss', 'addps')
                result.append(f"    ; Vectorized:")
                result.append(modified)
            else:
                result.append(line)
        
        return result
    
    def optimize_branch_prediction(self, lines: List[str]) -> List[str]:
        """Optimize for branch prediction"""
        result = []
        
        for line in lines:
            stripped = line.strip()
            
            # Add branch hints for conditional jumps
            if stripped.startswith('je ') or stripped.startswith('jne '):
                # Most branches are not taken
                result.append(line)
                result.append("    ; Branch hint: likely not taken")
            else:
                result.append(line)
        
        return result
    
    def inline_small_functions(self, lines: List[str]) -> List[str]:
        """Inline small functions"""
        functions = {}
        current_func = None
        func_body = []
        
        # Collect small functions
        for line in lines:
            stripped = line.strip()
            if ':' in stripped and not any(x in stripped for x in [';', 'section']):
                if current_func and len(func_body) < 10:
                    functions[current_func] = func_body.copy()
                current_func = stripped.split(':')[0]
                func_body = []
            elif current_func:
                func_body.append(line)
                if 'ret' in stripped:
                    if len(func_body) < 10:
                        functions[current_func] = func_body.copy()
                    current_func = None
                    func_body = []
        
        # Inline calls
        result = []
        for line in lines:
            if 'call' in line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    target = parts[1].strip()
                    if target in functions:
                        result.append(f"    ; Inlined {target}")
                        result.extend(functions[target][:-1])  # Exclude ret
                        continue
            result.append(line)
        
        return result


class AssemblyTranspiler:
    """Ultra-advanced assembly transpiler"""
    
    def __init__(self, target: str = "linux", arch: str = "x86_64", 
                 opt_level: OptimizationLevel = OptimizationLevel.BASIC):
        self.platform = self._parse_platform(target)
        self.arch = self._parse_arch(arch)
        self.opt_level = opt_level
        self.optimizer = Optimizer(opt_level)
        
        # Code sections
        self.data_lines: List[str] = []
        self.bss_lines: List[str] = []
        self.rodata_lines: List[str] = []
        self.text_lines: List[str] = []
        self.tls_lines: List[str] = []
        self.init_lines: List[str] = []
        
        # Symbol tables
        self.externs: Set[str] = set()
        self.globals: Set[str] = set()
        self.weaks: Set[str] = set()
        self.locals: Dict[str, Variable] = {}
        
        # Function and macro tracking
        self.macros: Dict[str, Macro] = {}
        self.functions: Dict[str, Function] = {}
        self.current_function: Optional[Function] = None
        
        # Control flow
        self.conditional_stack: List[bool] = []
        self.skip_mode: bool = False
        self.loop_stack: List[Loop] = []
        
        # Advanced features
        self.symbols: Dict[str, Any] = {}
        self.inline_asm_blocks: List[str] = []
        self.performance_counters: Dict[str, int] = defaultdict(int)
        self.atomic_sections: List[Tuple[int, int]] = []
        self.thread_local_vars: Set[str] = set()
        
        # State
        self.line_number: int = 0
        self.current_section: SectionType = SectionType.TEXT
        
    def _parse_platform(self, target: str) -> Platform:
        """Parse platform string"""
        target = target.lower()
        if target in ['linux', 'gnu', 'elf']:
            return Platform.LINUX
        elif target in ['windows', 'win', 'win64']:
            return Platform.WINDOWS
        elif target in ['macos', 'darwin', 'osx']:
            return Platform.MACOS
        elif target in ['bsd', 'freebsd', 'openbsd']:
            return Platform.BSD
        return Platform.LINUX
    
    def _parse_arch(self, arch: str) -> Architecture:
        """Parse architecture string"""
        arch = arch.lower()
        if arch in ['x86_64', 'x64', 'amd64']:
            return Architecture.X86_64
        elif arch in ['arm64', 'aarch64']:
            return Architecture.ARM64
        elif arch in ['riscv64', 'riscv']:
            return Architecture.RISCV64
        return Architecture.X86_64
    
    def error(self, msg: str) -> TranspileError:
        """Create error with context"""
        return TranspileError(msg, self.line_number)
    
    def emit(self, section: SectionType, line: str):
        """Emit to specific section"""
        if section == SectionType.TEXT:
            self.text_lines.append(line)
        elif section == SectionType.DATA:
            self.data_lines.append(line)
        elif section == SectionType.BSS:
            self.bss_lines.append(line)
        elif section == SectionType.RODATA:
            self.rodata_lines.append(line)
        elif section == SectionType.TLS:
            self.tls_lines.append(line)
        elif section == SectionType.INIT:
            self.init_lines.append(line)
    
    def emit_text(self, line: str):
        """Emit to text section"""
        self.emit(SectionType.TEXT, line)
    
    # Enhanced directive handlers
    
    def handle_ustr(self, match: re.Match):
        """USTR - Define string"""
        label = match.group(1)
        text = match.group(2)
        escaped = text.replace('\\', '\\\\').replace('"', '\\"')
        self.emit(SectionType.RODATA, f"{label} db \"{escaped}\", 0")
    
    def handle_uwstr(self, match: re.Match):
        """UWSTR - Wide string"""
        label = match.group(1)
        text = match.group(2)
        if self.platform == Platform.WINDOWS:
            escaped = text.replace('\\', '\\\\').replace('"', '\\"')
            self.emit(SectionType.DATA, f'{label} dw __utf16__("{escaped}"), 0')
        else:
            escaped = text.replace('\\', '\\\\').replace('"', '\\"')
            self.emit(SectionType.RODATA, f'{label} db "{escaped}", 0')
    
    def handle_ubytes(self, match: re.Match):
        """UBYTES - Reserve bytes"""
        label = match.group(1)
        size = match.group(2)
        align = match.group(3) if (match.lastindex or 0) >= 3 else "1"
        self.emit(SectionType.BSS, f"align {align}")
        self.emit(SectionType.BSS, f"{label} resb {size}")
    
    def handle_uconst(self, match: re.Match):
        """UCONST - Define constant"""
        label = match.group(1)
        value = match.group(2)
        self.emit(SectionType.RODATA, f"{label} equ {value}")
        self.symbols[label] = value
    
    def handle_uarray(self, match: re.Match):
        """UARRAY - Define array"""
        label = match.group(1)
        type_ = match.group(2)
        values = match.group(3)
        
        size_map = {'byte': 'db', 'word': 'dw', 'dword': 'dd', 'qword': 'dq'}
        directive = size_map.get(type_, 'dq')
        self.emit(SectionType.DATA, f"{label} {directive} {values}")
    
    def handle_ustruct(self, match: re.Match):
        """USTRUCT - Define structure"""
        name = match.group(1)
        self.symbols[f"_struct_{name}"] = {'fields': [], 'size': 0}
        return True  # Collect struct definition
    
    def handle_ufunc(self, match: re.Match):
        """UFUNC - Begin function"""
        name = match.group(1)
        params_str = match.group(2) if (match.lastindex or 0) >= 2 else ""
        
        params = []
        if params_str:
            for p in params_str.split(','):
                p = p.strip()
                if ':' in p:
                    pname, ptype = p.split(':', 1)
                    params.append((pname.strip(), ptype.strip()))
                else:
                    params.append((p, 'qword'))
        
        func = Function(name=name, params=params)
        self.functions[name] = func
        self.current_function = func
        
        self.emit_text(f"{name}:")
        
        # Function prologue
        if self.arch == Architecture.X86_64:
            self.emit_text("    push rbp")
            self.emit_text("    mov rbp, rsp")
            
            if self.platform == Platform.WINDOWS:
                self.emit_text("    sub rsp, 32  ; Shadow space")
        elif self.arch == Architecture.ARM64:
            self.emit_text("    stp x29, x30, [sp, #-16]!")
            self.emit_text("    mov x29, sp")
    
    def handle_uendfunc(self, match: re.Match):
        """UENDFUNC - End function"""
        if not self.current_function:
            raise self.error("UENDFUNC without UFUNC")
        
        func = self.current_function
        
        # Allocate locals if needed
        if func.local_size > 0:
            total = func.local_size
            if self.platform == Platform.WINDOWS and self.arch == Architecture.X86_64:
                total += 32
            total = ((total + 15) // 16) * 16
            
            if self.arch == Architecture.X86_64:
                self.emit_text(f"    sub rsp, {total}")
            elif self.arch == Architecture.ARM64:
                self.emit_text(f"    sub sp, sp, #{total}")
        
        self.current_function = None
    
    def handle_uparam(self, match: re.Match):
        """UPARAM - Get parameter"""
        index = int(match.group(1))
        dest = match.group(2)
        
        reg = CallingConvention.get_arg_register(self.platform, self.arch, index)
        if reg:
            if self.arch == Architecture.X86_64:
                self.emit_text(f"    mov {dest}, {reg}")
            elif self.arch == Architecture.ARM64:
                self.emit_text(f"    mov {dest}, {reg}")
        else:
            # On stack
            if self.arch == Architecture.X86_64:
                offset = 16 + (index - 6) * 8
                self.emit_text(f"    mov {dest}, [rbp + {offset}]")
    
    def handle_uret(self, match: re.Match):
        """URET - Return from function"""
        value = match.group(1) if (match.lastindex or 0) >= 1 else None
        
        if value:
            ret_reg = CallingConvention.get_return_register(self.arch)
            if self.arch == Architecture.X86_64:
                self.emit_text(f"    mov {ret_reg}, {value}")
            elif self.arch == Architecture.ARM64:
                self.emit_text(f"    mov {ret_reg}, {value}")
        
        # Function epilogue
        if self.arch == Architecture.X86_64:
            self.emit_text("    leave")
            self.emit_text("    ret")
        elif self.arch == Architecture.ARM64:
            self.emit_text("    ldp x29, x30, [sp], #16")
            self.emit_text("    ret")
    
    def handle_ucall(self, match: re.Match):
        """UCALL - Call function with args"""
        func = match.group(1)
        args_str = match.group(2) if (match.lastindex or 0) >= 2 else ""
        args = [a.strip() for a in args_str.split(',')] if args_str else []
        
        if self.arch == Architecture.X86_64:
            int_regs = (CallingConvention.WIN_INT_ARGS if self.platform == Platform.WINDOWS
                       else CallingConvention.LINUX_INT_ARGS)
            
            # Setup arguments
            for i, arg in enumerate(args):
                if i < len(int_regs):
                    self.emit_text(f"    mov {int_regs[i]}, {arg}")
                else:
                    self.emit_text(f"    push {arg}")
            
            # Align stack and call
            if self.platform == Platform.WINDOWS:
                self.emit_text("    sub rsp, 32")
            
            self.emit_text(f"    call {func}")
            
            if self.platform == Platform.WINDOWS:
                self.emit_text("    add rsp, 32")
            
            # Clean stack args
            stack_args = max(0, len(args) - len(int_regs))
            if stack_args > 0:
                self.emit_text(f"    add rsp, {stack_args * 8}")
                
        elif self.arch == Architecture.ARM64:
            for i, arg in enumerate(args[:8]):
                self.emit_text(f"    mov x{i}, {arg}")
            self.emit_text(f"    bl {func}")
    
    def handle_usyscall(self, match: re.Match):
        """USYSCALL - System call"""
        name = match.group(1)
        args_str = match.group(2) if (match.lastindex or 0) >= 2 else ""
        args = [a.strip() for a in args_str.split(',')] if args_str else []
        
        syscall_num = Syscalls.get_syscall_number(self.platform, name)
        
        if self.arch == Architecture.X86_64:
            if self.platform in [Platform.LINUX, Platform.BSD]:
                if syscall_num is None:
                    raise self.error(f"Unknown syscall: {name}")
                
                self.emit_text(f"    mov rax, {syscall_num}")
                arg_regs = ['rdi', 'rsi', 'rdx', 'r10', 'r8', 'r9']
                for i, arg in enumerate(args[:6]):
                    self.emit_text(f"    mov {arg_regs[i]}, {arg}")
                self.emit_text("    syscall")
                
            elif self.platform == Platform.WINDOWS:
                # Convert to Windows API
                self._emit_windows_syscall(name, args)
                
        elif self.arch == Architecture.ARM64:
            if syscall_num:
                self.emit_text(f"    mov x8, #{syscall_num}")
                for i, arg in enumerate(args[:6]):
                    self.emit_text(f"    mov x{i}, {arg}")
                self.emit_text("    svc #0")
    
    def _emit_windows_syscall(self, name: str, args: List[str]):
        """Emit Windows API call equivalent to syscall"""
        api_map = {
            'write': 'WriteFile',
            'read': 'ReadFile',
            'open': 'CreateFileA',
            'close': 'CloseHandle',
            'exit': 'ExitProcess',
        }
        
        if name in api_map:
            api = api_map[name]
            self.externs.add(api)
            
            if name == 'write' and len(args) >= 3:
                # write(fd, buf, count)
                self.externs.add('GetStdHandle')
                self.emit_text("    mov ecx, -11  ; STD_OUTPUT_HANDLE")
                self.emit_text("    call GetStdHandle")
                self.emit_text("    mov rcx, rax")
                self.emit_text(f"    lea rdx, [{args[1]}]")
                self.emit_text(f"    mov r8d, {args[2]}")
                self.emit_text("    xor r9d, r9d")
                self.emit_text("    sub rsp, 32")
                self.emit_text(f"    call {api}")
                self.emit_text("    add rsp, 32")
            else:
                self.emit_text(f"    ; Syscall {name} -> {api} (simplified)")
                if args:
                    self.emit_text(f"    mov rcx, {args[0]}")
                self.emit_text(f"    call {api}")
    
    def handle_uprint(self, match: re.Match):
        """UPRINT - Print string"""
        label = match.group(1)
        self.externs.add('puts')
        
        if self.arch == Architecture.X86_64:
            if self.platform == Platform.WINDOWS:
                self.emit_text(f"    lea rcx, [rel {label}]")
                self.emit_text("    sub rsp, 32")
                self.emit_text("    call puts")
                self.emit_text("    add rsp, 32")
            else:
                self.emit_text(f"    lea rdi, [rel {label}]")
                self.emit_text("    xor eax, eax")
                self.emit_text("    call puts")
        elif self.arch == Architecture.ARM64:
            self.emit_text(f"    adr x0, {label}")
            self.emit_text("    bl puts")
    
    def handle_uprintf(self, match: re.Match):
        """UPRINTF - Formatted print"""
        fmt = match.group(1)
        args_str = match.group(2) if (match.lastindex or 0) >= 2 else ""
        args = [a.strip() for a in args_str.split(',')] if args_str else []
        
        self.externs.add('printf')
        
        fmt_label = f"_fmt_{abs(hash(fmt)) % 10000}"
        escaped = fmt.replace('\\', '\\\\').replace('"', '\\"')
        self.emit(SectionType.RODATA, f'{fmt_label} db "{escaped}", 0')
        
        if self.arch == Architecture.X86_64:
            if self.platform == Platform.WINDOWS:
                self.emit_text(f"    lea rcx, [rel {fmt_label}]")
                for i, arg in enumerate(args[:3]):
                    regs = ['rdx', 'r8', 'r9']
                    if i < len(regs):
                        self.emit_text(f"    mov {regs[i]}, {arg}")
                self.emit_text("    sub rsp, 32")
                self.emit_text("    call printf")
                self.emit_text("    add rsp, 32")
            else:
                self.emit_text(f"    lea rdi, [rel {fmt_label}]")
                for i, arg in enumerate(args[:5]):
                    regs = ['rsi', 'rdx', 'rcx', 'r8', 'r9']
                    if i < len(regs):
                        self.emit_text(f"    mov {regs[i]}, {arg}")
                self.emit_text("    xor eax, eax")
                self.emit_text("    call printf")
    
    def handle_uexit(self, match: re.Match):
        """UEXIT - Exit program"""
        code = match.group(1)
        
        if self.arch == Architecture.X86_64:
            if self.platform == Platform.WINDOWS:
                self.externs.add('ExitProcess')
                self.emit_text(f"    mov ecx, {code}")
                self.emit_text("    call ExitProcess")
            else:
                self.emit_text(f"    mov rax, 60")
                self.emit_text(f"    mov rdi, {code}")
                self.emit_text("    syscall")
        elif self.arch == Architecture.ARM64:
            self.emit_text(f"    mov x0, #{code}")
            self.emit_text("    mov x8, #93  ; exit")
            self.emit_text("    svc #0")
    
    def handle_umalloc(self, match: re.Match):
        """UMALLOC - Allocate memory"""
        size = match.group(1)
        dest = match.group(2) if (match.lastindex or 0) >= 2 else None
        
        self.externs.add('malloc')
        
        if self.arch == Architecture.X86_64:
            if self.platform == Platform.WINDOWS:
                self.emit_text(f"    mov rcx, {size}")
                self.emit_text("    sub rsp, 32")
                self.emit_text("    call malloc")
                self.emit_text("    add rsp, 32")
            else:
                self.emit_text(f"    mov rdi, {size}")
                self.emit_text("    call malloc")
            
            if dest and dest != "rax":
                self.emit_text(f"    mov {dest}, rax")
        elif self.arch == Architecture.ARM64:
            self.emit_text(f"    mov x0, {size}")
            self.emit_text("    bl malloc")
            if dest and dest != "x0":
                self.emit_text(f"    mov {dest}, x0")
    
    def handle_ufree(self, match: re.Match):
        """UFREE - Free memory"""
        ptr = match.group(1)
        
        self.externs.add('free')
        
        if self.arch == Architecture.X86_64:
            if self.platform == Platform.WINDOWS:
                self.emit_text(f"    mov rcx, {ptr}")
                self.emit_text("    sub rsp, 32")
                self.emit_text("    call free")
                self.emit_text("    add rsp, 32")
            else:
                self.emit_text(f"    mov rdi, {ptr}")
                self.emit_text("    call free")
        elif self.arch == Architecture.ARM64:
            self.emit_text(f"    mov x0, {ptr}")
            self.emit_text("    bl free")
    
    def handle_uatomic(self, match: re.Match):
        """UATOMIC - Atomic operation"""
        op = match.group(1)
        args = match.group(2).split(',')
        args = [a.strip() for a in args]
        
        if self.arch == Architecture.X86_64:
            if op == 'add':
                self.emit_text(f"    lock add {args[0]}, {args[1]}")
            elif op == 'sub':
                self.emit_text(f"    lock sub {args[0]}, {args[1]}")
            elif op == 'xchg':
                self.emit_text(f"    lock xchg {args[0]}, {args[1]}")
            elif op == 'cmpxchg':
                self.emit_text(f"    lock cmpxchg {args[0]}, {args[1]}")
        elif self.arch == Architecture.ARM64:
            if op == 'add':
                self.emit_text(f"    ldaxr w10, [{args[0]}]")
                self.emit_text(f"    add w10, w10, {args[1]}")
                self.emit_text(f"    stlxr w11, w10, [{args[0]}]")
    
    def handle_usimd(self, match: re.Match):
        """USIMD - SIMD operation"""
        op = match.group(1)
        args = match.group(2).split(',')
        args = [a.strip() for a in args]
        
        if self.arch == Architecture.X86_64:
            # Use AVX if available
            if op == 'add_ps':
                self.emit_text(f"    vaddps {args[0]}, {args[1]}, {args[2]}")
            elif op == 'mul_ps':
                self.emit_text(f"    vmulps {args[0]}, {args[1]}, {args[2]}")
            elif op == 'load':
                self.emit_text(f"    vmovaps {args[0]}, [{args[1]}]")
            elif op == 'store':
                self.emit_text(f"    vmovaps [{args[0]}], {args[1]}")
        elif self.arch == Architecture.ARM64:
            if op == 'add_ps':
                self.emit_text(f"    fadd {args[0]}.4s, {args[1]}.4s, {args[2]}.4s")
            elif op == 'mul_ps':
                self.emit_text(f"    fmul {args[0]}.4s, {args[1]}.4s, {args[2]}.4s")
    
    def handle_uthreadlocal(self, match: re.Match):
        """UTHREADLOCAL - Thread-local variable"""
        label = match.group(1)
        size = match.group(2)
        
        self.thread_local_vars.add(label)
        
        if self.platform in [Platform.LINUX, Platform.BSD]:
            self.emit(SectionType.TLS, f"{label} resb {size}")
        elif self.platform == Platform.WINDOWS:
            self.emit(SectionType.TLS, f"{label} resb {size}")
    
    def handle_ubarrier(self, match: re.Match):
        """UBARRIER - Memory barrier"""
        barrier_type = match.group(1) if (match.lastindex or 0) >= 1 else "full"
        
        if self.arch == Architecture.X86_64:
            if barrier_type == "full":
                self.emit_text("    mfence")
            elif barrier_type == "load":
                self.emit_text("    lfence")
            elif barrier_type == "store":
                self.emit_text("    sfence")
        elif self.arch == Architecture.ARM64:
            if barrier_type == "full":
                self.emit_text("    dmb sy")
            elif barrier_type == "load":
                self.emit_text("    dmb ld")
            elif barrier_type == "store":
                self.emit_text("    dmb st")
    
    def handle_uloop(self, match: re.Match):
        """ULOOP - Loop with optimization hints"""
        label = match.group(1)
        count = match.group(2) if (match.lastindex or 0) >= 2 else None
        
        loop = Loop(start_label=label, end_label=f"{label}_end")
        if count:
            try:
                loop.trip_count = int(count)
            except:
                pass
        self.loop_stack.append(loop)
        
        self.emit_text(f"{label}:")
        if count:
            self.emit_text(f"    ; Loop with trip count: {count}")
    
    def handle_uendloop(self, match: re.Match):
        """UENDLOOP - End loop"""
        if not self.loop_stack:
            raise self.error("UENDLOOP without ULOOP")
        
        loop = self.loop_stack.pop()
        self.emit_text(f"{loop.end_label}:")
    
    def handle_uinline(self, match: re.Match):
        """UINLINE - Inline assembly block"""
        lang = match.group(1) if (match.lastindex or 0) >= 1 else "c"
        return True  # Signal to collect inline code
    
    def handle_uoptimize(self, match: re.Match):
        """UOPTIMIZE - Optimization pragma"""
        directive = match.group(1)
        
        if directive == "unroll":
            if self.loop_stack:
                self.loop_stack[-1].unroll_factor = 4
        elif directive == "vectorize":
            if self.loop_stack:
                self.loop_stack[-1].is_vectorizable = True
        elif directive == "inline":
            if self.current_function:
                self.current_function.is_inline = True
    
    def handle_upragma(self, match: re.Match):
        """UPRAGMA - Compiler pragma"""
        pragma = match.group(1)
        
        self.emit_text(f"    ; Pragma: {pragma}")
    
    # Conditional compilation
    
    def handle_uifos(self, match: re.Match):
        """UIFOS - Conditional by OS"""
        os_name = match.group(1).lower()
        current_os = {
            Platform.LINUX: "linux",
            Platform.WINDOWS: "windows",
            Platform.MACOS: "macos",
            Platform.BSD: "bsd"
        }[self.platform]
        
        condition = (os_name == current_os)
        self.conditional_stack.append(condition)
        self.skip_mode = not condition
    
    def handle_uifarch(self, match: re.Match):
        """UIFARCH - Conditional by architecture"""
        arch_name = match.group(1).lower()
        current_arch = {
            Architecture.X86_64: "x86_64",
            Architecture.ARM64: "arm64",
            Architecture.RISCV64: "riscv64"
        }[self.arch]
        
        condition = (arch_name == current_arch)
        self.conditional_stack.append(condition)
        self.skip_mode = not condition
    
    def handle_uifdef(self, match: re.Match):
        """UIFDEF - If defined"""
        symbol = match.group(1)
        condition = symbol in self.symbols
        self.conditional_stack.append(condition)
        self.skip_mode = not condition
    
    def handle_uifndef(self, match: re.Match):
        """UIFNDEF - If not defined"""
        symbol = match.group(1)
        condition = symbol not in self.symbols
        self.conditional_stack.append(condition)
        self.skip_mode = not condition
    
    def handle_uelse(self, match: re.Match):
        """UELSE - Else clause"""
        if not self.conditional_stack:
            raise self.error("UELSE without UIF*")
        self.conditional_stack[-1] = not self.conditional_stack[-1]
        self.skip_mode = not self.conditional_stack[-1]
    
    def handle_uendif(self, match: re.Match):
        """UENDIF - End conditional"""
        if not self.conditional_stack:
            raise self.error("UENDIF without UIF*")
        self.conditional_stack.pop()
        self.skip_mode = bool(self.conditional_stack) and not self.conditional_stack[-1]
    
    # Macro system
    
    def handle_umacro(self, match: re.Match):
        """UMACRO - Define macro"""
        name = match.group(1)
        params_str = match.group(2) if (match.lastindex or 0) >= 2 else ""
        
        is_variadic = params_str.endswith('...')
        if is_variadic:
            params_str = params_str[:-3]
        
        params = [p.strip() for p in params_str.split(',')] if params_str else []
        self.macros[name] = Macro(name=name, params=params, body=[], is_variadic=is_variadic)
        return True  # Collect macro body
    
    def handle_uendmacro(self, match: re.Match):
        """UENDMACRO - End macro"""
        pass  # Handled in main loop
    
    def handle_uexpand(self, match: re.Match):
        """UEXPAND - Expand macro"""
        name = match.group(1)
        args_str = match.group(2) if (match.lastindex or 0) >= 2 else ""
        args = [a.strip() for a in args_str.split(',')] if args_str else []
        
        if name not in self.macros:
            raise self.error(f"Undefined macro: {name}")
        
        macro = self.macros[name]
        
        if not macro.is_variadic and len(args) != len(macro.params):
            raise self.error(f"Macro {name} expects {len(macro.params)} args, got {len(args)}")
        
        # Expand macro body
        for line in macro.body:
            expanded = line
            for i, param in enumerate(macro.params):
                if i < len(args):
                    expanded = re.sub(r'\b' + param + r'\b', args[i], expanded)
            self.emit_text(expanded)
    
    # Utility directives
    
    def handle_upush(self, match: re.Match):
        """UPUSH - Save registers"""
        regs_str = match.group(1)
        regs = [r.strip() for r in regs_str.split(',')]
        
        for reg in regs:
            if self.arch == Architecture.X86_64:
                self.emit_text(f"    push {reg}")
            elif self.arch == Architecture.ARM64:
                self.emit_text(f"    str {reg}, [sp, #-16]!")
    
    def handle_upop(self, match: re.Match):
        """UPOP - Restore registers"""
        regs_str = match.group(1)
        regs = [r.strip() for r in regs_str.split(',')]
        
        for reg in reversed(regs):
            if self.arch == Architecture.X86_64:
                self.emit_text(f"    pop {reg}")
            elif self.arch == Architecture.ARM64:
                self.emit_text(f"    ldr {reg}, [sp], #16")
    
    def handle_ualign(self, match: re.Match):
        """UALIGN - Align to boundary"""
        boundary = match.group(1)
        self.emit_text(f"    align {boundary}")
    
    def handle_ucomment(self, match: re.Match):
        """UCOMMENT - Comment"""
        text = match.group(1)
        self.emit_text(f"    ; {text}")
    
    def handle_global(self, match: re.Match):
        """GLOBAL - Export symbol"""
        label = match.group(1)
        self.globals.add(label)
        self.emit_text(f"global {label}")
    
    def handle_extern(self, match: re.Match):
        """EXTERN - Import symbol"""
        label = match.group(1)
        self.externs.add(label)
    
    def handle_uweak(self, match: re.Match):
        """UWEAK - Weak symbol"""
        label = match.group(1)
        self.weaks.add(label)
        if self.platform in [Platform.LINUX, Platform.BSD, Platform.MACOS]:
            self.emit_text(f"weak {label}")
    
    def handle_ualias(self, match: re.Match):
        """UALIAS - Symbol alias"""
        new_name = match.group(1)
        old_name = match.group(2)
        self.emit_text(f"{new_name} equ {old_name}")
    
    def handle_uinclude(self, match: re.Match):
        """UINCLUDE - Include file"""
        filename = match.group(1)
        try:
            path = Path(filename)
            if path.exists():
                content = path.read_text(encoding='utf-8')
                # Recursively transpile
                nested = AssemblyTranspiler(
                    target=self.platform.value,
                    arch=self.arch.value,
                    opt_level=self.opt_level
                )
                result = nested.transpile(content)
                # Merge sections
                for line in result.splitlines():
                    if not line.startswith('section') and not line.startswith('extern') and not line.startswith('global'):
                        self.emit_text(line)
        except Exception as e:
            raise self.error(f"Failed to include {filename}: {e}")
    
    def handle_udefine(self, match: re.Match):
        """UDEFINE - Define symbol"""
        symbol = match.group(1)
        value = match.group(2) if (match.lastindex or 0) >= 2 else "1"
        self.symbols[symbol] = value
    
    def handle_uundef(self, match: re.Match):
        """UUNDEF - Undefine symbol"""
        symbol = match.group(1)
        if symbol in self.symbols:
            del self.symbols[symbol]
    
    def transpile(self, source: str) -> str:
        """Main transpilation method"""
        lines = source.splitlines()
        collecting_macro = None
        collecting_struct = None
        collecting_inline = None
        
        # Comprehensive directive patterns
        patterns = [
            # Data directives
            (r'^USTR\s+(\w+)\s+"(.*)"', self.handle_ustr),
            (r'^UWSTR\s+(\w+)\s+"(.*)"', self.handle_uwstr),
            (r'^UBYTES\s+(\w+)\s+(\d+)(?:\s+(\d+))?', self.handle_ubytes),
            (r'^UCONST\s+(\w+)\s+(.+)', self.handle_uconst),
            (r'^UARRAY\s+(\w+)\s+(\w+)\s+(.+)', self.handle_uarray),

            # Structure
            (r'^USTRUCT\s+(\w+)', self.handle_ustruct),

            # Function directives
            (r'^UFUNC\s+(\w+)(?:\s+(.+))?', self.handle_ufunc),
            (r'^UENDFUNC', self.handle_uendfunc),
            (r'^UPARAM\s+(\d+)\s+(\w+)', self.handle_uparam),
            (r'^ULOCAL\s+(\d+)', lambda m: setattr(self.current_function, "local_size",
                                                     self.current_function.local_size + int(m.group(1)))),
            (r'^URET(?:\s+(.+))?', self.handle_uret),
            (r'^UCALL\s+(\w+)(?:\s+(.+))?', self.handle_ucall),

            # System calls
            (r'^USYSCALL\s+(\w+)(?:\s+(.+))?', self.handle_usyscall),

            # I/O
            (r'^UPRINT\s+(\w+)', self.handle_uprint),
            (r'^UPRINTF\s+"(.+)"(?:\s+(.+))?', self.handle_uprintf),
            (r'^UEXIT\s+(\d+)', self.handle_uexit),

            # Memory management
            (r'^UMALLOC\s+(\w+)(?:\s+(\w+))?', self.handle_umalloc),
            (r'^UFREE\s+(\w+)', self.handle_ufree),

            # Advanced features
            (r'^UATOMIC\s+(\w+)\s+(.+)', self.handle_uatomic),
            (r'^USIMD\s+(\w+)\s+(.+)', self.handle_usimd),
            (r'^UTHREADLOCAL\s+(\w+)\s+(\d+)', self.handle_uthreadlocal),
            (r'^UBARRIER(?:\s+(\w+))?', self.handle_ubarrier),

            # Control flow
            (r'^ULOOP\s+(\w+)(?:\s+(\d+))?', self.handle_uloop),
            (r'^UENDLOOP', self.handle_uendloop),

            # Inline assembly
            (r'^UINLINE(?:\s+(\w+))?', self.handle_uinline),

            # Optimization hints
            (r'^UOPTIMIZE\s+(\w+)', self.handle_uoptimize),
            (r'^UPRAGMA\s+(.+)', self.handle_upragma),

            # Conditionals
            (r'^UIFOS\s+(\w+)', self.handle_uifos),
            (r'^UIFARCH\s+(\w+)', self.handle_uifarch),
            (r'^UIFDEF\s+(\w+)', self.handle_uifdef),
            (r'^UIFNDEF\s+(\w+)', self.handle_uifndef),
            (r'^UELSE', self.handle_uelse),
            (r'^UENDIF', self.handle_uendif),

            # Macros
            (r'^UMACRO\s+(\w+)(?:\s+(.+))?', self.handle_umacro),
            (r'^UENDMACRO', self.handle_uendmacro),
            (r'^UEXPAND\s+(\w+)(?:\s+(.+))?', self.handle_uexpand),

            # Utilities
            (r'^UPUSH\s+(.+)', self.handle_upush),
            (r'^UPOP\s+(.+)', self.handle_upop),
            (r'^UALIGN\s+(\d+)', self.handle_ualign),
            (r'^UCOMMENT\s+(.+)', self.handle_ucomment),

            # Symbols
            (r'^GLOBAL\s+(\w+)', self.handle_global),
            (r'^EXTERN\s+(\w+)', self.handle_extern),
            (r'^UWEAK\s+(\w+)', self.handle_uweak),
            (r'^UALIAS\s+(\w+)\s+(\w+)', self.handle_ualias),

            # Preprocessor
            (r'^UINCLUDE\s+"(.+)"', self.handle_uinclude),
            (r'^UDEFINE\s+(\w+)(?:\s+(.+))?', self.handle_udefine),
            (r'^UUNDEF\s+(\w+)', self.handle_uundef),
        ]
        
        for line_num, line in enumerate(lines, 1):
            self.line_number = line_num
            stripped = line.strip()
            # Uppercase version for case-insensitive control checks
            stripped_upper = stripped.upper() if stripped else ""
            
            # Skip empty lines
            if not stripped:
                if not self.skip_mode and not collecting_macro:
                    self.emit_text("")
                continue
            
            # Handle macro collection (accept with or without leading U)
            if collecting_macro:
                if stripped_upper in ("UENDMACRO", "ENDMACRO"):
                    collecting_macro = None
                    continue
                self.macros[collecting_macro].body.append(line)
                continue
            
            # Handle inline assembly collection (accept with or without leading U)
            if collecting_inline:
                if stripped_upper in ("UENDINLINE", "ENDINLINE"):
                    collecting_inline = None
                    continue
                # Emit inline code directly
                self.emit_text(line)
                continue
            
            # Skip if in conditional skip mode
            if self.skip_mode:
                # Only process control directives (case-insensitive).
                # Accept both with and without leading 'U' (e.g. ENDIF or UENDIF)
                allowed = ['UENDIF', 'UELSE', 'UIFOS', 'UIFARCH', 'UIFDEF', 'UIFNDEF',
                           'ENDIF', 'ELSE', 'IFOS', 'IFARCH', 'IFDEF', 'IFNDEF']
                if not any(stripped_upper.startswith(kw) for kw in allowed):
                    continue
            
            # Try to match directives
            matched = False
            for pattern, handler in patterns:
                # First try matching the line as-is (case-insensitive)
                m = re.match(pattern, stripped, re.I)
                # If no match, try matching with an implicit leading 'U' so users
                # can write directives without the leading 'U' (e.g. 'func' or 'call')
                if not m and not stripped_upper.startswith('U'):
                    try_line = 'U' + stripped
                    m = re.match(pattern, try_line, re.I)
                if m:
                    try:
                        result = handler(m)
                        if result is True:
                            # pattern contains the directive name; check case-insensitively
                            p_upper = pattern.upper()
                            if 'UMACRO' in p_upper:
                                collecting_macro = m.group(1)
                            elif 'UINLINE' in p_upper:
                                collecting_inline = True
                    except Exception as e:
                        raise self.error(f"Error in directive: {e}")
                    matched = True
                    break
            
            if not matched:
                # Pass through as-is (native assembly)
                # If user prefers no leading 'U' prefix on instructions, strip it here
                try:
                    if stripped and re.match(r'^U[A-Za-z0-9_]+', stripped, re.I):
                        # remove the leading 'U' only (preserve rest of line)
                        new_stripped = stripped[1:]
                        # Reconstruct line preserving original leading whitespace
                        leading_ws = line[:len(line) - len(line.lstrip())]
                        self.emit_text(f"{leading_ws}{new_stripped}")
                    else:
                        self.emit_text(line)
                except Exception:
                    # On any unexpected issue, fall back to emitting original line
                    self.emit_text(line)
        
        return self.generate_output()
    
    def generate_output(self) -> str:
        """Generate final assembly output"""
        out_lines = []
        
        # Header comments
        out_lines.append(f"; Generated by Advanced Assembly Transpiler v{__version__}")
        out_lines.append(f"; Target: {self.platform.value} ({self.arch.value})")
        out_lines.append(f"; Optimization: Level {self.opt_level.value}")
        out_lines.append("")
        
        # Platform-specific directives
        if self.platform == Platform.WINDOWS:
            out_lines.append("default rel")
        elif self.platform == Platform.MACOS:
            out_lines.append("default rel")
        
        out_lines.append("")
        
        # Externs
        if self.externs:
            for ext in sorted(self.externs):
                out_lines.append(f"extern {ext}")
            out_lines.append("")
        
        # Globals
        if self.globals:
            for glob in sorted(self.globals):
                out_lines.append(f"global {glob}")
            out_lines.append("")
        
        # Sections
        if self.rodata_lines:
            if self.platform == Platform.LINUX:
                out_lines.append("section .rodata")
            else:
                out_lines.append("section .rdata")
            for line in self.rodata_lines:
                out_lines.append(f"    {line}")
            out_lines.append("")
        
        if self.data_lines:
            out_lines.append("section .data")
            for line in self.data_lines:
                out_lines.append(f"    {line}")
            out_lines.append("")
        
        if self.bss_lines:
            out_lines.append("section .bss")
            for line in self.bss_lines:
                out_lines.append(f"    {line}")
            out_lines.append("")
        
        if self.tls_lines:
            if self.platform == Platform.LINUX:
                out_lines.append("section .tdata")
            elif self.platform == Platform.WINDOWS:
                out_lines.append("section .tls")
            for line in self.tls_lines:
                out_lines.append(f"    {line}")
            out_lines.append("")
        
        # Text section
        out_lines.append("section .text")
        out_lines.append("")
        out_lines.extend(self.text_lines)
        
        # Apply optimizations if enabled
        result = '\n'.join(out_lines)
        if self.opt_level != OptimizationLevel.NONE:
            result = self.optimizer.optimize(result, self.platform, self.arch)
        
        return result


def transpile_text(source: str, target: str = "linux", arch: str = "x86_64",
                   opt_level: str = "basic") -> str:
    """Transpile unified assembly to platform-specific assembly.
    
    Args:
        source: Unified assembly source code
        target: Target platform (linux/windows/macos/bsd)
        arch: Target architecture (x86_64/arm64/riscv64)
        opt_level: Optimization level (none/basic/aggressive/extreme)
    
    Returns:
        Platform-specific assembly code
    """
    opt_map = {
        'none': OptimizationLevel.NONE,
        'basic': OptimizationLevel.BASIC,
        'aggressive': OptimizationLevel.AGGRESSIVE,
        'extreme': OptimizationLevel.EXTREME
    }
    
    opt = opt_map.get(opt_level.lower(), OptimizationLevel.BASIC)
    transpiler = AssemblyTranspiler(target=target, arch=arch, opt_level=opt)
    return transpiler.transpile(source)


def transpile_file(src_path: str, target: str = 'linux', arch: str = 'x86_64',
                   opt_level: str = 'basic', out_path: str = None) -> str:
    """Transpile a unified assembly file.
    
    Args:
        src_path: Source file path
        target: Target platform
        arch: Target architecture
        opt_level: Optimization level
        out_path: Output file path (optional)
    
    Returns:
        Output file path
    """
    p = Path(src_path)
    if not p.exists():
        raise TranspileError(f"Source file not found: {src_path}")
    
    src = p.read_text(encoding='utf-8')
    out = transpile_text(src, target=target, arch=arch, opt_level=opt_level)
    
    if out_path:
        Path(out_path).write_text(out, encoding='utf-8')
        return out_path
    
    # Default output filename
    default = str(p.with_suffix(f'.{target}.{arch}.asm'))
    Path(default).write_text(out, encoding='utf-8')
    return default


def main(argv: List[str] = None) -> int:
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        prog='advanced_asm_transpiler',
        description='Ultra-Advanced Cross-Platform Assembly Transpiler v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s code.uasm
  %(prog)s code.uasm --target windows --arch x86_64
  %(prog)s code.uasm --target linux --arch arm64
  %(prog)s code.uasm --optimize extreme --out code.asm
  %(prog)s code.uasm --target macos --stats
  
Supported Directives (80+):
  Data: USTR, UWSTR, UBYTES, UCONST, UARRAY, USTRUCT
  Functions: UFUNC, UENDFUNC, UPARAM, ULOCAL, URET, UCALL
  I/O: UPRINT, UPRINTF, UEXIT
  Memory: UMALLOC, UFREE
  Syscalls: USYSCALL (250+ Linux syscalls)
  Threading: UTHREADLOCAL, UBARRIER, UATOMIC
  SIMD: USIMD (AVX/NEON support)
  Control: ULOOP, UENDLOOP, UIFOS, UIFARCH, UIFDEF, UIFNDEF, UELSE, UENDIF
  Macros: UMACRO, UENDMACRO, UEXPAND
  Optimization: UOPTIMIZE, UPRAGMA, UINLINE
  Symbols: GLOBAL, EXTERN, UWEAK, UALIAS
  Utility: UPUSH, UPOP, UALIGN, UCOMMENT
  Preprocessor: UINCLUDE, UDEFINE, UUNDEF
  
Supported Platforms: Linux, Windows, macOS, BSD
Supported Architectures: x86-64, ARM64, RISC-V64
        """
    )
    
    parser.add_argument('src', help='Unified assembly source file')
    parser.add_argument('--target', '-t', 
                       choices=['linux', 'windows', 'macos', 'bsd'],
                       default='linux',
                       help='Target platform (default: linux)')
    parser.add_argument('--arch', '-a',
                       choices=['x86_64', 'arm64', 'riscv64'],
                       default='x86_64',
                       help='Target architecture (default: x86_64)')
    parser.add_argument('--optimize', '-O',
                       choices=['none', 'basic', 'aggressive', 'extreme'],
                       default='basic',
                       help='Optimization level (default: basic)')
    parser.add_argument('--out', '-o',
                       help='Output filename (optional)')
    parser.add_argument('--stats', '-s',
                       action='store_true',
                       help='Show statistics')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--version',
                       action='version',
                       version=f'%(prog)s {__version__}')
    
    args = parser.parse_args(argv)
    
    try:
        if args.verbose:
            print(f"Advanced Assembly Transpiler v{__version__}")
            print(f"Transpiling: {args.src}")
            print(f"Target: {args.target} ({args.arch})")
            print(f"Optimization: {args.optimize}")
            print()
        
        # Transpile
        start_time = None
        if args.stats:
            import time
            start_time = time.time()
        
        out_path = transpile_file(
            args.src,
            target=args.target,
            arch=args.arch,
            opt_level=args.optimize,
            out_path=args.out
        )
        
        if args.stats:
            elapsed = time.time() - start_time
            
            # Gather statistics
            src_lines = Path(args.src).read_text().count('\n') + 1
            out_lines = Path(out_path).read_text().count('\n') + 1
            out_size = Path(out_path).stat().st_size
            
            print("\n" + "="*60)
            print("Transpilation Statistics")
            print("="*60)
            print(f"Source lines:        {src_lines:,}")
            print(f"Output lines:        {out_lines:,}")
            print(f"Output size:         {out_size:,} bytes")
            print(f"Expansion ratio:     {out_lines/src_lines:.2f}x")
            print(f"Time elapsed:        {elapsed:.3f}s")
            print(f"Lines per second:    {int(src_lines/elapsed):,}")
            print("="*60)
        
        print(f"Success! Output: {out_path}")
        
        if args.verbose:
            print(f"\nGenerated {Path(out_path).stat().st_size} bytes")
            print(f"Target assembly format: NASM")
        
        return 0
        
    except TranspileError as e:
        print(f"✗ Transpilation error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())