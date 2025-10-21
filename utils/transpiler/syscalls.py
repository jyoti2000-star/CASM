from typing import Optional
from .enums import Platform, Architecture

class Syscalls:
    LINUX_X86_64 = {
        'read': 0, 'write': 1, 'open': 2, 'close': 3,
        'stat': 4, 'fstat': 5, 'lstat': 6, 'poll': 7,
        'lseek': 8, 'mmap': 9, 'mprotect': 10, 'munmap': 11,
        'brk': 12, 'rt_sigaction': 13, 'rt_sigprocmask': 14,
        'ioctl': 16, 'pread64': 17, 'pwrite64': 18,
        'readv': 19, 'writev': 20, 'access': 21, 'pipe': 22,
        'select': 23, 'sched_yield': 24, 'mremap': 25,
        'msync': 26, 'mincore': 27, 'madvise': 28,
        'dup': 32, 'dup2': 33, 'pause': 34, 'nanosleep': 35,
        'getpid': 39, 'socket': 41, 'connect': 42,
        'accept': 43, 'sendto': 44, 'recvfrom': 45,
        'sendmsg': 46, 'recvmsg': 47, 'shutdown': 48,
        'bind': 49, 'listen': 50, 'getsockname': 51,
        'getpeername': 52, 'socketpair': 53, 'setsockopt': 54,
        'getsockopt': 55, 'clone': 56, 'fork': 57,
        'vfork': 58, 'execve': 59, 'exit': 60,
        'wait4': 61, 'kill': 62, 'uname': 63,
    }
    
    MACOS_X86_64 = {
        'exit': 0x2000001, 'fork': 0x2000002, 'read': 0x2000003,
        'write': 0x2000004, 'open': 0x2000005, 'close': 0x2000006,
        'wait4': 0x2000007, 'link': 0x2000009, 'unlink': 0x200000a,
        'chdir': 0x200000c, 'fchdir': 0x200000d, 'mknod': 0x200000e,
        'chmod': 0x200000f, 'chown': 0x2000010, 'getfsstat': 0x2000012,
        'getpid': 0x2000014, 'setuid': 0x2000017, 'getuid': 0x2000018,
        'geteuid': 0x2000019, 'ptrace': 0x200001a, 'recvmsg': 0x200001b,
        'sendmsg': 0x200001c, 'recvfrom': 0x200001d, 'accept': 0x200001e,
        'getpeername': 0x200001f, 'getsockname': 0x2000020,
    }
    
    WINDOWS_SYSCALLS = {
        'NtReadFile': 0x0006,
        'NtWriteFile': 0x0008,
        'NtClose': 0x000F,
        'NtOpenFile': 0x0033,
        'NtCreateFile': 0x0055,
    }
    
    @staticmethod
    def get_syscall_number(platform: Platform, arch: Architecture, name: str) -> Optional[int]:
        name = name.lower()
        
        if platform == Platform.LINUX and arch == Architecture.X86_64:
            return Syscalls.LINUX_X86_64.get(name)
        elif platform == Platform.MACOS:
            return Syscalls.MACOS_X86_64.get(name)
        elif platform == Platform.WINDOWS:
            return Syscalls.WINDOWS_SYSCALLS.get(name)
        
        return None
