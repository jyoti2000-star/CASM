#!/usr/bin/env python3

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import platform
import json
from pathlib import Path

class Architecture(Enum):
    """Supported CPU architectures"""
    X86_64 = "x86_64"
    X86_32 = "x86_32"

class OperatingSystem(Enum):
    """Supported operating systems"""
    WINDOWS = "windows"

class CallingConvention(Enum):
    """Calling conventions"""
    MS_X64 = "ms_x64"        # Windows x64
    CDECL = "cdecl"          # x86 32-bit

@dataclass
class PlatformConfig:
    """Platform-specific configuration"""
    os: OperatingSystem
    arch: Architecture
    calling_convention: CallingConvention
    pointer_size: int = 8
    word_size: int = 4
    endianness: str = "little"
    
    # Assembly format specifics
    object_format: str = "win64"
    entry_point: str = "_start"
    global_directive: str = "global _start"
    section_prefix: str = "section"
    comment_char: str = ";"
    
    # Register names
    registers: Dict[str, str] = field(default_factory=dict)
    
    # System call interface
    syscall_numbers: Dict[str, Union[str, int]] = field(default_factory=dict)
    syscall_instruction: str = "syscall"
    
    # ABI-specific details
    stack_alignment: int = 16
    red_zone_size: int = 0
    supports_rip_relative: bool = True
    supports_32bit_absolute: bool = True
    
    # Optimization flags
    optimization_flags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'os': self.os.value,
            'arch': self.arch.value,
            'calling_convention': self.calling_convention.value,
            'pointer_size': self.pointer_size,
            'word_size': self.word_size,
            'endianness': self.endianness,
            'object_format': self.object_format,
            'entry_point': self.entry_point,
            'global_directive': self.global_directive,
            'section_prefix': self.section_prefix,
            'comment_char': self.comment_char,
            'registers': self.registers,
            'syscall_numbers': self.syscall_numbers,
            'syscall_instruction': self.syscall_instruction,
            'stack_alignment': self.stack_alignment,
            'red_zone_size': self.red_zone_size,
            'supports_rip_relative': self.supports_rip_relative,
            'supports_32bit_absolute': self.supports_32bit_absolute,
            'optimization_flags': self.optimization_flags
        }

class AssemblyConverter:
    """Converts normal assembly code into target architecture-specific code"""
    
    def __init__(self):
        self.platforms: Dict[str, PlatformConfig] = {}
        self.current_platform: Optional[PlatformConfig] = None
        self.host_platform: Optional[PlatformConfig] = None
        
        # Initialize platform configurations
        self._initialize_platforms()
        
        # Detect host platform
        self.host_platform = self._detect_host_platform()
        self.current_platform = self.host_platform
    
    def _initialize_platforms(self):
        """Initialize all supported platform configurations"""
        

        
        # Windows x86_64
        self.platforms["windows-x86_64"] = PlatformConfig(
            os=OperatingSystem.WINDOWS,
            arch=Architecture.X86_64,
            calling_convention=CallingConvention.MS_X64,
            object_format="win64",
            entry_point="main",
            global_directive="global main",
            registers={
                'syscall_num': 'rax',
                'arg1': 'rcx', 'arg2': 'rdx', 'arg3': 'r8', 'arg4': 'r9',
                'arg5': '[rsp+32]', 'arg6': '[rsp+40]',  # Stack parameters
                'return': 'rax',
                'stack_pointer': 'rsp',
                'base_pointer': 'rbp'
            },
            syscall_numbers={
                # Windows uses a different syscall mechanism
                # These are NT system service numbers (simplified)
                'NtReadFile': 0x006,
                'NtWriteFile': 0x008,
                'NtOpenFile': 0x033,
                'NtClose': 0x00f,
                'NtCreateFile': 0x055,
                'NtTerminateProcess': 0x02c,
                'NtAllocateVirtualMemory': 0x018,
                'NtFreeVirtualMemory': 0x01e,
                'NtQueryInformationProcess': 0x019,
                'NtWaitForSingleObject': 0x004
            },
            syscall_instruction="syscall",  # Or int 0x2e for older systems
            red_zone_size=0,  # Windows x64 doesn't have a red zone
            supports_32bit_absolute=False,
            optimization_flags={
                'shadow_space': 32,  # Windows x64 calling convention
                'stack_reserve': True
            }
        )
        

    
    def _detect_host_platform(self) -> PlatformConfig:
        """Detect the host platform automatically"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Map platform.machine() output to our architectures
        arch_map = {
            'x86_64': Architecture.X86_64,
            'amd64': Architecture.X86_64,
            'i386': Architecture.X86_32,
            'i686': Architecture.X86_32,
        }
        
        os_map = {
            'windows': OperatingSystem.WINDOWS,
        }
        
        detected_os = os_map.get(system, OperatingSystem.WINDOWS)
        detected_arch = arch_map.get(machine, Architecture.X86_64)
        
        platform_key = f"{detected_os.value}-{detected_arch.value}"
        
        if platform_key in self.platforms:
            return self.platforms[platform_key]
        else:
            # Return a default configuration if exact match not found
            return self.platforms.get("windows-x86_64", list(self.platforms.values())[0])
    
    def set_target_platform(self, target: str) -> bool:
        """Set the target platform for cross-compilation"""
        if target in self.platforms:
            self.current_platform = self.platforms[target]
            return True
        return False
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available target platforms"""
        return list(self.platforms.keys())
    
    def get_platform_info(self, platform_name: str = None) -> Optional[PlatformConfig]:
        """Get detailed platform information"""
        if platform_name is None:
            return self.current_platform
        return self.platforms.get(platform_name)
    
    def is_cross_compiling(self) -> bool:
        """Check if we're cross-compiling"""
        if not self.host_platform or not self.current_platform:
            return False
        
        return (self.host_platform.os != self.current_platform.os or 
                self.host_platform.arch != self.current_platform.arch)
    
    def get_cross_compilation_tools(self) -> Dict[str, str]:
        """Get cross-compilation toolchain information"""
        if not self.is_cross_compiling():
            return {}
        
        target = self.current_platform
        tools = {}
        
        # GNU toolchain naming convention
        if target.arch == Architecture.X86_32:
            prefix = "i686-elf"
        else:
            prefix = "x86_64-elf"
        
        tools = {
            'assembler': f"{prefix}-as",
            'linker': f"{prefix}-ld", 
            'objcopy': f"{prefix}-objcopy",
            'objdump': f"{prefix}-objdump",
            'strip': f"{prefix}-strip"
        }
        
        return tools
    
    def validate_platform_support(self, features: List[str]) -> Dict[str, bool]:
        """Validate if current platform supports requested features"""
        if not self.current_platform:
            return {feature: False for feature in features}
        
        support = {}
        
        for feature in features:
            if feature == "rip_relative":
                support[feature] = self.current_platform.supports_rip_relative
            elif feature == "32bit_absolute":
                support[feature] = self.current_platform.supports_32bit_absolute
            elif feature == "red_zone":
                support[feature] = self.current_platform.red_zone_size > 0
            elif feature == "syscall":
                support[feature] = bool(self.current_platform.syscall_numbers)
            elif feature == "large_memory":
                support[feature] = self.current_platform.pointer_size >= 8
            else:
                support[feature] = True  # Assume supported for unknown features
        
        return support
    
    def generate_platform_report(self) -> str:
        """Generate comprehensive platform compatibility report"""
        report = []
        report.append("=" * 80)
        report.append("CROSS-PLATFORM COMPATIBILITY REPORT")
        report.append("=" * 80)
        
        if self.host_platform:
            report.append(f"Host Platform: {self.host_platform.os.value}-{self.host_platform.arch.value}")
        
        if self.current_platform:
            report.append(f"Target Platform: {self.current_platform.os.value}-{self.current_platform.arch.value}")
            report.append(f"Cross-compiling: {'Yes' if self.is_cross_compiling() else 'No'}")
            report.append("")
            
            # Current platform details
            report.append("Target Platform Configuration:")
            report.append("-" * 40)
            report.append(f"  Architecture: {self.current_platform.arch.value}")
            report.append(f"  Operating System: {self.current_platform.os.value}")
            report.append(f"  Calling Convention: {self.current_platform.calling_convention.value}")
            report.append(f"  Object Format: {self.current_platform.object_format}")
            report.append(f"  Entry Point: {self.current_platform.entry_point}")
            report.append(f"  Pointer Size: {self.current_platform.pointer_size} bytes")
            report.append(f"  Stack Alignment: {self.current_platform.stack_alignment} bytes")
            report.append(f"  Red Zone Size: {self.current_platform.red_zone_size} bytes")
            report.append(f"  RIP-relative Support: {self.current_platform.supports_rip_relative}")
            report.append(f"  32-bit Absolute Support: {self.current_platform.supports_32bit_absolute}")
            report.append("")
            
            # Register mapping
            report.append("Register Mapping:")
            report.append("-" * 40)
            for purpose, reg in self.current_platform.registers.items():
                report.append(f"  {purpose}: {reg}")
            report.append("")
            
            # System calls (show first 10)
            if self.current_platform.syscall_numbers:
                report.append("System Calls (first 10):")
                report.append("-" * 40)
                for i, (name, num) in enumerate(self.current_platform.syscall_numbers.items()):
                    if i >= 10:
                        report.append(f"  ... and {len(self.current_platform.syscall_numbers) - 10} more")
                        break
                    report.append(f"  {name}: {num}")
                report.append("")
        
        # Cross-compilation tools
        if self.is_cross_compiling():
            tools = self.get_cross_compilation_tools()
            if tools:
                report.append("Cross-compilation Tools:")
                report.append("-" * 40)
                for tool, command in tools.items():
                    report.append(f"  {tool}: {command}")
                report.append("")
        
        # Available platforms
        report.append("Available Target Platforms:")
        report.append("-" * 40)
        for platform_name in sorted(self.get_available_platforms()):
            platform_config = self.platforms[platform_name]
            report.append(f"  {platform_name} ({platform_config.os.value}, {platform_config.arch.value})")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def export_platform_config(self, filename: str):
        """Export current platform configuration to JSON"""
        if not self.current_platform:
            return
        
        config_data = {
            'platform': f"{self.current_platform.os.value}-{self.current_platform.arch.value}",
            'host_platform': f"{self.host_platform.os.value}-{self.host_platform.arch.value}" if self.host_platform else None,
            'is_cross_compiling': self.is_cross_compiling(),
            'configuration': self.current_platform.to_dict()
        }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def import_platform_config(self, filename: str) -> bool:
        """Import platform configuration from JSON"""
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            platform_name = config_data.get('platform')
            if platform_name in self.platforms:
                self.current_platform = self.platforms[platform_name]
                return True
            
            return False
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
    
    def convert_assembly_code(self, assembly_code: str, target_platform: str = None) -> str:
        """Convert normal assembly code to target architecture"""
        if target_platform:
            old_platform = self.current_platform
            if not self.set_target_platform(target_platform):
                raise ValueError(f"Unsupported target platform: {target_platform}")
        
        try:
            converted_code = self._apply_platform_conversions(assembly_code)
            return converted_code
        finally:
            if target_platform and 'old_platform' in locals():
                self.current_platform = old_platform
    
    def _apply_platform_conversions(self, assembly_code: str) -> str:
        """Apply platform-specific conversions to assembly code"""
        if not self.current_platform:
            return assembly_code
        
        lines = assembly_code.split('\n')
        converted_lines = []
        
        for line in lines:
            converted_line = self._convert_line(line.strip())
            converted_lines.append(converted_line)
        
        # Add platform-specific headers and footers
        result = self._add_platform_headers(converted_lines)
        return result
    
    def _convert_line(self, line: str) -> str:
        """Convert a single line of assembly code"""
        if not line or line.startswith(';') or line.startswith('#'):
            return line
        
        # Convert syscalls
        line = self._convert_syscalls(line)
        
        # Convert registers
        line = self._convert_registers(line)
        
        # Convert addressing modes
        line = self._convert_addressing(line)
        
        # Convert calling conventions
        line = self._convert_calling_convention(line)
        
        return line
    
    def _convert_syscalls(self, line: str) -> str:
        """Convert syscall instructions and numbers"""
        if not self.current_platform:
            return line
        
        # Convert syscall instruction
        if 'syscall' in line.lower():
            line = line.replace('syscall', self.current_platform.syscall_instruction)
        
        # Convert common syscall numbers for Windows
        syscall_map = {
            # Windows syscall numbers
            'mov rax, 1': f"mov {self.current_platform.registers.get('syscall_num', 'rax')}, {self.current_platform.syscall_numbers.get('NtWriteFile', 0x008)}",
            'mov rax, 60': f"mov {self.current_platform.registers.get('syscall_num', 'rax')}, {self.current_platform.syscall_numbers.get('NtTerminateProcess', 0x02c)}",
            'mov rax, 0': f"mov {self.current_platform.registers.get('syscall_num', 'rax')}, {self.current_platform.syscall_numbers.get('NtReadFile', 0x006)}",
        }
        
        for windows_syscall, target_syscall in syscall_map.items():
            if windows_syscall in line:
                line = line.replace(windows_syscall, target_syscall)
        
        return line
    
    def _convert_registers(self, line: str) -> str:
        """Convert register usage based on calling convention"""
        if not self.current_platform:
            return line
        
        # Convert argument registers based on calling convention
        if self.current_platform.calling_convention == CallingConvention.MS_X64:
            # Convert System V to MS x64 calling convention
            reg_map = {
                'rdi': 'rcx',  # First argument
                'rsi': 'rdx',  # Second argument
                'rdx': 'r8',   # Third argument
                'rcx': 'r9',   # Fourth argument
            }
            
            for sysv_reg, ms_reg in reg_map.items():
                # Replace register references while preserving instruction format
                import re
                pattern = r'\b' + sysv_reg + r'\b'
                line = re.sub(pattern, ms_reg, line)
        
        return line
    
    def _convert_addressing(self, line: str) -> str:
        """Convert addressing modes for target architecture"""
        if not self.current_platform:
            return line
        
        # Handle RIP-relative addressing
        if not self.current_platform.supports_rip_relative and '[rip+' in line:
            # Convert RIP-relative to absolute addressing
            import re
            line = re.sub(r'\[rip\+([^\]]+)\]', r'[\1]', line)
        
        # Handle 32-bit absolute addressing
        if not self.current_platform.supports_32bit_absolute:
            # Need more complex handling here based on specific cases
            pass
        
        return line
    
    def _convert_calling_convention(self, line: str) -> str:
        """Convert function calling conventions"""
        if not self.current_platform:
            return line
        
        # Add shadow space for Windows x64
        if (self.current_platform.calling_convention == CallingConvention.MS_X64 and 
            'call' in line.lower()):
            # This would need more sophisticated analysis to properly add shadow space
            pass
        
        return line
    
    def _add_platform_headers(self, lines: List[str]) -> str:
        """Add platform-specific assembly headers and directives"""
        if not self.current_platform:
            return '\n'.join(lines)
        
        header_lines = []
        
        # Add format directive
        if self.current_platform.object_format == "win64":
            header_lines.append("format PE64 console")
        else:
            header_lines.append("format PE64 console")
        
        # Add entry point
        header_lines.append(f"entry {self.current_platform.entry_point}")
        header_lines.append("")
        
        # Add global directive if needed
        if self.current_platform.global_directive:
            header_lines.append(self.current_platform.global_directive)
            header_lines.append("")
        
        # Add section directive
        header_lines.append(f"{self.current_platform.section_prefix} .text")
        header_lines.append(f"{self.current_platform.entry_point}:")
        header_lines.append("")
        
        # Combine header with converted code
        result_lines = header_lines + lines
        
        return '\n'.join(result_lines)
    
    def convert_file(self, input_file: str, output_file: str, target_platform: str = None) -> bool:
        """Convert an assembly file to target architecture"""
        try:
            with open(input_file, 'r') as f:
                assembly_code = f.read()
            
            converted_code = self.convert_assembly_code(assembly_code, target_platform)
            
            with open(output_file, 'w') as f:
                f.write(converted_code)
            
            return True
        except Exception as e:
            print(f"Error converting file: {e}")
            return False
    
    def get_conversion_report(self, assembly_code: str, target_platform: str = None) -> str:
        """Generate a report of what conversions would be applied"""
        if target_platform:
            old_platform = self.current_platform
            if not self.set_target_platform(target_platform):
                return f"Error: Unsupported target platform: {target_platform}"
        
        try:
            report = []
            report.append("Assembly Conversion Report")
            report.append("=" * 30)
            report.append(f"Target Platform: {self.current_platform.os.value}-{self.current_platform.arch.value}")
            report.append(f"Architecture: {self.current_platform.arch.value}")
            report.append(f"Calling Convention: {self.current_platform.calling_convention.value}")
            report.append(f"Object Format: {self.current_platform.object_format}")
            report.append("")
            
            lines = assembly_code.split('\n')
            changes_made = 0
            
            for i, line in enumerate(lines, 1):
                original = line.strip()
                if not original or original.startswith(';'):
                    continue
                
                converted = self._convert_line(original)
                if converted != original:
                    report.append(f"Line {i}: {original} -> {converted}")
                    changes_made += 1
            
            if changes_made == 0:
                report.append("No conversions needed for target platform.")
            else:
                report.append(f"\nTotal conversions: {changes_made}")
            
            return '\n'.join(report)
        finally:
            if target_platform and 'old_platform' in locals():
                self.current_platform = old_platform

# Global assembly converter instance
global_assembly_converter = AssemblyConverter()