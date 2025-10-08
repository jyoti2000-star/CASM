#!/usr/bin/env python3
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

class BuildTools:
    """Build tool detection and management"""
    
    @staticmethod
    def check_nasm() -> bool:
        """Check if NASM is available"""
        return shutil.which('nasm') is not None
    
    @staticmethod
    def check_mingw() -> bool:
        """Check if MinGW-w64 is available"""
        return shutil.which('x86_64-w64-mingw32-gcc') is not None
    
    @staticmethod
    def check_wine() -> bool:
        """Check if Wine is available"""
        return shutil.which('wine') is not None or shutil.which('wine64') is not None
    
    @staticmethod
    def get_missing_tools() -> List[str]:
        """Get list of missing build tools"""
        missing = []
        
        if not BuildTools.check_nasm():
            missing.append('nasm')
        
        if not BuildTools.check_mingw():
            missing.append('mingw-w64')
        
        return missing
    
    @staticmethod
    def print_install_instructions():
        """Print installation instructions for missing tools"""
        missing = BuildTools.get_missing_tools()
        
        if not missing:
            print("[INFO] All build tools are available")
            return
        
        print(f"[ERROR] Missing build tools: {', '.join(missing)}")
        print("\nInstallation instructions:")
        
        if 'nasm' in missing:
            print("  NASM Assembler:")
            print("    macOS: brew install nasm")
            print("    Ubuntu/Debian: sudo apt install nasm")
            print("    Windows: Download from https://www.nasm.us/")
        
        if 'mingw-w64' in missing:
            print("  MinGW-w64 Toolchain:")
            print("    macOS: brew install mingw-w64")
            print("    Ubuntu/Debian: sudo apt install gcc-mingw-w64")
            print("    Windows: Use MSYS2 or TDM-GCC")
        
        if not BuildTools.check_wine():
            print("  Wine (optional, for running Windows executables):")
            print("    macOS: brew install --cask wine-stable")
            print("    Ubuntu/Debian: sudo apt install wine")

class AssemblyValidator:
    """Validate generated assembly code"""
    
    @staticmethod
    def validate_syntax(assembly_code: str) -> Tuple[bool, List[str]]:
        """Basic assembly syntax validation"""
        lines = assembly_code.split('\n')
        issues = []
        
        has_main = False
        has_text_section = False
        has_extern_printf = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for required elements
            if 'main:' in stripped:
                has_main = True
            elif 'section .text' in stripped:
                has_text_section = True
            elif 'extern printf' in stripped:
                has_extern_printf = True
            
            # Check for common issues
            if stripped.endswith(':') and not stripped.startswith(';'):
                # Label should not have indentation
                if line.startswith('    '):
                    issues.append(f"Line {i}: Label should not be indented: {stripped}")
            
            # Check for proper instruction indentation
            if stripped and not stripped.startswith(';') and not stripped.endswith(':') and not stripped.startswith('section') and not stripped.startswith('extern') and not stripped.startswith('global'):
                if not line.startswith('    '):
                    issues.append(f"Line {i}: Instruction should be indented: {stripped}")
        
        # Check required elements
        if not has_main:
            issues.append("Missing main label")
        if not has_text_section:
            issues.append("Missing .text section")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def print_validation_report(assembly_code: str):
        """Print validation report"""
        is_valid, issues = AssemblyValidator.validate_syntax(assembly_code)
        
        if is_valid:
            print("[INFO] Assembly validation passed")
        else:
            print(f"[WARNING] Assembly validation found {len(issues)} issue(s):")
            for issue in issues:
                print(f"  {issue}")

# Global instances
build_tools = BuildTools()
validator = AssemblyValidator()