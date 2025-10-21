from typing import Dict, Optional
from pathlib import Path
import shutil
import subprocess
from .enums import CompilerType, Platform, Architecture

class CompilerDetector:
    """Detect and validate available compilers"""
    
    @staticmethod
    def detect_compilers() -> Dict[CompilerType, Path]:
        """Detect available compilers on the system"""
        compilers = {}
        
        for compiler_type in CompilerType:
            path = CompilerDetector.find_compiler(compiler_type)
            if path:
                compilers[compiler_type] = path
        
        return compilers
    
    @staticmethod
    def find_compiler(compiler_type: CompilerType) -> Optional[Path]:
        """Find a specific compiler"""
        if compiler_type == CompilerType.ZIGCC:
            zig_path = shutil.which("zig")
            if zig_path:
                return Path(zig_path)
            return None
        
        compiler_name = compiler_type.value
        compiler_path = shutil.which(compiler_name)
        
        if compiler_path:
            return Path(compiler_path)
        
        return None
    
    @staticmethod
    def get_compiler_version(compiler_path: Path) -> Optional[str]:
        """Get compiler version"""
        try:
            result = subprocess.run(
                [str(compiler_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip().split('\n')[0]
        except Exception:
            return None
    
    @staticmethod
    def select_best_compiler(
        platform: Platform,
        arch: Architecture,
        available: Dict[CompilerType, Path]
    ) -> Optional[CompilerType]:
        """Select the best compiler for target platform/arch"""
        if platform == Platform.WINDOWS:
            preferences = [CompilerType.MSVC, CompilerType.CLANG, CompilerType.GCC]
        elif platform == Platform.MACOS:
            preferences = [CompilerType.CLANG, CompilerType.GCC]
        else:
            preferences = [CompilerType.GCC, CompilerType.CLANG]
        
        for compiler in preferences:
            if compiler in available:
                return compiler
        
        if available:
            return next(iter(available.keys()))
        
        return None
