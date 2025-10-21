"""C processor package - re-exports for convenience"""
from .enums import CompilerType, OptimizationLevel, WarningLevel, Platform, Architecture, VariableScope, CallingConvention
from .data import CVariable, CFunction, CHeader, CCodeBlock, CompilationResult, CompilerConfig
from .exceptions import CompilerError, VariableError, HeaderError
from .detector import CompilerDetector
from .analyzers import VariableAnalyzer, FunctionAnalyzer
from .processor import CCodeProcessor, c_processor, compile_c_inline, analyze_c_code

__all__ = [
    'CompilerType','OptimizationLevel','WarningLevel','Platform','Architecture','VariableScope','CallingConvention',
    'CVariable','CFunction','CHeader','CCodeBlock','CompilationResult','CompilerConfig',
    'CompilerError','VariableError','HeaderError','CompilerDetector',
    'VariableAnalyzer','FunctionAnalyzer','CCodeProcessor','c_processor','compile_c_inline','analyze_c_code'
]
