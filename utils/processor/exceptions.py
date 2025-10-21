from typing import Optional

class CompilerError(Exception):
    """Compilation error"""
    def __init__(self, message: str, stderr: str = "", compiler: Optional[str] = None):
        self.message = message
        self.stderr = stderr
        self.compiler = compiler
        super().__init__(message)

class VariableError(Exception):
    """Variable-related error"""
    pass

class HeaderError(Exception):
    """Header-related error"""
    pass
