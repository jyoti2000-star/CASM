from typing import Optional
from .diagnostics import DiagnosticEngine
from .lexer import CASMLexer
from .parser import CASMParser
from .codegen import AssemblyCodeGenerator

class CompilerConfig:
    def __init__(self):
        self.optimization_level = 0
        self.target_arch = "x86_64"
        self.emit_debug_info = False
        self.verbose = False
        self.warnings_as_errors = False
        self.unused_variable_warning = True
        self.output_format = "nasm"

class CASMCompiler:
    def __init__(self, config: Optional[CompilerConfig] = None):
        self.config = config or CompilerConfig()
        self.diagnostics = DiagnosticEngine()
        
    def compile(self, source: str, filename: str = "<stdin>") -> Optional[str]:
        if self.config.verbose:
            print("Phase 1: Lexical Analysis")
        lexer = CASMLexer(source, filename)
        tokens = lexer.tokenize()
        if lexer.diagnostics.has_errors():
            lexer.diagnostics.print_all()
            return None
        if self.config.verbose:
            print(f"  Generated {len(tokens)} tokens")

        if self.config.verbose:
            print("Phase 2: Parsing")
        parser = CASMParser(tokens)
        ast = parser.parse()
        if parser.diagnostics.has_errors():
            parser.diagnostics.print_all()
            return None
        if not ast:
            print("Parse failed: no AST generated")
            return None
        if self.config.verbose:
            print(f"  Generated AST with {len(ast.statements)} statements")

        if self.config.verbose:
            print("Phase 5: Code Generation")
        codegen = AssemblyCodeGenerator()
        assembly = codegen.generate(ast)
        if codegen.diagnostics.has_errors():
            codegen.diagnostics.print_all()
            return None

        if parser.diagnostics.warning_count > 0:
            parser.diagnostics.print_all()
        if codegen.diagnostics.warning_count > 0:
            codegen.diagnostics.print_all()

        if self.config.verbose:
            print(f"Compilation successful!")
            print(f"  {parser.diagnostics.warning_count} warnings")

        return assembly

def create_default_compiler() -> CASMCompiler:
    config = CompilerConfig()
    return CASMCompiler(config)

def compile_string(source: str, verbose: bool = False) -> Optional[str]:
    config = CompilerConfig()
    config.verbose = verbose
    compiler = CASMCompiler(config)
    return compiler.compile(source)

def compile_file(filepath: str, verbose: bool = False) -> Optional[str]:
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        config = CompilerConfig()
        config.verbose = verbose
        compiler = CASMCompiler(config)
        return compiler.compile(source, filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
