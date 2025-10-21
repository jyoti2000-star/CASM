"""
Compatibility wrapper: delegate core CASMLexer functionality to the
implementation in src.compiler.lexer when available. The core pipeline
expects CASMLexer(source, filename) with a .tokenize() method that returns
token list; src.compiler.lexer provides a similar but slightly different API.
"""
from typing import List, Tuple, Dict, Optional
from .tokens import TokenType, Token, SourceLocation
from .diagnostics import DiagnosticEngine

try:
    # Prefer the compiler lexer implementation
    from src.compiler.lexer import CASMLexer as _CompilerCASMLexer
except Exception:
    _CompilerCASMLexer = None


class CASMLexer:
    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.diagnostics = DiagnosticEngine()

        if _CompilerCASMLexer:
            # The compiler lexer expects either a filename or text; create an
            # instance and keep it for tokenize delegation.
            try:
                # The compiler lexer.tokenize(text) expects text input, so no
                # constructor side-effects are assumed. We'll not rely on
                # _CompilerCASMLexer(filename) signature; instead we'll call its
                # tokenize method directly.
                self._delegate = _CompilerCASMLexer()
            except TypeError:
                # If the constructor signature differs, don't crash — fallback
                self._delegate = None
        else:
            self._delegate = None

    def tokenize(self) -> List[Token]:
        """Return a list of tokens for the stored source text.

        The core pipeline calls CASMLexer(source, filename).tokenize() so
        provide that behavior by delegating to the compiler lexer if
        available, otherwise perform a minimal split into EOF token.
        """
        if self._delegate is not None:
            # compiler.CASMLexer.tokenize expects text input
            try:
                return self._delegate.tokenize(self.source)
            except TypeError:
                # Some versions may expect no args and use internal filename; try both
                try:
                    return self._delegate.tokenize()
                except Exception as e:
                    raise

        # Minimal fallback tokenization to avoid hard failure. This will
        # produce an EOF token only and likely fail later in parsing, but it
        # prevents import-time crashes.
        from .tokens import Token, TokenType, SourceLocation
        eof = Token(type=TokenType.EOF, value='', location=SourceLocation(line=1, column=1, file=self.filename, raw_line=''))
        return [eof]
