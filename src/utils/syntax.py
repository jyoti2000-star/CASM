#!/usr/bin/env python3
from typing import List, Optional

from ..core.lexer import CASMLexer
from ..core.parser import CASMParser, ParseError
from ..core.tokens import LexerError


def check_syntax(content: str, filename: Optional[str] = None) -> List[dict]:
    """Check syntax using the project's lexer and parser.

    Returns a list of error dicts. Empty list means no syntax errors.
    """
    errors: List[dict] = []

    # First run the lexer to catch lexical errors
    try:
        lexer = CASMLexer()
        tokens = lexer.tokenize(content)
    except Exception as e:
        # LexerError may expose line/column attributes
        line = getattr(e, 'line', None)
        column = getattr(e, 'column', None)
        errors.append({'line': line, 'column': column, 'message': f'Lexer error: {e}'})
        return errors

    # Then run the parser to catch parse errors
    try:
        parser = CASMParser()
        parser.parse(tokens)
    except ParseError as pe:
        tok = getattr(pe, 'token', None)
        if tok is not None:
            errors.append({'line': tok.line, 'column': tok.column, 'message': pe.message})
        else:
            errors.append({'line': None, 'column': None, 'message': str(pe)})
    except Exception as e:
        # Generic fallback
        line = getattr(e, 'line', None)
        column = getattr(e, 'column', None)
        errors.append({'line': line, 'column': column, 'message': f'Parse error: {e}'})

    return errors


def format_errors(errors: List[dict], filename: Optional[str] = None) -> str:
    """Return a printable multi-line string for errors."""
    if not errors:
        return ""

    lines = []
    for err in errors:
        line = err.get('line')
        col = err.get('column')
        msg = err.get('message')
        location = filename or '<input>'
        if line is not None:
            location = f"{location}:{line}"
            if col is not None:
                location = f"{location}:{col}"
        lines.append(f"Syntax error at {location}: {msg}")

    return '\n'.join(lines)
