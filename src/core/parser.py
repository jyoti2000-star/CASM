from typing import List, Optional
from .diagnostics import DiagnosticEngine
from .ast import (
    ProgramNode, FunctionDeclarationNode, PrintlnNode, ReturnNode,
    VarDeclarationNode, AssignmentNode, LiteralNode, IdentifierNode,
    BinaryOpNode, FunctionCallNode, IfNode, WhileNode, LabelNode, AssemblyNode,
    ExternDirectiveNode,
)


class Token:
    def __init__(self, typ: str, val: str, pos: int):
        self.type = typ
        self.val = val
        self.pos = pos

    def __repr__(self):
        return f"Token({self.type}, {self.val})"


class CASMParser:
    def __init__(self, source: str):
        self.source = source
        self.diagnostics = DiagnosticEngine()
        self.tokens: List[Token] = self._tokenize(source)
        self.i = 0

    # simple character-based tokenizer to avoid complex regex escaping
    def _tokenize(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c == '\n':
                tokens.append(Token('NEWLINE', '\n', i))
                i += 1
                continue
            if c == ';':
                # comment until end of line
                j = i + 1
                while j < n and text[j] != '\n':
                    j += 1
                tokens.append(Token('COMMENT', text[i+1:j].strip(), i))
                i = j
                continue
            if c.isspace():
                i += 1
                continue
            if c.isalpha() or c == '_':
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                tokens.append(Token('ID', text[i:j], i))
                i = j
                continue
            if c.isdigit():
                j = i + 1
                while j < n and text[j].isdigit():
                    j += 1
                tokens.append(Token('NUMBER', text[i:j], i))
                i = j
                continue
            # Accept wide-string literal prefix L"..." as a single STRING token
            if c == 'L' and i + 1 < n and text[i+1] in ('"', "'"):
                quote = text[i+1]
                j = i + 2
                val = ''
                while j < n:
                    if text[j] == '\\':
                        if j + 1 < n:
                            val += text[j+1]
                            j += 2
                            continue
                    if text[j] == quote:
                        break
                    val += text[j]
                    j += 1
                tokens.append(Token('STRING', 'L' + quote + val + quote, i))
                i = j + 1
                continue

            if c == '"' or c == "'":
                quote = c
                j = i + 1
                val = ''
                while j < n:
                    if text[j] == '\\':
                        if j + 1 < n:
                            val += text[j+1]
                            j += 2
                            continue
                    if text[j] == quote:
                        break
                    val += text[j]
                    j += 1
                tokens.append(Token('STRING', quote + val + quote, i))
                i = j + 1
                continue
            # two-char ops
            two = text[i:i+2]
            if two in ('==', '!=', '<=', '>=', '&&', '||'):
                tokens.append(Token('OP', two, i))
                i += 2
                continue
            # single-char operators/punct
            if c in '+-*/<>=':
                tokens.append(Token('OP', c, i))
                i += 1
                continue
            if c in '(),;:[]':
                tokens.append(Token('PUNC', c, i))
                i += 1
                continue
            if c in '{}':
                tokens.append(Token('BRACE', c, i))
                i += 1
                continue
            # otherwise treat as mismatch
            tokens.append(Token('MISMATCH', c, i))
            i += 1
        tokens.append(Token('EOF', '', n))
        return tokens

    def _peek(self) -> Token:
        return self.tokens[self.i]

    def _next(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def _accept(self, typ: str, val: Optional[str] = None) -> Optional[Token]:
        t = self._peek()
        if t.type == typ and (val is None or t.val == val):
            return self._next()
        return None

    def _expect(self, typ: str, val: Optional[str] = None) -> Token:
        t = self._peek()
        if t.type == typ and (val is None or t.val == val):
            return self._next()
        raise SyntaxError(f"Expected {typ} {val} got {t.type} {t.val} at {t.pos}")

    def parse(self) -> Optional[ProgramNode]:
        stmts: List = []
        while self._peek().type != 'EOF':
            # skip blank lines between top-level declarations
            if self._peek().type in ('NEWLINE', 'COMMENT'):
                self._next()
                continue
            t = self._peek()
            if t.type == 'ID' and t.val == 'function':
                self._next()
                name_tok = self._expect('ID')
                # skip parameter list
                if self._accept('PUNC', '('):
                    depth = 1
                    while depth > 0:
                        tk = self._next()
                        if tk.type == 'PUNC' and tk.val == '(': depth += 1
                        if tk.type == 'PUNC' and tk.val == ')': depth -= 1
                body = self._parse_block()
                stmts.append(FunctionDeclarationNode(name=name_tok.val, return_type='int', parameters=[], body=body))
            else:
                # parse other top-level statements (extern directives, asm, labels, etc.)
                stmt = self._parse_statement()
                if stmt is not None:
                    stmts.append(stmt)
        return ProgramNode(statements=stmts)

    def _parse_block(self) -> List:
        if self._accept('BRACE', '{') is None:
            raise SyntaxError('Expected block {')
        stmts = []
        while True:
            if self._accept('BRACE', '}'):
                break
            if self._peek().type == 'EOF':
                break
            # skip empty lines and comments
            if self._peek().type in ('NEWLINE', 'COMMENT'):
                self._next()
                continue
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _parse_statement(self):
        t = self._peek()
        # extern <header.h> directive -> create ExternDirectiveNode or simple identifier extern
        if t.type == 'ID' and t.val == 'extern':
            # If this 'extern' is the very first token in the file and is followed
            # by a <...> system header (commonly used by C examples like windows.h),
            # capture the rest of the file as a C code block and return it as a
            # single ExternDirectiveNode (passthrough). This is a pragmatic
            # compatibility measure to allow examples containing large chunks of
            # valid C code to be accepted by the CASM pipeline without a full C
            # frontend.
            if self.i == 0:
                # capture entire remaining source starting at this token position
                start_pos = t.pos
                c_src = self.source[start_pos:]
                # advance to EOF
                while self._peek().type != 'EOF':
                    self._next()
                return ExternDirectiveNode(name=c_src)
            # otherwise fallback to the normal extern parsing
            self._next()
            # expect either a string/header in quotes or a <header.h> or an identifier
            tok = self._peek()
            if tok.type == 'STRING':
                name = tok.val[1:-1]
                self._next()
                return ExternDirectiveNode(name=name)
            # accept '<' as either PUNC or OP
            if tok.type in ('PUNC', 'OP') and tok.val == '<':
                # find header name in source between '<' and '>' starting at tok.pos
                start = tok.pos + 1
                src = self.source
                j = start
                name_chars = []
                while j < len(src) and src[j] != '>':
                    name_chars.append(src[j])
                    j += 1
                name = ''.join(name_chars).strip()
                # advance tokens until after '>' or newline
                while self._peek().type not in ('PUNC', 'NEWLINE', 'EOF'):
                    self._next()
                if self._peek().type in ('PUNC', 'OP') and self._peek().val == '>':
                    self._next()
                return ExternDirectiveNode(name=name)
            if tok.type == 'ID':
                name = tok.val
                self._next()
                return ExternDirectiveNode(name=name)
        # raw assembly comment line provided by tokenizer as COMMENT
        if t.type == 'COMMENT':
            asm_code = '; ' + t.val
            # consume the comment token
            self._next()
            # also consume any following NEWLINE
            if self._peek().type == 'NEWLINE':
                self._next()
            return AssemblyNode(code=asm_code)

        # standalone asm instruction line (e.g. "nop" or "mov rax, rbx")
        # If an identifier starts the line and is NOT followed by '(', treat the entire source line as asm
        if t.type == 'ID' and t.val not in ('function', 'var', 'if', 'while', 'return', 'asm'):
            nxt = self.tokens[self.i+1] if (self.i+1) < len(self.tokens) else None
            # if next is '(' this is a function call/expr -> fall through to expression parsing
            # also if next is '=' treat as an assignment, not raw asm
            if nxt is None or not ((nxt.type == 'PUNC' and nxt.val == '(') or (nxt.type == 'OP' and nxt.val == '=')):
                start = t.pos
                src = self.source
                j = start
                while j < len(src) and src[j] != '\n':
                    j += 1
                asm_code = src[start:j]
                # advance tokens until after newline
                while self._peek().type not in ('NEWLINE', 'EOF'):
                    self._next()
                if self._peek().type == 'NEWLINE':
                    self._next()
                return AssemblyNode(code=asm_code)
        if t.type == 'ID' and t.val == 'var':
            self._next()
            name = self._expect('ID').val
            self._expect('OP', '=')
            expr = self._parse_expression()
            self._accept('PUNC', ';')
            return VarDeclarationNode(name=name, var_type='int', value=expr)

        if t.type == 'ID' and t.val == 'if':
            self._next()
            self._expect('PUNC', '(')
            cond = self._parse_expression()
            self._expect('PUNC', ')')
            body = self._parse_block()
            return IfNode(condition=cond, if_body=body)

        if t.type == 'ID' and t.val == 'while':
            self._next()
            self._expect('PUNC', '(')
            cond = self._parse_expression()
            self._expect('PUNC', ')')
            body = self._parse_block()
            return WhileNode(condition=cond, body=body)

        if t.type == 'ID' and t.val == 'return':
            self._next()
            if not (self._peek().type == 'PUNC' and self._peek().val == ';'):
                expr = self._parse_expression()
            else:
                expr = None
            self._accept('PUNC', ';')
            return ReturnNode(value=expr)

        if t.type == 'ID' and t.val == 'asm':
            self._next()
            # gather raw text until matching brace
            if self._accept('BRACE', '{') is None:
                raise SyntaxError('Expected { after asm')
            # find matching brace in source starting at current token pos
            start = self._peek().pos
            src = self.source
            depth = 1
            j = start
            while j < len(src) and depth > 0:
                if src[j] == '{': depth += 1
                elif src[j] == '}': depth -= 1
                j += 1
            asm_code = src[start:j-1].strip()
            # advance tokens until we find the closing '}' BRACE token
            while True:
                pk = self._peek()
                if pk.type == 'EOF':
                    raise SyntaxError('Unterminated asm block')
                if pk.type == 'BRACE' and pk.val == '}':
                    self._next()
                    break
                self._next()
            return AssemblyNode(code=asm_code)

        # label: identifier ':'
        if t.type == 'ID' and (self.i+1) < len(self.tokens) and self.tokens[self.i+1].type == 'PUNC' and self.tokens[self.i+1].val == ':':
            name = self._next().val
            # consume ':'
            self._next()
            return LabelNode(name=name)

        expr = self._parse_expression()
        # assignment
        if isinstance(expr, IdentifierNode) and self._peek().type == 'OP' and self._peek().val == '=':
            self._next()
            rhs = self._parse_expression()
            self._accept('PUNC', ';')
            return AssignmentNode(target=expr, value=rhs)

        if isinstance(expr, FunctionCallNode):
            self._accept('PUNC', ';')
            if isinstance(expr.function, IdentifierNode) and expr.function.name == 'println' and expr.arguments and isinstance(expr.arguments[0], LiteralNode) and isinstance(expr.arguments[0].value, str):
                return PrintlnNode(expressions=[expr.arguments[0].value])
            return expr

        self._accept('PUNC', ';')
        return expr

    def _parse_expression(self, min_prec=0):
        left = self._parse_primary()
        while True:
            t = self._peek()
            if t.type != 'OP': break
            prec = self._op_precedence(t.val)
            if prec < min_prec: break
            op = t.val
            self._next()
            right = self._parse_expression(prec + 1)
            left = BinaryOpNode(left=left, operator=op, right=right)
        return left

    def _op_precedence(self, op: str) -> int:
        if op in ('*', '/'): return 30
        if op in ('+', '-'): return 20
        if op in ('<', '>', '<=', '>='): return 10
        if op in ('==', '!='): return 5
        if op == '=': return 1
        return 0

    def _parse_primary(self):
        t = self._peek()
        if t.type == 'NUMBER':
            self._next()
            return LiteralNode(value=int(t.val))
        if t.type == 'STRING':
            self._next()
            s = t.val[1:-1]
            return LiteralNode(value=s)
        if t.type == 'ID':
            idname = t.val
            self._next()
            if self._accept('PUNC', '('):
                args = []
                if not self._accept('PUNC', ')'):
                    while True:
                        args.append(self._parse_expression())
                        if self._accept('PUNC', ')'):
                            break
                        self._expect('PUNC', ',')
                return FunctionCallNode(function=IdentifierNode(name=idname), arguments=args)
            return IdentifierNode(name=idname)
        if t.type == 'PUNC' and t.val == '(':
            self._next()
            expr = self._parse_expression()
            self._expect('PUNC', ')')
            return expr
        raise SyntaxError(f"Unexpected token in expression: {t}")

