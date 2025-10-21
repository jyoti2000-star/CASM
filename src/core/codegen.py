from typing import List
import re
from .ast import ProgramNode, FunctionDeclarationNode, PrintlnNode, ReturnNode
from .diagnostics import DiagnosticEngine
from .ast import VarDeclarationNode, AssignmentNode, LiteralNode, IdentifierNode, BinaryOpNode, FunctionCallNode


class AssemblyCodeGenerator:
    def __init__(self, target_os: str = 'linux'):
        self.target_os = target_os
        self.data_section: List[str] = []
        self.rodata_section: List[str] = []
        self.text_section: List[str] = []
        self.str_counter = 0
        self.diagnostics = DiagnosticEngine()
        self.label_counter = 0
        # collected external symbols (from ExternDirectiveNode)
        self.externs = set()
        # argument register ordering (basic)
        if self.target_os == 'windows':
            self.arg_regs = ['rcx', 'rdx', 'r8', 'r9']
        else:
            self.arg_regs = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']

    def _emit_call_puts(self, label: str):
        # Emit code to call puts with address of label in appropriate register
        if self.target_os == 'windows':
            self.text_section.append(f'    lea rcx, [rel {label}]')
            self.text_section.append('    call puts')
        else:
            self.text_section.append(f'    lea rdi, [rel {label}]')
            self.text_section.append('    call puts')

    def _new_str_label(self) -> str:
        self.str_counter += 1
        return f"str{self.str_counter}"

    def generate(self, program: ProgramNode) -> str:
        # Walk program statements (functions)
        # collect externs from ExternDirectiveNode
        for stmt in program.statements:
            if stmt.__class__.__name__ == 'ExternDirectiveNode':
                name = getattr(stmt, 'name', None)
                if name:
                    # If the 'name' looks like a captured C code block (contains '\n' or large content),
                    # preserve it in the assembly output as a commented C block so users can see/compile it
                    # separately. Otherwise, map known headers to extern functions as before.
                    if '\n' in name or len(name) > 200 or 'windows.h' in name.lower():
                        # Emit the C block as comments in the text section so it survives
                        self.text_section.append('; --- begin embedded C code passthrough ---')
                        for line in name.splitlines():
                            self.text_section.append('; ' + line)
                        self.text_section.append('; --- end embedded C code passthrough ---')
                        continue
                    # map some known headers to extern functions
                    if 'math.h' in name or 'math' in name:
                        # common math functions
                        for fn in ('sin', 'cos', 'tan', 'sqrt'):
                            self.externs.add(fn)
                    else:
                        # treat other extern directives as symbol names
                        self.externs.add(name)

        for stmt in program.statements:
            if isinstance(stmt, FunctionDeclarationNode):
                self._emit_function(stmt)

        parts: List[str] = []

        # Ensure commonly used runtime symbols are exported when needed
        if any(re.search(r'\bcall\s+(_?puts)\b', line) for line in self.text_section):
            self.externs.add('puts')

        if self.externs:
            for ext in sorted(self.externs):
                parts.append(f'extern {ext}')
            parts.append('')

        if self.rodata_section:
            parts.append('section .rodata')
            parts.extend(self.rodata_section)
            parts.append('')

        parts.append('section .text')
        parts.extend(self.text_section)
        return '\n'.join(parts)

    def _emit_function(self, func: FunctionDeclarationNode):
        name = func.name
        self.text_section.append(f'global {name}')
        self.text_section.append(f'{name}:')
        self.text_section.append('    push rbp')
        self.text_section.append('    mov rbp, rsp')
        if self.target_os == 'windows':
            # reserve x64 shadow space for callees (32 bytes)
            self.text_section.append('    sub rsp, 32    ; shadow space')

        # simple local variable area tracking
        local_offset = 0
        locals_map = {}

        # First pass: compute locals map and total local allocation size
        for s in func.body or []:
            if isinstance(s, VarDeclarationNode):
                local_offset += 8
                locals_map[s.name] = -local_offset

        total_local_alloc = local_offset
        # align to 16 bytes for stack alignment
        if total_local_alloc % 16 != 0:
            pad = 16 - (total_local_alloc % 16)
            total_local_alloc += pad

        # emit single allocation for all locals (after shadow space if any)
        if total_local_alloc > 0:
            self.text_section.append(f'    sub rsp, {total_local_alloc}    ; alloc locals')

        # emit body
        # emit body via helper (handles nested blocks properly)
        self._emit_block(func.body or [], locals_map, total_local_alloc)

        # restore stack to rbp then pop rbp to correctly unwind frame
        self.text_section.append('    mov rsp, rbp')
        self.text_section.append('    pop rbp')
        self.text_section.append('    ret')

    def _emit_expr(self, expr, locals_map, local_offset):
        # Emit code to evaluate expr and leave result in rax
        if isinstance(expr, LiteralNode):
            if isinstance(expr.value, int):
                self.text_section.append(f'    mov rax, {expr.value}')
            else:
                # string literal: load address into rax
                label = self._new_str_label()
                self.rodata_section.append(f"{label}: db \"{expr.value}\", 0")
                self.text_section.append(f'    lea rax, [rel {label}]')

        elif isinstance(expr, IdentifierNode):
            name = expr.name
            if name in locals_map:
                self.text_section.append(f'    mov rax, [rbp{locals_map[name]}]')
            else:
                self.text_section.append('    mov rax, 0')

        elif isinstance(expr, BinaryOpNode):
            # evaluate left -> rax, push, evaluate right -> rbx, compute
            self._emit_expr(expr.left, locals_map, local_offset)
            self.text_section.append('    push rax')
            self._emit_expr(expr.right, locals_map, local_offset)
            self.text_section.append('    mov rbx, rax')
            self.text_section.append('    pop rax')
            op = expr.operator
            if op == '+':
                self.text_section.append('    add rax, rbx')
            elif op == '-':
                self.text_section.append('    sub rax, rbx')
            elif op == '*':
                self.text_section.append('    imul rax, rbx')
            elif op == '/':
                # idiv requires rdx:rax, keep simple and do unsigned div
                self.text_section.append('    cqo')
                self.text_section.append('    idiv rbx')
            elif op in ('<', '>', '<=', '>=', '==', '!='):
                # comparison: rax = left, rbx = right -> set rax to 1 or 0
                # cmp rax, rbx ; setcc al ; movzx rax, al
                if op == '<':
                    self.text_section.append('    cmp rax, rbx')
                    self.text_section.append('    setl al')
                elif op == '>':
                    self.text_section.append('    cmp rax, rbx')
                    self.text_section.append('    setg al')
                elif op == '<=':
                    self.text_section.append('    cmp rax, rbx')
                    self.text_section.append('    setle al')
                elif op == '>=':
                    self.text_section.append('    cmp rax, rbx')
                    self.text_section.append('    setge al')
                elif op == '==':
                    self.text_section.append('    cmp rax, rbx')
                    self.text_section.append('    sete al')
                elif op == '!=':
                    self.text_section.append('    cmp rax, rbx')
                    self.text_section.append('    setne al')
                self.text_section.append('    movzx rax, al')
            else:
                self.text_section.append('    ; unknown binop')

    def _emit_block(self, stmts, locals_map, local_offset):
        """Emit a sequence of statements (used for function bodies and nested blocks)."""
        for s in stmts:
            # Handle assignment expressed as BinaryOpNode(left=id, operator='=', right=expr)
            if isinstance(s, BinaryOpNode) and s.operator == '=' and isinstance(s.left, IdentifierNode):
                # evaluate right-hand side into rax
                self._emit_expr(s.right, locals_map, local_offset)
                name = s.left.name
                if name in locals_map:
                    self.text_section.append(f'    mov [rbp{locals_map[name]}], rax')
                continue
            # If node
            if s.__class__.__name__ == 'IfNode':
                cond = s.condition
                lbl_else = f"L{self.label_counter+1}"
                lbl_end = f"L{self.label_counter+2}"
                self.label_counter += 2
                # evaluate cond -> rax
                self._emit_expr(cond, locals_map, local_offset)
                self.text_section.append('    cmp rax, 0')
                self.text_section.append(f'    je {lbl_else}')
                # emit if body
                self._emit_block(s.if_body or [], locals_map, local_offset)
                self.text_section.append(f'    jmp {lbl_end}')
                self.text_section.append(f'{lbl_else}:')
                # else (empty for now)
                self.text_section.append(f'{lbl_end}:')
                continue

            # While node
            if s.__class__.__name__ == 'WhileNode':
                lbl_start = f"L{self.label_counter+1}"
                lbl_end = f"L{self.label_counter+2}"
                self.label_counter += 2
                self.text_section.append(f'{lbl_start}:')
                self._emit_expr(s.condition, locals_map, local_offset)
                self.text_section.append('    cmp rax, 0')
                self.text_section.append(f'    je {lbl_end}')
                self._emit_block(s.body or [], locals_map, local_offset)
                self.text_section.append(f'    jmp {lbl_start}')
                self.text_section.append(f'{lbl_end}:')
                continue

            # Label
            if s.__class__.__name__ == 'LabelNode':
                self.text_section.append(f'{s.name}:')
                continue

            # Inline assembly block
            if s.__class__.__name__ == 'AssemblyNode':
                # paste asm lines directly, but replace references to local variables
                # and indent to match function body
                def replace_ident(match):
                    name = match.group(0)
                    if name in locals_map:
                        return f'[rbp{locals_map[name]}]'
                    return name

                for line in s.code.splitlines():
                    if not line.strip():
                        self.text_section.append(line)
                        continue
                    transformed = re.sub(r"\b([A-Za-z_]\w*)\b", replace_ident, line)
                    self.text_section.append('    ' + transformed.lstrip())
                continue
            # Println
            if isinstance(s, PrintlnNode):
                for expr in s.expressions:
                    # expression may be a string literal or expression node
                    if isinstance(expr, str):
                        label = self._new_str_label()
                        self.rodata_section.append(f"{label}: db \"{expr}\", 0")
                        self._emit_call_puts(label)
                    else:
                        # evaluate expr to rax for possible future handling
                        self._emit_expr(expr, locals_map, local_offset)
                        pass

            # Variable declaration
            elif isinstance(s, VarDeclarationNode):
                # locals were allocated in a single prologue allocation
                # emit initializer if present
                if s.value is not None:
                    self._emit_expr(s.value, locals_map, local_offset)
                    # move rax to local slot
                    self.text_section.append(f'    mov [rbp{locals_map[s.name]}], rax')

            elif isinstance(s, AssignmentNode):
                # evaluate rhs -> rax
                self._emit_expr(s.value, locals_map, local_offset)
                # lhs identifier
                if isinstance(s.target, IdentifierNode):
                    name = s.target.name
                    if name in locals_map:
                        self.text_section.append(f'    mov [rbp{locals_map[name]}], rax')
                    else:
                        # global variable not supported; ignore
                        pass

            elif isinstance(s, FunctionCallNode):
                # very simple: only support puts-like calls with literal arg
                if isinstance(s.function, IdentifierNode) and s.function.name == 'puts' and s.arguments:
                    arg = s.arguments[0]
                    if isinstance(arg, LiteralNode) and isinstance(arg.value, str):
                        label = self._new_str_label()
                        self.rodata_section.append(f"{label}: db \"{arg.value}\", 0")
                        self._emit_call_puts(label)

            elif isinstance(s, ReturnNode):
                # Lower return value: if it's an integer literal emit mov rax, <int>
                # otherwise evaluate the expression into rax using _emit_expr
                if s.value is None:
                    self.text_section.append('    mov rax, 0')
                else:
                    # integer literal
                    if isinstance(s.value, LiteralNode) and isinstance(s.value.value, int):
                        self.text_section.append(f'    mov rax, {s.value.value}')
                    else:
                        # evaluate arbitrary expression into rax
                        self._emit_expr(s.value, locals_map, local_offset)
                # jump to function epilogue by emitting mov rsp, rbp; pop rbp; ret
                self.text_section.append('    mov rsp, rbp')
                self.text_section.append('    pop rbp')
                self.text_section.append('    ret')


