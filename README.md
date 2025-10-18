CASM (C Assembly) is a modern, readable assembly-oriented language that compiles to x86-64 (Windows) assembly. It aims to blend simple high-level constructs with low-level assembly and inline C where helpful.

## Highlights (what's new)

- Implicit embedded C detection: C-style lines (lines starting with `#` or ending with `;`) are detected automatically and collected into C blocks. You no longer need legacy `_c_` / `_endc_` markers.
- `var` declarations: use `var <type> <name> [size] [= value]` to declare typed variables. Buffers use `resb` in `.bss`, strings and other initialized values go into `.data`.
- `extern` forwarding: `extern <header>` or `extern "header.h"`/`extern <header.h>` can forward includes into the generated combined C file; plain `extern symbol` emits assembler `extern symbol`.
- Salted/randomized names: labels and `var_` names include a per-run salt to avoid predictable symbol names.
- Cleaner CLI: quieter output by default, a single consolidated error or a single final success message is printed.

## Requirements

- Python 3.8+
- NASM (for assembling)
- x86_64-w64-mingw32-gcc (preferred) or `gcc`/`clang` as fallbacks for compiling embedded C

On macOS you can install dependencies with Homebrew:

```bash
brew install nasm mingw-w64
```

On Debian/Ubuntu:

```bash
sudo apt install nasm gcc-mingw-w64
```

Note: running compiled Windows executables locally requires Wine; building works without it.

## Quick usage

The CLI now uses a single-entry invocation. Run `casm.py <file>` and you'll be presented with an interactive selector to choose which artifact to produce (executable, assembly, flat binary, or object file).

Interactive mode (recommended for local use):

```bash
python3 casm.py examples/main.asm
# then select one of: Compile to Executable | Generate Assembly | Compile to Binary | Compile to Object File
```

Non-interactive flags (useful for scripts / CI):

- `--cflags "<flags>"` — pass extra C compilation flags (overrides auto-detected pkg-config flags).
- `--ldflags "<flags>"` — pass extra linker flags.
  -- `-d` | `--debug-save` — save intermediate C and assembly artifacts for debugging.
- `-t` | `--type` <asm|exe|bin|obj> — non-interactive selection (useful for scripts/CI)

Examples:

Generate assembly and write it to `output/`:

```bash
python3 casm.py examples/main.asm
# select "Generate Assembly"
```

Non-interactive example (produce assembly without the interactive prompt):

```bash
python3 casm.py examples/main.asm -t asm
```

You can also use the newer, shorter unified directives (no leading "U" and in lowercase).
For example the file `examples/ufunc_nou.asm` demonstrates `func` / `endfunc` / `call` / `ret` / `exit` usage.

Generate assembly from the new-style example non-interactively:

```bash
python3 casm.py examples/ufunc_nou.asm -t asm
```

If you prefer to run the standalone transpiler (converts unified directives into target NASM):

```bash
# Transpile using the bundled transpiler (module form)
python3 -m src.utils.asm_transpiler examples/ufunc_nou.asm --target linux --arch x86_64 -o output/ufunc_nou.asm

# Or call the script directly (same effect)
python3 src/utils/asm_transpiler.py examples/ufunc_nou.asm --target linux --arch x86_64 -o output/ufunc_nou.asm
```

Compile to a Windows executable (non-interactive, passing custom linker flags):

```bash
python3 casm.py examples/main.asm --ldflags "-L/some/mingw/libs -lSDL2"
# select "Compile to Executable" in the interactive prompt
```

Produce a flat binary (bin) and save intermediate files for debugging:

```bash
python3 casm.py examples/main.asm -d --cflags "-I/some/include" \
  --ldflags "-L/some/lib -lSDL2"
# select "Compile to Binary"
```

Install and run globally

You can install a small wrapper script that lets you run CASM as a regular command (`casm`) on your system. The repository includes `install.sh` which installs a convenient wrapper into `/usr/local/bin` (or a platform-appropriate location).

On macOS / Linux:

```bash
./install.sh           # installs wrapper into /usr/local/bin by default (may prompt for sudo)
# then run using the global wrapper:
casm c examples/badusb.asm   # 'c' is a short alias for 'compile'
casm asm examples/badusb.asm # generate assembly only
```

If `install.sh` reports that the destination is not on your PATH, add the printed directory to your PATH (example shown by the installer).

## Language reference (current syntax)

This section describes the syntax the current parser and lexer accept.

### High-level statements

- `var` declarations
- `if` / `else` / `endif`
- `while` / `endwhile`
- `for` / `endfor` with `for i in range(N)` syntax
- `print` and `scan` (I/O)
- `extern` for assembler externs or forwarded C includes
- Inline C lines are detected implicitly (see below)
- Raw assembly lines are accepted and passed through

### `var` declaration

Syntax (examples):

```asm
var int n 5
var int n = 5
var str name "Alice"
var buffer buf[128]
var int array[4] 1,2,3,4
```

Notes:

- `var <type> <name> [size] [= value]` — `size` is optional and used for arrays/buffers.
- `buffer` declarations generate `.bss` (uninitialized) entries using `resb`.
- `str` and initialized variables go to `.data` using `db`, `dd`, etc.
- If no value is provided, sensible defaults are used (0 for ints, `""` for strings, etc.).

Supported low-level types and assembler directive mapping

CASM's `var` declarations are intentionally flexible and can map directly to low-level assembler storage directives. This makes it easy to declare exact-sized storage or constants when you need precise control.

Common mappings and examples:

- Equates / constants:

  - `equ` (constant/equate) — use `var equ NAME = 4` or `var const NAME = 4` to emit an assembler `NAME equ 4` (read-only constant).

- Initialized storage (placed in `.data`):

  - `db` / byte-sized values: `var byte b = 0x1` -> `b db 0x1`
  - `dw` / word (2 bytes): `var word w = 0x1234` -> `w dw 0x1234`
  - `dd` / dword (4 bytes): `var int v = 5` or `var dword v = 5` -> `v dd 5`
  - `dq` / quad (8 bytes): `var qword x = 0` -> `x dq 0`

- Uninitialized / reserve in `.bss` (buffers):
  - `resb`, `resw`, `resd`, `resq` — use `var buffer buf[128]` or `var resb buf 128` to reserve raw bytes without initializing them.

Notes:

- The `var` front-end accepts common, human-friendly type names (`int`, `byte`, `qword`, `buffer`, `str`, etc.) and maps them to the appropriate assembler directive (`dd`, `db`, `dq`, `resb`, ...).
- Use `str` or `db` style declarations for textual data so they land in `.data` with a terminating zero when appropriate.
- If you need a raw equate or explicit directive, prefer `var equ` / `var const` or the corresponding explicit type (`var db`, `var dd`, `var dq`, `var resb`) — the compiler will emit the equivalent assembler directive.

### Assignments

```asm
name = 10
n = n + 1
sum = a + b
```

Assignment parsing accepts an identifier followed by `=` and the rest of the line as an expression.

### Control flow

If statements:

```asm
if n >= 10
    print "big"
else
    print "small"
endif
```

While loops:

```asm
while i < 10
    i = i + 1
endwhile
```

For loops:

```asm
for i in range(10)
    print i
endfor
```

### Input / Output

- `print <string|expression>` — prints a string or expression.
- `scan <identifier>` — reads a value into a CASM variable.

### Externs and C include forwarding

- `extern math.h` (or `extern <math.h>` / `extern "math.h"`) will be forwarded as an include to the generated C file used to compile embedded C blocks.
- `extern SomeSymbol` (plain identifier) is treated as an assembler extern and emitted as `extern SomeSymbol` in the final assembly.

### Embedded C (implicit blocks)

The lexer detects C-style lines automatically. A line is considered C if it:

- starts with `#` (preprocessor directive), or
- ends with a semicolon `;` (C statement), or
- is part of a contiguous sequence of such lines.

The compiler groups contiguous C lines into a C block, forwards recorded headers (from `extern`), exposes CASM variables to the C file as `extern` declarations and `#define` aliases, then compiles the combined C with GCC (or a fallback). The assembly extracted from compiled C is inserted back into the CASM output where the C block was.

C blocks may reference CASM variables using either the CASM variable name (an alias is provided via `#define`) or their assembler label.

If GCC fails to compile the combined C file, the build aborts and the top-level CLI prints a concise error message.

### Raw assembly

Any line that is not recognized as a CASM keyword/statement is treated as raw assembly (the lexer will emit `ASSEMBLY_LINE` tokens). These are passed directly to the output, possibly with minor normalization by the pretty-printer and assembly fixer.

## Unified directives (U-prefixed and short aliases)

CASM now supports a unified, platform-abstract directive set implemented by the bundled asm transpiler. Each directive has a canonical U-prefixed form (used internally and by older code) and a short, friendlier alias that you can write without the leading `U` and in lowercase. The transpiler accepts both styles interchangeably.

Key points:

- You can write `UFUNC name` or `func name` (both accepted).
- The transpiler recognizes directives case-insensitively and will also accept end markers both with and without `U` (e.g. `UENDMACRO` or `endmacro`).
- Use the U-prefixed forms when you want to be explicit; use the lowercase aliases for brevity.

Common directives and their short aliases

- Functions and calls

  - UFUNC <name> (alias: func <name>) — begin function (emits prologue)
  - UENDFUNC (alias: endfunc) — end function
  - UCALL <name> [args...] (alias: call <name> [args...]) — call function, args handled by transpiler
  - URET [value] (alias: ret [value]) — return from function

- Parameters / locals

  - UPARAM <index> <dest> (alias: param <index> <dest>) — move nth argument into local register/var
  - ULOCAL <bytes> (alias: local <bytes>) — reserve local stack space (bytes)

- Data and constants

  - USTR <label> "text" (alias: str <label> "text") — define a string constant
  - UWSTR <label> "text" (alias: wstr <label> "text") — wide string (Windows) or normal otherwise
  - UCONST <name> <value> (alias: const <name> <value>) — assembler equ/constant
  - UARRAY <label> <type> <values> (alias: array <label> <type> <values>) — define an array
  - UBYTES <label> <count> (alias: bytes <label> <count>) — reserve bytes in BSS

- I/O and libc helpers

  - UPRINT <label> (alias: print <label>) — call puts/prints for a label
  - UPRINTF "fmt" [args...] (alias: printf "fmt" [args...]) — formatted print
  - UEXIT <code> (alias: exit <code>) — program exit (transpiler emits correct syscall or API)

- Memory and syscalls

  - UMALLOC <size> [dest] (alias: malloc <size> [dest])
  - UFREE <ptr> (alias: free <ptr>)
  - USYSCALL <name> [args] (alias: syscall <name> [args]) — platform syscall wrapper

- Control / optimization

  - ULOOP <label> [count] (alias: loop <label> [count])
  - UENDLOOP (alias: endloop)
  - UOPTIMIZE <directive> (alias: optimize <directive>)
  - UPRAGMA <text> (alias: pragma <text>)

- Inline / macros

  - UINLINE [lang] (alias: inline [lang]) — begin inline block
  - UENDINLINE / ENDINLINE — end inline block
  - UMACRO <name> [params] (alias: macro <name> [params]) — begin macro
  - UENDMACRO / ENDMACRO — end macro
  - UEXPAND <name> [args] (alias: expand <name> [args]) — expand macro

- Misc utilities
  - UPUSH <regs> / UPOP <regs> (alias: push/pop) — save/restore registers
  - UALIGN <n> (alias: align <n>) — align to boundary
  - UCOMMENT <text> (alias: comment <text>) — assembly comment emission
  - GLOBAL <name> / EXTERN <name> — same as before; exports/imports

Examples

Function + call (short aliases):

```asm
func myfunc
    mov eax, 7
    ret eax
endfunc

call myfunc
```

Same using canonical U-forms:

```asm
UFUNC myfunc
    mov eax, 7
    URET eax
UENDFUNC

UCALL myfunc
```

Formatted print + exit:

```asm
printf "%s\n", msg
exit 0
```

Notes and migration tips

- The transpiler accepts both `U`-prefixed directives and the short aliases. You can migrate incrementally: existing files with `U*` directives continue to work.
- If you prefer the small, human-friendly form, use the short aliases (lowercase). The code generator (codegen) can be updated to emit the short forms automatically — tell me if you want me to do that across the project.
- For C-interop, keep using `extern` for forwarded includes; the transpiler will still generate correct externs for libc functions it needs (printf, malloc, free, etc.).

Advanced: argument passing and calling conventions

The `UCALL` / `call` directive lets the transpiler handle platform ABI differences for you. When you write:

```asm
call printf rdi rsi
```

the transpiler will place arguments in the correct registers for the target platform (System V vs Windows x64) and handle shadow space/stack alignment when needed. Use explicit argument expressions (registers, immediates, or memory operands) separated by commas.

If you want the call to push extra stack args (beyond register arguments), list them after the register args; the transpiler will emit pushes and cleanup.

---

## Examples

Here are updated examples that match the current syntax and behaviors.

### Simple hybrid example (`examples/hybrid_example.casm`)

```asm
; Hybrid CASM program with implicit C blocks

var int n 5
var int fact 0
var int sum 0
var buffer buf[125]

print " Hybrid Example: C + Assembly "
print "Calculating factorial of"
print n

; C will be detected implicitly (lines ending with ';')
printf("Starting C block;\n");
int i = 0;
; compute factorial in C and store into CASM variable `fact`
int tmp = 1;
for (i = 1; i <= n; i++) tmp *= i;
var_fact = tmp;

print "Now compute sum 0..(n-1) using raw assembly"

; Raw assembly block
mov ecx, 0
mov eax, 0
sum_loop:
    add eax, ecx
    inc ecx
    cmp ecx, dword [rel var_n]
    jl sum_loop

mov dword [rel var_sum], eax

print "Done"
```

## Tooling / internals

High-level pipeline:

1. Lexer: tokenizes the source; detects implicit C lines and emits `C_INLINE` tokens for them.
2. Parser: builds AST nodes including `VarDeclarationNode`, `IfNode`, `WhileNode`, `ForNode`, `CCodeBlockNode`, `ExternDirectiveNode`.
3. Code generation: `AssemblyCodeGenerator` visits AST, produces `.data`, `.bss`, `.text` sections and placeholders for C blocks.
4. C processing: `c_processor` collects forwarded includes and C blocks, writes a combined C file, runs GCC to produce assembly, extracts relevant assembly fragments and string constants.
5. Prettifier/Assembly fixer: `prettify` organizes sections and ensures `str_... db ...` lines are in `.data` (not `.bss`); `assembly_fixer` applies NASM-friendly transformations.

## Contributing

If you add syntax or change parsing behavior, update the `README.md` and add small test cases under `examples/` and `test/` to validate the behavior.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Docker helper (cross-build)

If you don't have a native cross-toolchain installed, the easiest and most
reproducible way to build Linux/Windows binaries from macOS or other hosts is
to use Docker. The repository includes a small helper script that runs CASM
inside a linux/amd64 container and writes the resulting artifacts back into
your project's `output/` directory.

Usage (example):

```bash
# make the script executable once
chmod +x tools/docker-build-linux-x86_64.sh

# Build a linux/amd64 executable from macOS (or any host with Docker):
./tools/docker-build-linux-x86_64.sh examples/ufunc_nou.asm --platform linux --arch x86_64 -t exe
```

What the script does:

- Mounts the current repository into a Debian container (linux/amd64).
- Installs `nasm`, `build-essential` and `python3` inside the container.
- Runs `python3 casm.py` with the provided arguments so the assembly and
  linking steps happen in a linux/amd64 environment.

This avoids installing cross-linkers on the host and is the recommended path
for one-off cross-builds or CI integration. The produced files are available in
`output/` on the host filesystem because the repository root is bind-mounted.
