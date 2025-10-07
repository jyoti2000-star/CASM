# CASM - Clean Assembly Language Compiler

A clean, focused assembly language compiler with high-level constructs for educational and practical use.

## Features

CASM provides essential programming constructs while generating efficient x86-64 Windows assembly:

- **Control Flow**: `%if/%else/%endif`, `%while/%endwhile`, `%for/%endfor`
- **Variables**: `%var name value`
- **I/O**: `%println message`, `%scanf format variable`
- **Comments**: `;` for single-line comments
- **Inline Assembly**: Direct assembly code support

## Quick Start

### Installation

Requires:
- Python 3.7+
- NASM assembler
- MinGW-w64 (for Windows cross-compilation)
- Wine (for running executables on macOS/Linux)

```bash
# Install dependencies on macOS
brew install nasm mingw-w64 wine-stable

# Install dependencies on Ubuntu/Debian
sudo apt install nasm gcc-mingw-w64 wine
```

### Usage

```bash
# Compile and run
python3 casm.py run hello.casm

# Compile to executable only
python3 casm.py compile program.casm

# Generate assembly code only
python3 casm.py asm test.casm
```

## Language Reference

### Variable Declaration
```assembly
%var counter 0
%var message "Hello World"
%var buffer 1024
```

### Control Flow
```assembly
%if counter < 10
    %println "Counter is less than 10"
%else
    %println "Counter is 10 or greater"
%endif

%while counter < 5
    %println counter
    %var counter counter+1
%endwhile

%for i in range(10)
    %println i
%endfor
```

### Input/Output
```assembly
%println "Enter a number:"
%scanf "%d" number
%println "You entered:"
%println number
```

### Comments and Assembly
```assembly
; This is a comment
%println "Hello"    ; End-of-line comment

; Inline assembly is supported
mov rax, 42
push rax
```

## Example Programs

### Hello World
```assembly
; hello.casm
%println "Hello, World!"
```

### Simple Calculator
```assembly
; calc.casm
%var num1 0
%var num2 0
%var result 0

%println "Enter first number:"
%scanf "%d" num1

%println "Enter second number:"
%scanf "%d" num2

; Add the numbers (simplified)
%var result num1+num2
%println "Result:"
%println result
```

### Counting Loop
```assembly
; count.casm
%var i 0

%while i < 10
    %println i
    %var i i+1
%endwhile

%println "Done counting!"
```

## Architecture

CASM uses a clean 4-stage compilation pipeline:

1. **Lexical Analysis** (`src/core/lexer.py`) - Tokenizes CASM source code
2. **Parsing** (`src/core/parser.py`) - Builds Abstract Syntax Tree (AST)
3. **Code Generation** (`src/core/codegen.py`) - Converts AST to x86-64 assembly
4. **Assembly & Linking** (`src/compiler.py`) - NASM + MinGW-w64 toolchain

### Directory Structure
```
casm_clean/
├── casm.py              # Main entry point
├── src/
│   ├── core/
│   │   ├── tokens.py    # Token definitions
│   │   ├── lexer.py     # Lexical analyzer
│   │   ├── parser.py    # Parser
│   │   ├── ast_nodes.py # AST node definitions
│   │   └── codegen.py   # Code generator
│   ├── stdlib/
│   │   └── minimal.py   # Minimal standard library
│   └── compiler.py      # Main compiler class
├── tests/               # Test files
├── examples/            # Example programs
└── output/             # Generated files
```

## Target Platform

- **Architecture**: x86-64 only
- **Operating System**: Windows (PE32+ executables)
- **Calling Convention**: Windows x64
- **Assembler**: NASM (Intel syntax)
- **Linker**: MinGW-w64 GCC

Cross-compilation from macOS/Linux is supported via MinGW-w64 and Wine.

## Development

The codebase emphasizes:
- **Clean Architecture**: Separation of concerns, single responsibility
- **Focused Feature Set**: Only essential language constructs
- **Educational Value**: Clear, readable code for learning compiler design
- **Practical Use**: Generates working executables

## License

See LICENSE file for details.