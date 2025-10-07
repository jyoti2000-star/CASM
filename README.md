CASM (C Assembly) provides essential programming constructs while generating efficient x86-64 Windows assembly:

- **Control Flow**: `%if/%else/%endif`, `%while/%endwhile`, `%for/%endfor`
- **Variables**: `%var name value`
- **I/O**: `%println message`, `%scanf format variable`
- **Comments**: `;` for single-line comments
- **Inline Assembly**: Direct assembly code support
- **C Integration**: `%extern` headers and `%!` embedded C code blocks
- **Hybrid Programming**: Seamless C and Assembly interoperability

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
# Compile and run CASM programs
python3 casm.py compile hello.casm

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

### Comments and Raw Assembly
```assembly
; This is a comment
%println "Hello"    ; End-of-line comment

; Raw assembly is fully supported alongside CASM constructs
mov rax, 42
push rax
pop rbx

; Mix CASM variables with raw assembly
%var my_number 100
mov rax, [my_number]    ; Load CASM variable into register
add rax, 50             ; Add 50 using raw assembly
mov [my_number], rax    ; Store back to CASM variable

; Raw assembly loops and control flow
%var counter 0
loop_start:
    mov rax, [counter]
    inc rax
    mov [counter], rax
    cmp rax, 10
    jl loop_start

; System calls and low-level operations
mov rax, 1              ; sys_write
mov rdi, 1              ; stdout
mov rsi, message        ; message buffer
mov rdx, 13             ; message length
syscall

; Stack operations
push rbp
mov rbp, rsp
sub rsp, 32             ; Allocate stack space

; Register manipulation
xor rax, rax            ; Clear register
mov rbx, 0x12345678     ; Load immediate value
shl rbx, 4              ; Shift left
or rax, rbx             ; Bitwise OR

; Function calls and returns
call my_function
mov rsp, rbp
pop rbp
ret

my_function:
    ; Raw assembly function
    mov rax, 42
    ret
```

### C Code Integration (CASM C Assembly)

CASM supports seamless integration with C code through extern declarations and embedded C blocks:

#### External Headers
```assembly
%extern stdio.h
%extern math.h
%extern string.h
```

#### Embedded C Code Blocks
```assembly
%extern stdio.h

%var number 42

; Embed C code with %!
%! printf("Hello from C! Number: %d\n", 42);

%if number > 40
    %println "CASM: Number is greater than 40"
    %! printf("C: Indeed, %d > 40\n", 42);
%endif

; Multi-line C code blocks
%! int factorial(int n) {
%!     if (n <= 1) return 1;
%!     return n * factorial(n - 1);
%! }
%! 
%! int result = factorial(5);
%! printf("Factorial of 5 = %d\n", result);
```

#### Variable Sharing
CASM variables are automatically available in C code as extern declarations:

```assembly
%extern stdio.h

%var counter 10
%var message "Hello World"

%! printf("Counter from CASM: %d\n", counter);
%! printf("Message from CASM: %s\n", message);

; C code can also define functions callable from assembly
%! int add_numbers(int a, int b) {
%!     return a + b;
%! }
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

### C Integration Example
```assembly
; hybrid.casm - Demonstrates C and Assembly integration
%extern stdio.h
%extern math.h

%var radius 5.0
%var area 0.0

%println "=== Circle Area Calculator ==="
%println "Using C math functions with CASM variables"

%! // Calculate area using C math functions
%! double pi = 3.14159;
%! double calculated_area = pi * radius * radius;
%! printf("Radius: %.2f\n", radius);
%! printf("Area calculated in C: %.2f\n", calculated_area);

%var area calculated_area
%println "Area stored in CASM variable:"
%println area

%if area > 70
    %println "That's a large circle!"
    %! printf("Indeed, %.2f is quite large!\n", calculated_area);
%endif
```

### Assembly Integration Example
```assembly
; mixed.casm - Demonstrates raw assembly mixed with CASM constructs
%var array_size 5
%var sum 0
%var numbers_array 1024  ; Allocate space for array

%println "=== Assembly + CASM Array Sum ==="

; Initialize array with raw assembly
mov rdi, [numbers_array]    ; Get array address
mov rcx, 0                  ; Counter

init_loop:
    mov [rdi + rcx*4], ecx  ; Store counter value in array
    inc rcx
    cmp rcx, [array_size]
    jl init_loop

%println "Array initialized with assembly"

; Calculate sum using mixed approach
mov rsi, 0                  ; Sum register
mov rcx, 0                  ; Counter
mov rdi, [numbers_array]    ; Array address

sum_loop:
    mov eax, [rdi + rcx*4]  ; Load array element
    add rsi, rax            ; Add to sum
    inc rcx
    
    ; Use CASM for loop control
    %if rcx < array_size
        jmp sum_loop
    %endif

; Store result in CASM variable using assembly
mov [sum], rsi

%println "Sum calculated using assembly:"
%println sum

; Direct register manipulation for final calculation
mov rax, [sum]
imul rax, 2                 ; Multiply by 2
mov rbx, 10
xor rdx, rdx
div rbx                     ; Divide by 10

%println "Processed result (sum * 2 / 10):"
; Raw assembly to print the result
push rax
%println rax
pop rax
```

## Architecture

CASM uses a clean 4-stage compilation pipeline with C integration support:

1. **Lexical Analysis** (`src/core/lexer.py`) - Tokenizes CASM source code
2. **C Code Processing** (`src/utils/c_processor.py`) - Handles `%extern` and `%!` blocks
3. **Parsing** (`src/core/parser.py`) - Builds Abstract Syntax Tree (AST)
4. **Code Generation** (`src/core/codegen.py`) - Converts AST to x86-64 assembly
5. **Assembly & Linking** (`src/compiler.py`) - NASM + MinGW-w64 toolchain

### C Integration Pipeline

When C code is detected (`%extern` or `%!` blocks):
1. **C Preprocessing**: Extract and process embedded C code
2. **Header Management**: Handle extern declarations automatically
3. **Variable Binding**: Make CASM variables available to C code
4. **Code Generation**: Generate both C and assembly object files
5. **Linking**: Combine all objects into final executable

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
│   ├── utils/
│   │   ├── c_processor.py # C code integration handler
│   │   ├── build.py     # Build utilities
│   │   ├── formatter.py # Assembly formatting
│   │   └── colors.py    # Terminal colors
│   ├── stdlib/
│   │   └── minimal.py   # Minimal standard library
│   └── compiler.py      # Main compiler class
├── tests/               # Test files
├── examples/            # Example programs (including C integration)
└── output/             # Generated files
```

## CASM C Assembly Integration

CASM supports powerful hybrid programming through its **C Assembly** integration system. This allows you to:

- Use standard C library functions alongside CASM constructs
- Write complex algorithms in C while using CASM for system-level control
- Share variables seamlessly between C and Assembly code
- Leverage existing C libraries and functions

### Key Integration Features

1. **`%extern` Directives**: Include C headers for library functions
2. **`%!` Code Blocks**: Embed C code directly in CASM programs
3. **Automatic Variable Binding**: CASM variables accessible in C code
4. **Mixed Compilation**: Single command compiles both C and Assembly
5. **Standard Library Access**: Full access to C standard library functions

### Usage Examples

All existing CASM commands (`compile`, `asm`) automatically handle C integration:

```bash
# Compile hybrid C+Assembly program
python3 casm.py compile my_hybrid_program.casm

# Generate assembly (shows both C and Assembly output)
python3 casm.py asm complex_algorithm.casm
```

The compiler automatically detects C code blocks and handles the complete build pipeline, including:
- C code compilation
- Assembly generation  
- Object file linking
- Executable creation

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
- **Focused Feature Set**: Only essential language constructs plus powerful C integration
- **Educational Value**: Clear, readable code for learning compiler design
- **Practical Use**: Generates working executables with full C library support
- **Hybrid Programming**: Best of both worlds - C's expressiveness and Assembly's control

## License

See LICENSE file for details.