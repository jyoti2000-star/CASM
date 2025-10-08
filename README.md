# CASM - Clean Assembly Language

CASM is a modern, clean assembly-style language that compiles to efficient x86-64 Windows assembly. It provides intuitive syntax for essential programming constructs while maintaining the power and control of assembly language.

## Features

- **Clean Syntax**: Modern, readable syntax with `@type` variable declarations
- **Control Flow**: `if/endif`, `while/endwhile`, `for/endfor` constructs
- **Variables**: Typed variable declarations with automatic memory management
- **I/O Operations**: Simple `print` and `scan` statements
- **C Integration**: Seamless C code blocks with `_c_` and `_endc_` delimiters
- **Raw Assembly**: Direct assembly code support alongside high-level constructs
- **Comments**: `;` for single-line comments
- **Hybrid Programming**: Mix high-level constructs with low-level assembly

## Quick Start

### Installation

Requirements:
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

## Language Syntax Reference

### Variable Declaration

Variables are declared with `@type name = value` syntax:

```casm
@int counter = 0
@int age = 25
@string message = "Hello World"
@float price = 99.99
```

### Variable Assignment

After declaration, variables can be assigned new values:

```casm
@int x = 10
x = 15
x = x + 5
```

### Input/Output Operations

#### Print Statement
```casm
print "Hello, World!"
print "The answer is:"
print 42
```

#### Scan Statement
```casm
@int number = 0
print "Enter a number:"
scan number
print "You entered:"
print number
```

### Control Flow

#### If Statements
```casm
@int age = 18

if age >= 18
    print "You are an adult"
endif

if age < 13
    print "You are a child"
else
    print "You are a teenager or adult"
endif
```

#### While Loops
```casm
@int counter = 0

while counter < 5
    print "Counter:"
    print counter
    counter = counter + 1
endwhile
```

#### For Loops
```casm
for i in range(10)
    print "Iteration:"
    print i
endfor
```

### Comments
```casm
; This is a single-line comment
@int x = 5  ; Comments can be at the end of lines

; Comments are ignored by the compiler
; They help document your code
```

### C Code Integration

CASM supports seamless integration with C code using `_c_` and `_endc_` delimiters:

```casm
@int number = 42

_c_
printf("Hello from C! Number: %d\n", number);

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int result = factorial(5);
printf("Factorial of 5 = %d\n", result);
_endc_

print "Back in CASM"
```

### Raw Assembly Integration

You can mix raw assembly code directly with CASM constructs:

```casm
@int my_var = 100

; Raw assembly alongside CASM
mov rax, 42
push rax

; Access CASM variables from assembly
mov eax, dword [rel var_my_var]
add eax, 50
mov dword [rel var_my_var], eax

if my_var > 120
    print "Variable was modified by assembly!"
endif
```

## Complete Example Programs

### Hello World
```casm
; hello.casm
print "Hello, World!"
```

### Interactive Calculator
```casm
; calculator.casm
@int num1 = 0
@int num2 = 0
@int result = 0

print "=== Simple Calculator ==="
print "Enter first number:"
scan num1

print "Enter second number:"
scan num2

result = num1 + num2

print "Sum:"
print result

result = num1 * num2
print "Product:"
print result
```

### Counting and Loops
```casm
; loops.casm
@int i = 0

print "=== While Loop Example ==="
while i < 5
    print "Count:"
    print i
    i = i + 1
endwhile

print "=== For Loop Example ==="
for j in range(3)
    print "For loop iteration:"
    print j
endfor

print "Done!"
```

### Conditional Logic
```casm
; conditions.casm
@int score = 85

print "=== Grade Calculator ==="
print "Score:"
print score

if score >= 90
    print "Grade: A"
else
    if score >= 80
        print "Grade: B"
    else
        if score >= 70
            print "Grade: C"
        else
            print "Grade: F"
        endif
    endif
endif
```

### C Integration Example
```casm
; hybrid.casm
@int radius = 5

print "=== Circle Calculator ==="

_c_
#include <stdio.h>
#include <math.h>

double pi = 3.14159;
double area = pi * radius * radius;
double circumference = 2 * pi * radius;

printf("Using C math functions:\n");
printf("Radius: %d\n", radius);
printf("Area: %.2f\n", area);
printf("Circumference: %.2f\n", circumference);

// Define a C function
int is_large_circle(double r) {
    return r > 10;
}

if (is_large_circle(radius)) {
    printf("This is a large circle!\n");
} else {
    printf("This is a small circle.\n");
}
_endc_

print "Back to CASM:"
if radius > 3
    print "CASM also thinks this circle is decent sized"
endif
```

### Assembly Integration Example
```casm
; assembly_mix.casm
@int array_size = 5
@int sum = 0

print "=== Mixed Assembly/CASM Example ==="

; Use raw assembly for performance-critical operations
mov ecx, 0              ; Initialize counter
mov eax, 0              ; Initialize sum

sum_loop:
    add eax, ecx        ; Add counter to sum
    inc ecx             ; Increment counter
    cmp ecx, dword [rel var_array_size]
    jl sum_loop         ; Continue if counter < array_size

; Store assembly result in CASM variable
mov dword [rel var_sum], eax

print "Sum calculated using assembly:"
print sum

; Continue with CASM logic
if sum > 10
    print "The sum is greater than 10"
    
    ; More assembly for complex operations
    mov eax, dword [rel var_sum]
    imul eax, 2         ; Multiply by 2
    mov dword [rel var_sum], eax
    
    print "Doubled sum:"
    print sum
endif
```

## Advanced Features

### Variable Types

CASM supports several data types:

```casm
@int whole_number = 42
@float decimal_number = 3.14
@string text = "Hello"
@char single_char = 'A'
```

### Complex Expressions

```casm
@int a = 10
@int b = 20
@int c = 0

c = a + b * 2
c = (a + b) / 2
c = a * a + b * b
```

### Nested Control Structures

```casm
@int i = 0
@int j = 0

while i < 3
    j = 0
    while j < 2
        print "Nested loop:"
        print i
        print j
        j = j + 1
    endwhile
    i = i + 1
endwhile
```

## Architecture

CASM uses a clean compilation pipeline:

1. **Lexical Analysis** - Tokenizes source code with clean syntax
2. **Parsing** - Builds Abstract Syntax Tree (AST) from tokens
3. **Code Generation** - Converts AST to x86-64 assembly
4. **C Integration** - Processes embedded C code blocks
5. **Assembly & Linking** - Uses NASM + MinGW-w64 toolchain

### Directory Structure
```
CASM/
├── casm.py              # Main entry point
├── src/
│   ├── core/
│   │   ├── tokens.py    # Token definitions
│   │   ├── lexer.py     # Lexical analyzer
│   │   ├── parser.py    # Parser
│   │   ├── ast_nodes.py # AST node definitions
│   │   └── codegen.py   # Code generator
│   ├── utils/
│   │   ├── c_processor.py # C code integration
│   │   ├── formatter.py # Assembly formatting
│   │   └── colors.py    # Terminal colors
│   └── compiler.py      # Main compiler class
├── examples/            # Example programs
├── test/               # Test files
└── output/             # Generated assembly files
```

## Syntax Comparison

### Old vs New Syntax

**Old Syntax:**
```casm
%var counter 0
%if counter < 10
    %println "Less than 10"
%endif
%while counter < 5
    %println counter
    %var counter counter+1
%endwhile
```

**New Clean Syntax:**
```casm
@int counter = 0
if counter < 10
    print "Less than 10"
endif
while counter < 5
    print counter
    counter = counter + 1
endwhile
```

## Target Platform

- **Architecture**: x86-64
- **Operating System**: Windows (PE32+ executables)
- **Calling Convention**: Windows x64
- **Assembler**: NASM (Intel syntax)
- **Linker**: MinGW-w64 GCC

Cross-compilation from macOS/Linux is supported via MinGW-w64 and Wine.

## Development

The codebase emphasizes:
- **Clean Architecture**: Clear separation of concerns
- **Modern Syntax**: Intuitive and readable language design
- **Educational Value**: Clean code for learning compiler design
- **Practical Use**: Generates working executables
- **Hybrid Programming**: Seamless integration of high-level and low-level code

## License

See LICENSE file for details.