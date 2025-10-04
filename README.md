# CASM - The C/Assembly Hybrid Compiler

Welcome to CASM, an advanced compiler designed to seamlessly blend the power and control of Assembly with the simplicity and readability of C-like high-level constructs. This document serves as a comprehensive guide to the CASM language syntax and the compilation process.

## Table of Contents

1.  [Overview](#1-overview)
2.  [Language Syntax](#2-language-syntax)
    - [File Structure and Sections](#file-structure-and-sections)
    - [Data Declarations](#data-declarations)
    - [High-Level Constructs (C-like)](#high-level-constructs-c-like)
      - [Variables](#variables)
      - [Procedures (Functions)](#procedures-functions)
      - [Control Flow](#control-flow)
      - [I/O Operations](#io-operations)
      - [Array Access](#array-access)
    - [Inline Assembly](#inline-assembly)
    - [Preprocessor Directives](#preprocessor-directives)
3.  [How to Compile](#3-how-to-compile)
    - [Basic Compilation](#basic-compilation)
    - [Command-Line Options](#command-line-options)
    - [Cross-Compilation](#cross-compilation)
4.  [Examples](#4-examples)
    - [Hello World](#hello-world)
    - [Looping and Conditionals](#looping-and-conditionals)
    - [Mixing Assembly and High-Level Code](#mixing-assembly-and-high-level-code)

---

## 1. Overview

CASM is a source-to-source compiler that translates a hybrid C/Assembly language into standard C code. This C code is then compiled into a native executable using a standard C compiler like GCC.

**Key Features:**

- **Hybrid Syntax:** Write low-level Assembly and high-level C-like code in the same file.
- **Familiarity:** Uses Intel syntax for Assembly and a simplified, C-inspired syntax for high-level constructs.
- **Preprocessor:** Supports `#define`, conditional compilation (`#ifdef`, etc.), and file includes.
- **Cross-Platform:** Can target different operating systems and architectures like Linux, Windows, macOS, x86_64, and ARM64.

## 2. Language Syntax

### File Structure and Sections

A CASM source file is organized into sections, similar to traditional assembly language.

- `section .data`: Used for declaring **initialized** global variables.
- `section .bss`: Used for declaring **uninitialized** global variables (reserving space).
- `section .text`: Contains the executable code, including the `main` entry point.

```
section .data
    my_message db "Hello", 0

section .bss
    my_buffer resb 100

section .text
global main

main:
    ; Your code here
    ret
```

### Data Declarations

You can declare data using traditional assembly directives.

| Directive | Description                | Example                            |
| :-------- | :------------------------- | :--------------------------------- |
| `db`      | Define Byte(s)             | `my_char db 'A'`, `msg db "Hi", 0` |
| `dd`      | Define Doubleword (32-bit) | `my_num dd 42`, `arr dd 1, 2, 3`   |
| `resb`    | Reserve Bytes              | `buffer resb 256`                  |
| `resd`    | Reserve Doublewords        | `value resd 1`                     |

### High-Level Constructs (C-like)

CASM provides several C-like keywords and structures to simplify development.

#### Variables

Declare local or global variables using the `var` keyword.

**Syntax:** `var <name>: <type> [= <initial_value>];`

```
; Global variable
var global_counter: int = 100;

main:
    ; Local variable
    var i: int
    var message: char* = "A local string"
    var prices: float[] = {1.99, 2.50, 10.0};
```

#### Procedures (Functions)

Define functions using `proc` and `endp`.

**Syntax:** `proc <name>[: <return_type> <param1>, <param2>, ...]`

```
; A simple procedure
proc print_hello
    print "Hello from a procedure!"
endp

; A function with parameters and a return value
proc add_numbers: int x, y
    var sum: int
    sum = x + y
    return sum
endp

main:
    call print_hello()
    var result: int
    result = add_numbers(10, 20)
    print "Result:", result
    ret
```

#### Control Flow

**`if / else if / else`**

Use C-style conditions. Braces `{}` are required.

```
var num: int = 10
if (num > 5) {
    print "Number is greater than 5"
} else if (num == 5) {
    print "Number is exactly 5"
} else {
    print "Number is less than 5"
}
```

**`for` Loop**

CASM supports a simplified `for` loop.

**Syntax:** `for <variable> = <start> to <end> { ... }`

```
var i: int
for i = 0 to 4 {
    print "Loop iteration:", i
}
```

#### I/O Operations

**`print`**

A versatile command for printing to the console. It works like `printf`.

```
print "Hello, World!"             ; Prints a simple string with a newline
print my_variable                 ; Prints the content of a variable
print "Count:", counter           ; Prints a label and a variable
print "Value is %d", my_num       ; Uses a format specifier
```

**`scanf`**

Read formatted input from the user.

```
var user_age: int
print "Enter your age: "
scanf "%d", user_age
```

#### Array Access

Access array elements using standard C-style bracket notation `[]`.

```
section .data
    my_array dd 10, 20, 30

section .text
main:
    print "First element:", my_array[0]  ; Prints 10
    my_array[1] = 99                     ; Modifies the second element
    print "Second element:", my_array[1] ; Prints 99
    ret
```

### Inline Assembly

You can write standard Intel-syntax x86 assembly directly within your code. The compiler will automatically convert it into GCC-style inline assembly.

```
var a: int = 10
var b: int = 20
var c: int

; Assembly block to add a and b
mov eax, [a]
add eax, [b]
mov [c], eax

print "The sum is:", c
```

### Preprocessor Directives

- `#define <NAME> <value>`: Defines a simple macro.
- `#define <NAME>(args) <body>`: Defines a function-like macro.
- `#ifdef <NAME>`, `#ifndef <NAME>`, `#else`, `#endif`: For conditional compilation.
- `.include "path/to/file.asm"`: Includes another source file.

## 3. How to Compile

The compiler is a Python script that translates your `.asm` file to a `.c` file and then invokes `gcc` to create an executable.

### Basic Compilation

The primary command is `c` (for compile).

```sh
python3 compiler.py c advanced_test.asm
```

This command will:

1.  Create `advanced_test.c`.
2.  Compile `advanced_test.c` into a native executable named `advanced_test`.

### Command-Line Options

| Option        | Description                                                   | Example                                     |
| :------------ | :------------------------------------------------------------ | :------------------------------------------ |
| `-o:<file>`   | Set the output executable name.                               | `python3 compiler.py c test.asm -o:my_app`  |
| `-d:release`  | Compiles with optimizations (`-O3`) and no debug info.        | `python3 compiler.py c test.asm -d:release` |
| `-d:debug`    | Compiles with debug info (`-g`) and no optimizations (`-O0`). | `python3 compiler.py c test.asm -d:debug`   |
| `-t:<target>` | Set the target platform (`linux`, `windows`, `macos`).        | `python3 compiler.py c test.asm -t:windows` |
| `-a:<arch>`   | Set the target architecture (`x86_64`, `arm64`).              | `python3 compiler.py c test.asm -a:arm64`   |

### Cross-Compilation

You can easily cross-compile for other platforms. For example, to compile for Windows from macOS or Linux, you need a MinGW toolchain installed.

```sh
python3 compiler.py c my_program.asm -t:windows -o:my_program.exe
```

## 4. Examples

### Hello World

```
; main.asm
section .data
    hello_msg db "Hello, World!", 0

section .text
    global main

main:
    print hello_msg
    ret
```

### Looping and Conditionals

```
; loops.asm
section .text
    global main

main:
    var k: int
    for k = 1 to 10 {
        if k % 2 == 0 {
            print k, " is even."
        } else {
            print k, " is odd."
        }
    }
    ret
```

### Mixing Assembly and High-Level Code

```
; mix.asm
section .data
    num1 dd 100
    num2 dd 50

section .text
    global main

main:
    var result: int

    ; Use assembly for subtraction
    mov eax, [num1]
    sub eax, [num2]
    mov [result], eax

    ; Use high-level print
    print "100 - 50 =", result
    ret
```
