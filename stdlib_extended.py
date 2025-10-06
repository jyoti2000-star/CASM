#!/usr/bin/env python3

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import os
import json
from pathlib import Path

@dataclass
class StdLibFunction:
    """Represents a standard library function"""
    name: str
    description: str
    parameters: List[str] = field(default_factory=list)
    return_type: str = "void"
    assembly_code: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    platform_specific: Dict[str, List[str]] = field(default_factory=dict)
    example_usage: str = ""
    complexity: str = "O(1)"  # Time complexity
    category: str = "utility"

class StandardLibrary:
    """Extended standard library for HLASM"""
    
    def __init__(self):
        self.functions: Dict[str, StdLibFunction] = {}
        self.categories: Dict[str, List[str]] = {}
        
        # Initialize with built-in functions
        self._register_string_functions()
        self._register_math_functions()
        self._register_memory_functions()
        self._register_io_functions()
        self._register_conversion_functions()
        self._register_system_functions()
        self._register_crypto_functions()
        self._register_data_structure_functions()
    
    def _register_string_functions(self):
        """Register string manipulation functions"""
        
        # Enhanced strlen with null pointer check
        self.functions["strlen_safe"] = StdLibFunction(
            name="strlen_safe",
            description="Calculate string length with null pointer check",
            parameters=["str_ptr", "result_ptr"],
            return_type="int",
            assembly_code=[
                "__strlen_safe:",
                "    push rbp",
                "    mov rbp, rsp",
                "    test rdi, rdi        ; Check for null pointer",
                "    jz .null_ptr",
                "    xor rax, rax",
                ".loop:",
                "    cmp byte [rdi + rax], 0",
                "    je .done",
                "    inc rax",
                "    jmp .loop",
                ".done:",
                "    pop rbp",
                "    ret",
                ".null_ptr:",
                "    mov rax, -1          ; Return -1 for null pointer",
                "    pop rbp",
                "    ret"
            ],
            category="string",
            example_usage="%strlen_safe string_var, length_var",
            complexity="O(n)"
        )
        
        # String reverse
        self.functions["strrev"] = StdLibFunction(
            name="strrev",
            description="Reverse a string in place",
            parameters=["str_ptr"],
            assembly_code=[
                "__strrev:",
                "    push rbp",
                "    mov rbp, rsp",
                "    push rbx",
                "    push rcx",
                "    push rdx",
                "    ",
                "    ; Find string length",
                "    mov rbx, rdi         ; Save string pointer",
                "    call __strlen",
                "    mov rcx, rax         ; String length",
                "    ",
                "    ; Setup for reversal",
                "    mov rsi, rbx         ; Start pointer",
                "    lea rdi, [rbx + rcx - 1]  ; End pointer",
                "    ",
                ".reverse_loop:",
                "    cmp rsi, rdi",
                "    jge .done",
                "    ",
                "    ; Swap characters",
                "    mov al, byte [rsi]",
                "    mov dl, byte [rdi]",
                "    mov byte [rsi], dl",
                "    mov byte [rdi], al",
                "    ",
                "    inc rsi",
                "    dec rdi",
                "    jmp .reverse_loop",
                "    ",
                ".done:",
                "    pop rdx",
                "    pop rcx",
                "    pop rbx",
                "    pop rbp",
                "    ret"
            ],
            category="string",
            example_usage="%strrev my_string",
            complexity="O(n)"
        )
        
        # String to uppercase
        self.functions["strupper"] = StdLibFunction(
            name="strupper",
            description="Convert string to uppercase",
            parameters=["str_ptr"],
            assembly_code=[
                "__strupper:",
                "    push rbp",
                "    mov rbp, rsp",
                "    xor rcx, rcx",
                ".loop:",
                "    mov al, byte [rdi + rcx]",
                "    test al, al",
                "    jz .done",
                "    cmp al, 'a'",
                "    jb .next",
                "    cmp al, 'z'",
                "    ja .next",
                "    sub al, 32           ; Convert to uppercase",
                "    mov byte [rdi + rcx], al",
                ".next:",
                "    inc rcx",
                "    jmp .loop",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="string",
            example_usage="%strupper my_string",
            complexity="O(n)"
        )
    
    def _register_math_functions(self):
        """Register mathematical functions"""
        
        # Power function (integer exponentiation)
        self.functions["pow"] = StdLibFunction(
            name="pow",
            description="Calculate base^exponent (integer)",
            parameters=["base", "exponent", "result_ptr"],
            assembly_code=[
                "__pow:",
                "    push rbp",
                "    mov rbp, rsp",
                "    push rbx",
                "    ",
                "    mov rax, 1           ; Result = 1",
                "    mov rbx, rdi         ; Base",
                "    mov rcx, rsi         ; Exponent",
                "    ",
                "    test rcx, rcx",
                "    jz .done             ; 0^0 = 1",
                "    ",
                ".power_loop:",
                "    test rcx, 1          ; Check if exponent is odd",
                "    jz .even",
                "    imul rax, rbx        ; result *= base",
                ".even:",
                "    imul rbx, rbx        ; base *= base",
                "    shr rcx, 1           ; exponent /= 2",
                "    jnz .power_loop",
                "    ",
                ".done:",
                "    pop rbx",
                "    pop rbp",
                "    ret"
            ],
            category="math",
            example_usage="%pow 2, 8, result",
            complexity="O(log n)"
        )
        
        # Factorial
        self.functions["factorial"] = StdLibFunction(
            name="factorial",
            description="Calculate factorial of a number",
            parameters=["n", "result_ptr"],
            assembly_code=[
                "__factorial:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    cmp rdi, 1",
                "    jle .base_case",
                "    ",
                "    push rdi",
                "    dec rdi",
                "    call __factorial",
                "    pop rdi",
                "    imul rax, rdi",
                "    jmp .done",
                "    ",
                ".base_case:",
                "    mov rax, 1",
                "    ",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="math",
            example_usage="%factorial 5, result",
            complexity="O(n)"
        )
        
        # GCD (Greatest Common Divisor)
        self.functions["gcd"] = StdLibFunction(
            name="gcd",
            description="Calculate greatest common divisor",
            parameters=["a", "b", "result_ptr"],
            assembly_code=[
                "__gcd:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    ; Euclidean algorithm",
                "    mov rax, rdi         ; a",
                "    mov rbx, rsi         ; b",
                "    ",
                ".gcd_loop:",
                "    test rbx, rbx",
                "    jz .done",
                "    ",
                "    xor rdx, rdx",
                "    div rbx              ; rax = a / b, rdx = a % b",
                "    mov rax, rbx         ; a = b",
                "    mov rbx, rdx         ; b = a % b",
                "    jmp .gcd_loop",
                "    ",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="math",
            example_usage="%gcd 48, 18, result",
            complexity="O(log min(a,b))"
        )
    
    def _register_memory_functions(self):
        """Register memory management functions"""
        
        # Safe memory copy with bounds checking
        self.functions["memcpy_safe"] = StdLibFunction(
            name="memcpy_safe",
            description="Safe memory copy with bounds checking",
            parameters=["dest", "src", "size", "dest_size"],
            assembly_code=[
                "__memcpy_safe:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    ; Check if size > dest_size",
                "    cmp rdx, rcx",
                "    ja .error",
                "    ",
                "    ; Check for null pointers",
                "    test rdi, rdi",
                "    jz .error",
                "    test rsi, rsi", 
                "    jz .error",
                "    ",
                "    ; Perform copy",
                "    xor rax, rax",
                ".copy_loop:",
                "    cmp rax, rdx",
                "    jge .success",
                "    mov bl, byte [rsi + rax]",
                "    mov byte [rdi + rax], bl",
                "    inc rax",
                "    jmp .copy_loop",
                "    ",
                ".success:",
                "    mov rax, 0           ; Return 0 for success",
                "    jmp .done",
                "    ",
                ".error:",
                "    mov rax, -1          ; Return -1 for error",
                "    ",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="memory",
            example_usage="%memcpy_safe dest, src, 100, 256",
            complexity="O(n)"
        )
        
        # Memory comparison
        self.functions["memcmp"] = StdLibFunction(
            name="memcmp",
            description="Compare two memory regions",
            parameters=["ptr1", "ptr2", "size"],
            assembly_code=[
                "__memcmp:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    xor rax, rax",
                "    xor rcx, rcx",
                "    ",
                ".compare_loop:",
                "    cmp rcx, rdx",
                "    jge .equal",
                "    ",
                "    mov al, byte [rdi + rcx]",
                "    mov bl, byte [rsi + rcx]",
                "    cmp al, bl",
                "    jne .not_equal",
                "    ",
                "    inc rcx",
                "    jmp .compare_loop",
                "    ",
                ".equal:",
                "    xor rax, rax",
                "    jmp .done",
                "    ",
                ".not_equal:",
                "    sub rax, rbx         ; Return difference",
                "    ",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="memory",
            example_usage="%memcmp ptr1, ptr2, 16",
            complexity="O(n)"
        )
    
    def _register_io_functions(self):
        """Register I/O functions"""
        
        # Print integer
        self.functions["print_int"] = StdLibFunction(
            name="print_int",
            description="Print an integer to stdout",
            parameters=["number"],
            assembly_code=[
                "__print_int:",
                "    push rbp",
                "    mov rbp, rsp",
                "    sub rsp, 32          ; Buffer for digits",
                "    ",
                "    mov rax, rdi         ; Number to print",
                "    mov rbx, 10          ; Base 10",
                "    lea rsi, [rbp - 1]   ; End of buffer",
                "    mov byte [rsi], 0    ; Null terminator",
                "    ",
                "    ; Handle negative numbers",
                "    test rax, rax",
                "    jns .positive",
                "    neg rax",
                "    mov r8, 1            ; Negative flag",
                "    jmp .convert",
                "    ",
                ".positive:",
                "    xor r8, r8           ; Positive flag",
                "    ",
                ".convert:",
                "    dec rsi",
                "    xor rdx, rdx",
                "    div rbx",
                "    add dl, '0'",
                "    mov byte [rsi], dl",
                "    test rax, rax",
                "    jnz .convert",
                "    ",
                "    ; Add minus sign if negative",
                "    test r8, r8",
                "    jz .print",
                "    dec rsi",
                "    mov byte [rsi], '-'",
                "    ",
                ".print:",
                "    ; Calculate length and print",
                "    lea rdi, [rbp - 1]",
                "    sub rdi, rsi",
                "    mov rdx, rdi         ; Length",
                "    mov rdi, 1           ; stdout",
                "    mov rax, 1           ; sys_write",
                "    syscall",
                "    ",
                "    add rsp, 32",
                "    pop rbp",
                "    ret"
            ],
            category="io",
            example_usage="%print_int 12345",
            complexity="O(log n)"
        )
        
        # Read line from stdin
        self.functions["read_line"] = StdLibFunction(
            name="read_line",
            description="Read a line from stdin",
            parameters=["buffer", "max_size"],
            assembly_code=[
                "__read_line:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    mov rax, 0           ; sys_read",
                "    mov rdi, 0           ; stdin",
                "    ; rsi already contains buffer",
                "    mov rdx, rsi         ; max_size",
                "    syscall",
                "    ",
                "    ; Remove newline if present",
                "    test rax, rax",
                "    jle .done",
                "    dec rax",
                "    cmp byte [rsi + rax], 10  ; newline",
                "    jne .done",
                "    mov byte [rsi + rax], 0   ; replace with null",
                "    ",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="io",
            example_usage="%read_line buffer, 256",
            complexity="O(n)"
        )
    
    def _register_conversion_functions(self):
        """Register conversion functions"""
        
        # Enhanced atoi with error checking
        self.functions["atoi_safe"] = StdLibFunction(
            name="atoi_safe",
            description="Convert string to integer with error checking",
            parameters=["str_ptr", "result_ptr", "error_ptr"],
            assembly_code=[
                "__atoi_safe:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    ; Initialize",
                "    xor rax, rax         ; Result",
                "    xor rcx, rcx         ; Index",
                "    mov r8, 1            ; Sign (1 = positive, -1 = negative)",
                "    mov qword [rdx], 0   ; Clear error flag",
                "    ",
                "    ; Check for null pointer",
                "    test rdi, rdi",
                "    jz .error",
                "    ",
                "    ; Skip whitespace",
                ".skip_whitespace:",
                "    mov bl, byte [rdi + rcx]",
                "    cmp bl, ' '",
                "    je .next_char",
                "    cmp bl, 9            ; tab",
                "    je .next_char",
                "    jmp .check_sign",
                ".next_char:",
                "    inc rcx",
                "    jmp .skip_whitespace",
                "    ",
                "    ; Check for sign",
                ".check_sign:",
                "    mov bl, byte [rdi + rcx]",
                "    cmp bl, '-'",
                "    jne .check_plus",
                "    mov r8, -1",
                "    inc rcx",
                "    jmp .parse_digits",
                ".check_plus:",
                "    cmp bl, '+'",
                "    jne .parse_digits",
                "    inc rcx",
                "    ",
                "    ; Parse digits",
                ".parse_digits:",
                "    mov bl, byte [rdi + rcx]",
                "    test bl, bl",
                "    jz .done",
                "    ",
                "    ; Check if digit",
                "    cmp bl, '0'",
                "    jb .error",
                "    cmp bl, '9'",
                "    ja .error",
                "    ",
                "    ; Convert digit and add to result",
                "    sub bl, '0'",
                "    imul rax, 10",
                "    add rax, rbx",
                "    ",
                "    ; Check for overflow (simplified)",
                "    cmp rax, 0x7FFFFFFFFFFFFFFF",
                "    ja .error",
                "    ",
                "    inc rcx",
                "    jmp .parse_digits",
                "    ",
                ".done:",
                "    imul rax, r8         ; Apply sign",
                "    mov qword [rsi], rax ; Store result",
                "    jmp .exit",
                "    ",
                ".error:",
                "    mov qword [rdx], 1   ; Set error flag",
                "    xor rax, rax",
                "    ",
                ".exit:",
                "    pop rbp",
                "    ret"
            ],
            category="conversion",
            example_usage="%atoi_safe str_ptr, result, error_flag",
            complexity="O(n)"
        )
        
        # Convert integer to hex string
        self.functions["itohex"] = StdLibFunction(
            name="itohex",
            description="Convert integer to hexadecimal string",
            parameters=["number", "buffer"],
            assembly_code=[
                "__itohex:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    mov rax, rdi         ; Number",
                "    mov rbx, 16          ; Base 16",
                "    lea rcx, [rsi + 15]  ; End of buffer",
                "    mov byte [rcx + 1], 0 ; Null terminator",
                "    ",
                ".convert_loop:",
                "    xor rdx, rdx",
                "    div rbx",
                "    ",
                "    ; Convert digit to hex char",
                "    cmp dl, 10",
                "    jb .numeric",
                "    add dl, 'A' - 10",
                "    jmp .store",
                ".numeric:",
                "    add dl, '0'",
                ".store:",
                "    mov byte [rcx], dl",
                "    dec rcx",
                "    ",
                "    test rax, rax",
                "    jnz .convert_loop",
                "    ",
                "    ; Move result to start of buffer",
                "    inc rcx",
                "    mov rdi, rsi",
                "    mov rsi, rcx",
                "    call __strcpy",
                "    ",
                "    pop rbp",
                "    ret"
            ],
            category="conversion",
            example_usage="%itohex 255, hex_buffer",
            complexity="O(log n)"
        )
    
    def _register_system_functions(self):
        """Register system utility functions"""
        
        # Get current time (Unix timestamp)
        self.functions["get_time"] = StdLibFunction(
            name="get_time",
            description="Get current Unix timestamp",
            parameters=["time_ptr"],
            assembly_code=[
                "__get_time:",
                "    push rbp",
                "    mov rbp, rsp",
                "    sub rsp, 16          ; timespec structure",
                "    ",
                "    mov rax, 228         ; sys_clock_gettime",
                "    mov rdi, 0           ; CLOCK_REALTIME",
                "    lea rsi, [rbp - 16]  ; timespec pointer",
                "    syscall",
                "    ",
                "    ; Return seconds in rax",
                "    mov rax, qword [rbp - 16]",
                "    ",
                "    add rsp, 16",
                "    pop rbp",
                "    ret"
            ],
            platform_specific={
                "windows": [
                    "    ; Windows specific time implementation",
                ],
                "darwin": [
                    "    mov rax, 0x2000116   ; sys_gettimeofday (macOS)",
                ]
            },
            category="system",
            example_usage="%get_time current_time",
            complexity="O(1)"
        )
        
        # Sleep for specified milliseconds
        self.functions["sleep_ms"] = StdLibFunction(
            name="sleep_ms",
            description="Sleep for specified milliseconds",
            parameters=["milliseconds"],
            assembly_code=[
                "__sleep_ms:",
                "    push rbp",
                "    mov rbp, rsp",
                "    sub rsp, 16          ; timespec structure",
                "    ",
                "    ; Convert ms to seconds and nanoseconds",
                "    mov rax, rdi         ; milliseconds",
                "    mov rbx, 1000",
                "    xor rdx, rdx",
                "    div rbx              ; rax = seconds, rdx = remaining ms",
                "    ",
                "    mov qword [rbp - 16], rax  ; tv_sec",
                "    imul rdx, 1000000    ; convert ms to ns",
                "    mov qword [rbp - 8], rdx   ; tv_nsec",
                "    ",
                "    mov rax, 35          ; sys_nanosleep",
                "    lea rdi, [rbp - 16]  ; timespec pointer",
                "    xor rsi, rsi         ; remaining time (NULL)",
                "    syscall",
                "    ",
                "    add rsp, 16",
                "    pop rbp",
                "    ret"
            ],
            category="system",
            example_usage="%sleep_ms 1000",
            complexity="O(1)"
        )
    
    def _register_crypto_functions(self):
        """Register basic cryptographic functions"""
        
        # Simple XOR cipher
        self.functions["xor_encrypt"] = StdLibFunction(
            name="xor_encrypt",
            description="XOR encryption/decryption",
            parameters=["data", "key", "data_len", "key_len"],
            assembly_code=[
                "__xor_encrypt:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    xor rax, rax         ; data index",
                "    xor rbx, rbx         ; key index",
                "    ",
                ".crypt_loop:",
                "    cmp rax, rdx         ; check if done",
                "    jge .done",
                "    ",
                "    ; Get data and key bytes",
                "    mov r8b, byte [rdi + rax]",
                "    mov r9b, byte [rsi + rbx]",
                "    ",
                "    ; XOR and store",
                "    xor r8b, r9b",
                "    mov byte [rdi + rax], r8b",
                "    ",
                "    ; Advance indices",
                "    inc rax",
                "    inc rbx",
                "    cmp rbx, rcx         ; wrap key index",
                "    jl .crypt_loop",
                "    xor rbx, rbx",
                "    jmp .crypt_loop",
                "    ",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="crypto",
            example_usage="%xor_encrypt data, key, 128, 16",
            complexity="O(n)"
        )
        
        # Simple hash function (djb2)
        self.functions["hash_djb2"] = StdLibFunction(
            name="hash_djb2",
            description="DJB2 hash function",
            parameters=["data", "length"],
            assembly_code=[
                "__hash_djb2:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    mov rax, 5381        ; hash = 5381",
                "    xor rcx, rcx         ; i = 0",
                "    ",
                ".hash_loop:",
                "    cmp rcx, rsi",
                "    jge .done",
                "    ",
                "    ; hash = ((hash << 5) + hash) + c",
                "    shl rax, 5",
                "    add rax, rax",
                "    movzx rbx, byte [rdi + rcx]",
                "    add rax, rbx",
                "    ",
                "    inc rcx",
                "    jmp .hash_loop",
                "    ",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="crypto",
            example_usage="%hash_djb2 data, 256",
            complexity="O(n)"
        )
    
    def _register_data_structure_functions(self):
        """Register data structure functions"""
        
        # Array find
        self.functions["array_find"] = StdLibFunction(
            name="array_find",
            description="Find element in array",
            parameters=["array", "size", "element", "result_index"],
            assembly_code=[
                "__array_find:",
                "    push rbp",
                "    mov rbp, rsp",
                "    ",
                "    xor rax, rax         ; index = 0",
                "    ",
                ".search_loop:",
                "    cmp rax, rsi         ; check bounds",
                "    jge .not_found",
                "    ",
                "    cmp qword [rdi + rax*8], rdx",
                "    je .found",
                "    ",
                "    inc rax",
                "    jmp .search_loop",
                "    ",
                ".found:",
                "    mov qword [rcx], rax ; store index",
                "    mov rax, 0           ; return success",
                "    jmp .done",
                "    ",
                ".not_found:",
                "    mov qword [rcx], -1  ; store -1",
                "    mov rax, -1          ; return not found",
                "    ",
                ".done:",
                "    pop rbp",
                "    ret"
            ],
            category="data_structures",
            example_usage="%array_find my_array, 10, target, index",
            complexity="O(n)"
        )
        
        # Array sort (bubble sort - simple implementation)
        self.functions["array_sort"] = StdLibFunction(
            name="array_sort",
            description="Sort array using bubble sort",
            parameters=["array", "size"],
            assembly_code=[
                "__array_sort:",
                "    push rbp",
                "    mov rbp, rsp",
                "    push rbx",
                "    push rcx",
                "    push rdx",
                "    ",
                "    dec rsi              ; size - 1",
                "    ",
                ".outer_loop:",
                "    test rsi, rsi",
                "    jle .done",
                "    ",
                "    xor rcx, rcx         ; i = 0",
                "    ",
                ".inner_loop:",
                "    cmp rcx, rsi",
                "    jge .outer_next",
                "    ",
                "    ; Compare adjacent elements",
                "    mov rax, qword [rdi + rcx*8]",
                "    mov rbx, qword [rdi + rcx*8 + 8]",
                "    cmp rax, rbx",
                "    jle .no_swap",
                "    ",
                "    ; Swap elements",
                "    mov qword [rdi + rcx*8], rbx",
                "    mov qword [rdi + rcx*8 + 8], rax",
                "    ",
                ".no_swap:",
                "    inc rcx",
                "    jmp .inner_loop",
                "    ",
                ".outer_next:",
                "    dec rsi",
                "    jmp .outer_loop",
                "    ",
                ".done:",
                "    pop rdx",
                "    pop rcx",
                "    pop rbx",
                "    pop rbp",
                "    ret"
            ],
            category="data_structures",
            example_usage="%array_sort my_array, 10",
            complexity="O(n²)"
        )
    
    def get_function(self, name: str) -> Optional[StdLibFunction]:
        """Get a standard library function by name"""
        return self.functions.get(name)
    
    def get_functions_by_category(self, category: str) -> List[StdLibFunction]:
        """Get all functions in a category"""
        return [func for func in self.functions.values() if func.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        categories = set(func.category for func in self.functions.values())
        return sorted(list(categories))
    
    def generate_assembly_code(self, function_name: str, platform: str = "windows") -> List[str]:
        """Generate platform-specific assembly code for a function"""
        func = self.get_function(function_name)
        if not func:
            return []
        
        # Use platform-specific code if available
        if platform in func.platform_specific:
            return func.platform_specific[platform]
        
        return func.assembly_code
    
    def generate_library_documentation(self) -> str:
        """Generate comprehensive library documentation"""
        doc = []
        doc.append("=" * 80)
        doc.append("HLASM EXTENDED STANDARD LIBRARY DOCUMENTATION")
        doc.append("=" * 80)
        doc.append("")
        
        categories = self.get_all_categories()
        
        for category in categories:
            functions = self.get_functions_by_category(category)
            if not functions:
                continue
            
            doc.append(f"{category.upper()} FUNCTIONS")
            doc.append("=" * len(f"{category.upper()} FUNCTIONS"))
            doc.append("")
            
            for func in sorted(functions, key=lambda f: f.name):
                doc.append(f"Function: {func.name}")
                doc.append(f"Description: {func.description}")
                doc.append(f"Parameters: {', '.join(func.parameters)}")
                doc.append(f"Returns: {func.return_type}")
                doc.append(f"Complexity: {func.complexity}")
                if func.example_usage:
                    doc.append(f"Example: {func.example_usage}")
                if func.dependencies:
                    doc.append(f"Dependencies: {', '.join(func.dependencies)}")
                doc.append("")
        
        doc.append("=" * 80)
        return "\n".join(doc)
    
    def export_library_json(self, filename: str):
        """Export library definitions to JSON"""
        library_data = {}
        
        for name, func in self.functions.items():
            library_data[name] = {
                'name': func.name,
                'description': func.description,
                'parameters': func.parameters,
                'return_type': func.return_type,
                'assembly_code': func.assembly_code,
                'dependencies': func.dependencies,
                'platform_specific': func.platform_specific,
                'example_usage': func.example_usage,
                'complexity': func.complexity,
                'category': func.category
            }
        
        with open(filename, 'w') as f:
            json.dump(library_data, f, indent=2)
    
    def import_library_json(self, filename: str):
        """Import library definitions from JSON"""
        try:
            with open(filename, 'r') as f:
                library_data = json.load(f)
            
            for name, func_data in library_data.items():
                self.functions[name] = StdLibFunction(
                    name=func_data['name'],
                    description=func_data['description'],
                    parameters=func_data['parameters'],
                    return_type=func_data['return_type'],
                    assembly_code=func_data['assembly_code'],
                    dependencies=func_data['dependencies'],
                    platform_specific=func_data['platform_specific'],
                    example_usage=func_data['example_usage'],
                    complexity=func_data['complexity'],
                    category=func_data['category']
                )
            
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False

# Global standard library instance
global_stdlib = StandardLibrary()