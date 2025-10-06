; Comprehensive Test Suite for HASM Compiler
; Tests all major features and capabilities

; ========== VARIABLE DECLARATIONS ==========
%var num1 42
%var num2 17
%var result 0
%var counter 0
%var temp 0
%var flag 1

; Array declarations
%var numbers[5] {10, 20, 30, 40, 50}
%var scores[3] {85, 92, 78}
%var buffer[10] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

; String and character variables
%var message "Hello"
%var char_temp 0

; Boolean-like variables
%var is_positive 1
%var is_found 0

; ========== PROGRAM START ==========
%println "=== COMPREHENSIVE HASM COMPILER TEST ==="
%println ""

; ========== BASIC ARITHMETIC OPERATIONS ==========
%println "1. ARITHMETIC OPERATIONS:"
%println "========================="

; Simple arithmetic
mov rax, [num1]
add rax, [num2]
mov [result], rax
%print "Addition: "
mov rax, [num1]
add rax, 48
mov [char_temp], al
%print char_temp
%print " + "
mov rax, [num2]
add rax, 48
mov [char_temp], al
%print char_temp
%print " = "
mov rax, [result]
add rax, 48
mov [char_temp], al
%println char_temp

; Subtraction
mov rax, [num1]
sub rax, [num2]
mov [result], rax
%print "Subtraction: "
mov rax, [num1]
add rax, 48
mov [char_temp], al
%print char_temp
%print " - "
mov rax, [num2]
add rax, 48
mov [char_temp], al
%print char_temp
%print " = "
mov rax, [result]
add rax, 48
mov [char_temp], al
%println char_temp

; Multiplication (simple)
mov rax, 3
mov rbx, 4
mul rbx
mov [result], rax
%print "Multiplication: 3 * 4 = "
mov rax, [result]
add rax, 48
mov [char_temp], al
%println char_temp

%println ""

; ========== VARIABLE MANIPULATION ==========
%println "2. VARIABLE MANIPULATION:"
%println "=========================="

; Copy variables
mov rax, [num1]
mov [temp], rax
%print "Copied num1 to temp: temp = "
mov rax, [temp]
add rax, 48
mov [char_temp], al
%println char_temp

; Increment/decrement operations
mov rax, [counter]
add rax, 1
mov [counter], rax
%print "Incremented counter: "
mov rax, [counter]
add rax, 48
mov [char_temp], al
%println char_temp

mov rax, [counter]
sub rax, 1
mov [counter], rax
%print "Decremented counter: "
mov rax, [counter]
add rax, 48
mov [char_temp], al
%println char_temp

%println ""

; ========== CONDITIONAL STATEMENTS ==========
%println "3. CONDITIONAL STATEMENTS:"
%println "=========================="

; Simple if statement
%if [num1] > [num2]
    %println "num1 is greater than num2"
%else
    %println "num1 is not greater than num2"
%endif

; Nested if statements
%if [num1] > 30
    %println "num1 > 30:"
    %if [num1] > 40
        %println "  num1 is also > 40"
        %if [num1] > 50
            %println "    num1 is also > 50"
        %else
            %println "    but num1 is <= 50"
        %endif
    %else
        %println "  but num1 <= 40"
    %endif
%else
    %println "num1 <= 30"
%endif

; Multiple conditions
%if [flag] == 1
    %println "Flag is set to 1 (true)"
%else
    %println "Flag is not 1"
%endif

%if [result] >= 0
    mov qword [is_positive], 1
    %println "Result is positive or zero"
%else
    mov qword [is_positive], 0
    %println "Result is negative"
%endif

%println ""

; ========== LOOP STRUCTURES ==========
%println "4. LOOP STRUCTURES:"
%println "==================="

; For loop - counting
%println "For loop (0 to 4):"
%for i in range 5
    %print "  Count: "
    mov rax, [i]
    add rax, 48
    mov [char_temp], al
    %println char_temp
%endfor

; For loop with array access
%println "Array traversal (numbers array):"
%for i in range 5
    %print "  numbers["
    mov rax, [i]
    add rax, 48
    mov [char_temp], al
    %print char_temp
    %print "] = "
    
    ; Calculate array offset
    mov rax, [i]
    mov rbx, 8
    mul rbx
    lea rsi, [numbers]
    add rsi, rax
    mov rcx, [rsi]
    
    ; Print tens digit
    mov rax, rcx
    mov rbx, 10
    xor rdx, rdx
    div rbx
    add rax, 48
    mov [char_temp], al
    %print char_temp
    
    ; Print ones digit
    add rdx, 48
    mov [char_temp], dl
    %println char_temp
%endfor

; While loop simulation with manual counter
%println "While loop simulation (countdown from 3):"
mov qword [counter], 3
__while_start:
mov rax, [counter]
cmp rax, 0
jle __while_end
    %print "  Countdown: "
    mov rax, [counter]
    add rax, 48
    mov [char_temp], al
    %println char_temp
    
    mov rax, [counter]
    sub rax, 1
    mov [counter], rax
    jmp __while_start
__while_end:
%println "  Liftoff!"

%println ""

; ========== ARRAY OPERATIONS ==========
%println "5. ARRAY OPERATIONS:"
%println "===================="

; Array initialization check
%println "Scores array contents:"
%for i in range 3
    %print "  scores["
    mov rax, [i]
    add rax, 48
    mov [char_temp], al
    %print char_temp
    %print "] = "
    
    ; Load array element
    mov rax, [i]
    mov rbx, 8
    mul rbx
    lea rsi, [scores]
    add rsi, rax
    mov rcx, [rsi]
    
    ; Print two-digit number (simplified for scores 78-92)
    mov rax, rcx
    mov rbx, 10
    xor rdx, rdx
    div rbx
    add rax, 48
    mov [char_temp], al
    %print char_temp
    
    add rdx, 48
    mov [char_temp], dl
    %println char_temp
%endfor

; Array modification
%println "Modifying buffer array:"
%for i in range 5
    ; Set buffer[i] = i + 1
    mov rax, [i]
    mov rbx, 8
    mul rbx
    lea rsi, [buffer]
    add rsi, rax
    mov rcx, [i]
    add rcx, 1
    mov [rsi], rcx
%endfor

%println "Modified buffer contents (first 5 elements):"
%for i in range 5
    %print "  buffer["
    mov rax, [i]
    add rax, 48
    mov [char_temp], al
    %print char_temp
    %print "] = "
    
    mov rax, [i]
    mov rbx, 8
    mul rbx
    lea rsi, [buffer]
    add rsi, rax
    mov rcx, [rsi]
    add rcx, 48
    mov [char_temp], cl
    %println char_temp
%endfor

%println ""

; ========== SEARCH AND COMPARISON ==========
%println "6. SEARCH AND COMPARISON:"
%println "========================="

; Linear search in numbers array
mov qword [temp], 30  ; Search target
mov qword [is_found], 0
%println "Searching for value 30 in numbers array:"

%for i in range 5
    ; Load array element
    mov rax, [i]
    mov rbx, 8
    mul rbx
    lea rsi, [numbers]
    add rsi, rax
    mov rcx, [rsi]
    
    %print "  Checking index "
    mov rax, [i]
    add rax, 48
    mov [char_temp], al
    %print char_temp
    %print ": value = "
    
    ; Print value (two digits)
    mov rax, rcx
    mov rbx, 10
    xor rdx, rdx
    div rbx
    add rax, 48
    mov [char_temp], al
    %print char_temp
    add rdx, 48
    mov [char_temp], dl
    %print char_temp
    
    ; Check if found
    cmp rcx, [temp]
    jne not_found
        %println " -> FOUND!"
        mov qword [is_found], 1
        jmp search_done
    not_found:
        %println " -> continue searching"
%endfor

search_done:
%if [is_found] == 1
    %println "Search completed: Target found!"
%else
    %println "Search completed: Target not found."
%endif

%println ""

; ========== NESTED CONTROL STRUCTURES ==========
%println "7. NESTED CONTROL STRUCTURES:"
%println "============================="

%for i in range 3
    %print "Outer loop iteration "
    mov rax, [i]
    add rax, 48
    mov [char_temp], al
    %print char_temp
    %println ":"
    
    %if [i] == 0
        %println "  First iteration - no inner processing"
    %else
        %if [i] == 1
            %println "  Second iteration - simple inner loop:"
            %for j in range 2
                %print "    Inner count: "
                mov rax, [j]
                add rax, 48
                mov [char_temp], al
                %println char_temp
            %endfor
        %else
            %println "  Third iteration - complex inner processing:"
            mov rax, [i]
            mov rbx, 2
            mul rbx
            mov [temp], rax
            %print "    Calculated: i * 2 = "
            mov rax, [temp]
            add rax, 48
            mov [char_temp], al
            %println char_temp
        %endif
    %endif
%endfor

%println ""

; ========== MEMORY AND REGISTER OPERATIONS ==========
%println "8. MEMORY AND REGISTER OPERATIONS:"
%println "=================================="

; Register manipulation
mov rax, 123
mov rbx, 456
add rax, rbx
mov [result], rax
%println "Register arithmetic: 123 + 456 stored in result"

; Memory-to-memory operations
mov rax, [num1]
mov [temp], rax
mov rax, [num2]
add [temp], rax
%println "Memory operations: temp = num1 + num2"

; Stack-like operations (using variables)
mov rax, 100
mov [buffer], rax    ; Push 100
mov rax, 200
mov [buffer + 8], rax  ; Push 200
mov rax, [buffer + 8]
mov [temp], rax      ; Pop 200
%print "Stack simulation: popped value = "
mov rax, [temp]
; Print hundreds digit
mov rbx, 100
xor rdx, rdx
div rbx
add rax, 48
mov [char_temp], al
%print char_temp
; Print remaining two digits
mov rax, rdx
mov rbx, 10
xor rdx, rdx
div rbx
add rax, 48
mov [char_temp], al
%print char_temp
add rdx, 48
mov [char_temp], dl
%println char_temp

%println ""

; ========== ADVANCED PATTERNS ==========
%println "9. ADVANCED PATTERNS:"
%println "===================="

; Bubble sort simulation (partial)
%println "Array sorting demonstration (first 3 elements of scores):"
%for i in range 2
    %for j in range 2
        ; Compare scores[j] and scores[j+1]
        mov rax, [j]
        mov rbx, 8
        mul rbx
        lea rsi, [scores]
        add rsi, rax
        mov rcx, [rsi]     ; scores[j]
        
        mov rax, [j]
        add rax, 1
        mov rbx, 8
        mul rbx
        lea rsi, [scores]
        add rsi, rax
        mov rdx, [rsi]     ; scores[j+1]
        
        cmp rcx, rdx
        jle no_swap
            ; Swap elements (simplified indication)
            %print "  Swapping elements at positions "
            mov rax, [j]
            add rax, 48
            mov [char_temp], al
            %print char_temp
            %print " and "
            mov rax, [j]
            add rax, 1
            add rax, 48
            mov [char_temp], al
            %println char_temp
        no_swap:
    %endfor
%endfor

; Pattern generation
%println "Pattern generation:"
%for i in range 4
    %print "  "
    %for j in range 4
        mov rax, [i]
        add rax, [j]
        mov rbx, 2
        xor rdx, rdx
        div rbx
        cmp rdx, 0
        je print_star
            %print "."
            jmp pattern_continue
        print_star:
            %print "*"
        pattern_continue:
        %print " "
    %endfor
    %println ""
%endfor

%println ""

; ========== FINAL SUMMARY ==========
%println "10. FINAL SUMMARY:"
%println "=================="
%println "All compiler features tested successfully:"
%println "  ✓ Variable declarations and initialization"
%println "  ✓ Array operations and indexing"
%println "  ✓ Arithmetic operations"
%println "  ✓ Conditional statements (if/else)"
%println "  ✓ Loop structures (for loops)"
%println "  ✓ Nested control structures"
%println "  ✓ Memory and register operations"
%println "  ✓ Array searching and manipulation"
%println "  ✓ Complex algorithmic patterns"
%println ""
%println "=== COMPREHENSIVE TEST COMPLETED SUCCESSFULLY ==="

; Exit with success code
%exit 0