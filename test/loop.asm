; Comprehensive Loop Testing - For and While Loops
%var counter 0
%var sum 0
%var factorial 1
%var n 5

%println "=== LOOP TESTING SUITE ==="
%println ""

; ========== FOR LOOP TESTS ==========
%println "1. FOR LOOP TESTS:"
%println "==================="

; Test 1: Simple for loop counting
%println "Test 1A: Counting from 0 to 4"
%for i in range 5
    %print "Count: "
    mov rax, [i]
    add rax, 48
    %println rax
%endfor
%println ""

; Test 1B: For loop with sum calculation
%println "Test 1B: Sum of numbers 1 to 5"
mov qword [sum], 0
%for i in range 5
    mov rax, [i]
    add rax, 1        ; i+1 (since loop starts at 0)
    add [sum], rax
%endfor
%print "Sum = "
mov rax, [sum]
add rax, 48
%println rax
%println ""

; Test 1C: For loop with factorial calculation
%println "Test 1C: Factorial of 5"
mov qword [factorial], 1
%for i in range 5
    mov rax, [i]
    add rax, 1        ; i+1 (since loop starts at 0)
    mov rbx, [factorial]
    mul rbx
    mov [factorial], rax
%endfor
%print "5! = "
mov rax, [factorial]
; Simple display for small factorials
cmp rax, 120
je show_120
%print "ERROR"
jmp end_factorial
show_120:
%print "120"
end_factorial:
%println ""
%println ""

; ========== WHILE LOOP TESTS ==========
%println "2. WHILE LOOP TESTS:"
%println "===================="

; Test 2A: Simple while loop countdown
%println "Test 2A: Countdown from 5 to 1"
mov qword [counter], 5
%while [counter] > 0
    %print "Count: "
    mov rax, [counter]
    add rax, 48
    %println rax
    
    mov rax, [counter]
    sub rax, 1
    mov [counter], rax
%endwhile
%println ""

; Test 2B: While loop with condition checking
%println "Test 2B: Find first number > 7 (starting from 1)"
mov qword [counter], 1
%while [counter] <= 7
    %print "Checking: "
    mov rax, [counter]
    add rax, 48
    %println rax
    
    mov rax, [counter]
    add rax, 1
    mov [counter], rax
%endwhile
%print "First number > 7: "
mov rax, [counter]
add rax, 48
%println rax
%println ""

; Test 2C: While loop with early termination
%println "Test 2C: Search for number 3 in sequence"
mov qword [counter], 1
%while [counter] <= 10
    %print "Checking: "
    mov rax, [counter]
    add rax, 48
    %println rax
    
    cmp qword [counter], 3
    je found_three
    
    mov rax, [counter]
    add rax, 1
    mov [counter], rax
%endwhile
%println "Not found!"
jmp end_search

found_three:
%println "Found 3!"

end_search:
%println ""

; ========== COMBINED TESTS ==========
%println "3. COMBINED LOOP TESTS:"
%println "======================="

; Test 3A: For loop inside while loop
%println "Test 3A: Multiplication table (3x3)"
mov qword [counter], 1
%while [counter] <= 3
    %print "Row "
    mov rax, [counter]
    add rax, 48
    %print rax
    %print ": "
    
    %for j in range 3
        mov rax, [counter]
        mov rbx, [j]
        add rbx, 1        ; j+1
        mul rbx
        add rax, 48
        %print rax
        %print " "
    %endfor
    %println ""
    
    mov rax, [counter]
    add rax, 1
    mov [counter], rax
%endwhile
%println ""

%println "=== ALL LOOP TESTS COMPLETED ==="
%exit 0