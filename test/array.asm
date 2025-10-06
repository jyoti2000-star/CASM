; Quick verification test with array printing
%var matrix[4] {1, 2, 3, 4}
%var target 3
%var found 0
%var temp_char 0

; Print the array contents
%print "Array contents: "
%for i in range 4
    mov rax, [i]
    mov rbx, 8
    mul rbx  ; Calculate offset
    
    mov rsi, matrix
    add rsi, rax
    mov rcx, [rsi]  ; Load matrix[i]
    
    ; Print current element (simple digit printing)
    add rcx, 48  ; Convert to ASCII
    mov [temp_char], cl
    %print temp_char
    %print " "
%endfor
%println ""

; Print target value
%print "Looking for: "
mov rax, [target]
add rax, 48  ; Convert to ASCII
mov [temp_char], al
%println temp_char

; Search for target in first 2 elements
%for i in range 2
    ; Check if matrix[i] == target
    mov rax, [i]
    mov rbx, 8
    mul rbx  ; Calculate offset
    
    mov rsi, matrix
    add rsi, rax
    mov rcx, [rsi]  ; Load matrix[i]
    
    ; Print current search
    %print "Checking index "
    mov rax, [i]
    add rax, 48  ; Convert to ASCII
    mov [temp_char], al
    %print temp_char
    %print ": value = "
    add rcx, 48  ; Convert to ASCII
    mov [temp_char], cl
    %println temp_char
    
    ; Restore rcx for comparison
    sub rcx, 48
    cmp rcx, [target]
    jne skip_found
    
    mov qword [found], 1
    %println "Found target!"
    
skip_found:
    nop
%endfor

; Print result
%print "Found: "
mov rax, [found]
add rax, 48  ; Convert to ASCII
mov [temp_char], al
%println temp_char

; Exit with the found flag as exit code
mov rax, 60
mov rdi, [found]
syscall