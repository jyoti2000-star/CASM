; 09_assembly_integration.casm
; Demonstrates raw assembly code mixed with CASM constructs

@int array_size = 5
@int sum = 0
@int product = 1

print "=== Assembly Integration Demo ==="
print "Array size:"
print array_size

; Use raw assembly for performance-critical operations
print ""
print "Calculating sum using raw assembly:"

; Initialize registers and perform sum calculation
mov ecx, 1              ; Start from 1
mov eax, 0              ; Initialize sum

sum_loop:
    add eax, ecx        ; Add current number to sum
    inc ecx             ; Increment counter
    cmp ecx, dword [rel var_array_size]
    jle sum_loop        ; Continue if counter <= array_size

; Store assembly result in CASM variable
mov dword [rel var_sum], eax

print "Sum (1 to"
print array_size
print ") ="
print sum

; Calculate product using mixed assembly and CASM
print ""
print "Calculating product using mixed approach:"

mov ecx, 1              ; Start from 1
mov ebx, 1              ; Initialize product

product_loop:
    ; Check condition using assembly
    cmp ecx, dword [rel var_array_size]
    jg product_done
    
    ; Multiply using assembly
    imul ebx, ecx
    inc ecx
    jmp product_loop

product_done:
    ; Store result in CASM variable
    mov dword [rel var_product], ebx

print "Product (1 to"
print array_size
print ") ="
print product

; Advanced assembly operations
print ""
print "Advanced assembly operations:"

; Bit manipulation using assembly
mov eax, dword [rel var_sum]
mov ebx, eax
shl eax, 1              ; Left shift (multiply by 2)
shr ebx, 1              ; Right shift (divide by 2)

mov dword [rel var_sum], eax

print "Sum doubled using bit shift:"
print sum

; Use CASM control flow with assembly calculations
@int counter = 0
print ""
print "Mixed control flow demo:"

while counter < 3
    ; Assembly calculation inside CASM loop
    mov eax, dword [rel var_counter]
    imul eax, eax       ; Square the counter
    add eax, 10         ; Add 10
    
    ; Temporary storage and print
    push eax
    print "Counter squared + 10 ="
    mov eax, dword [rsp]  ; Get value from stack
    mov dword [rel var_sum], eax  ; Store in CASM variable temporarily
    pop eax
    print sum
    
    counter = counter + 1
endwhile

; Final assembly optimization
print ""
print "Final optimized calculation:"

; Complex calculation using assembly
mov eax, dword [rel var_array_size]
mov ebx, dword [rel var_product]
xor edx, edx            ; Clear remainder register
div ebx                 ; Divide array_size by product
mov ecx, eax            ; Store quotient
mov eax, edx            ; Get remainder

print "Array size divided by product:"
print "Quotient stored in assembly register"
print "Remainder stored in assembly register"

print ""
print "Assembly integration demo completed!"