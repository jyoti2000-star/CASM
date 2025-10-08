var int n = 10
var int sum = 0

    mov ecx, 0
    mov eax, 0
sum_loop:
    add eax, ecx
    inc ecx
    cmp ecx, dword [rel var_n]
    jl sum_loop

    mov dword [rel var_sum], eax

print "Sum is:"
print sum
