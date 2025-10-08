; hybrid_example.casm
; Simple hybrid CASM program demonstrating embedded C and raw assembly

extern WriteConsoleA
extern <math.h>
print "=== Hybrid Example: C + Assembly ==="

var int n 5
var int fact 0
var int sum 0
var buffer[125]

print "Calculating factorial of"
print n

; Use embedded C to compute factorial and assign result to CASM variable `fact`

fact = 1;

var int age = 18

if age >= 18
    print "You are an adult"
endif

if (age < 13)
    print "You are a child"
else
    fact = fact + 10;
endif


print "Now compute sum 0..(n-1) using raw assembly"

; Raw assembly: compute sum = 0 + 1 + ... + (n-1)
mov ecx, 0      ; counter = 0
mov eax, 0      ; accumulator = 0

sum_loop:
    add eax, ecx
    inc ecx
    cmp ecx, dword [rel var_n]
    jl sum_loop

; Store the assembly result into the CASM variable `sum`
mov dword [rel var_sum], eax

print "Assembly sum (0..n-1):"
print sum

print "Back in CASM: factorial and sum"
print fact
print sum

; End of hybrid example
