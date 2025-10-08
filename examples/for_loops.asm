; 06_for_loops.casm
; Demonstrates for loop constructs

print "=== For Loop Demo ==="

; Basic for loop
print "Basic counting with for loop:"
for i in range(5)
    print "Iteration:"
    print i
endfor

print ""
print "Counting to 10:"
for count in range(10)
    print count
endfor

; Nested for loops
print ""
print "=== Nested For Loops ==="
print "Multiplication table (3x3):"
for i in range(3)
    for j in range(3)
        print "Row"
        print i
        print "Col"
        print j
    endfor
endfor

; For loop with calculations
print ""
print "Powers of 2:"
@int power = 1
for i in range(8)
    print "2^"
    print i
    print "="
    print power
    power = power * 2
endfor