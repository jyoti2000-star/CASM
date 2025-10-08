; 05_while_loops.casm
; Demonstrates while loop constructs

@int counter = 0
@int sum = 0
@int factorial = 1
@int n = 5

print "=== While Loop Demo ==="

; Basic counting loop
print "Counting from 0 to 4:"
counter = 0
while counter < 5
    print counter
    counter = counter + 1
endwhile

; Sum calculation
print ""
print "Calculating sum of numbers 1 to 10:"
counter = 1
sum = 0
while counter <= 10
    sum = sum + counter
    counter = counter + 1
endwhile
print "Sum:"
print sum

; Factorial calculation
print ""
print "Calculating factorial of"
print n
counter = 1
factorial = 1
while counter <= n
    factorial = factorial * counter
    counter = counter + 1
endwhile
print "Factorial:"
print factorial

; Countdown
print ""
print "Countdown:"
counter = 5
while counter > 0
    print counter
    counter = counter - 1
endwhile
print "Blast off!"