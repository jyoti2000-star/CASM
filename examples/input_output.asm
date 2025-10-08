; 03_input_output.casm
; Demonstrates input and output operations

@int number = 0
@string user_name = ""

print "=== Input/Output Demo ==="
print "What is your name?"
scan user_name

print "Hello"
print user_name
print "Nice to meet you!"

print ""
print "Please enter your favorite number:"
scan number

print "Your favorite number is:"
print number

; Simple calculation with user input
number = number * 2

print "Your number doubled is:"
print number