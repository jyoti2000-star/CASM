; 02_variables.casm
; Demonstrates variable declaration and assignment with different types

; Integer variables
@int age = 25
@int year = 2024
@int score = 0

; String variables
@string name = "Alice"
@string greeting = "Hello"

; Display initial values
print "=== Variable Declaration Demo ==="
print "Name:"
print name
print "Age:"
print age
print "Year:"
print year

; Variable assignment
age = 30
score = 95
greeting = "Hi there"

print ""
print "=== After Assignment ==="
print "Updated age:"
print age
print "Score:"
print score
print "New greeting:"
print greeting

; Arithmetic operations
year = year + 1
score = score * 2

print ""
print "=== After Calculations ==="
print "Next year:"
print year
print "Doubled score:"
print score