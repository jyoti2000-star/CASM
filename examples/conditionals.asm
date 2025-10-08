; 04_conditionals.casm
; Demonstrates if/else conditional statements

@int score = 85
@int age = 20

print "=== Conditional Logic Demo ==="
print "Student score:"
print score
print "Student age:"
print age

; Grade calculation
print ""
print "Grade calculation:"
if score >= 90
    print "Grade: A (Excellent!)"
else
    if score >= 80
        print "Grade: B (Good job!)"
    else
        if score >= 70
            print "Grade: C (Fair)"
        else
            if score >= 60
                print "Grade: D (Needs improvement)"
            else
                print "Grade: F (Please study more)"
            endif
        endif
    endif
endif

; Age-based categorization
print ""
print "Age category:"
if age < 13
    print "Child"
else
    if age < 20
        print "Teenager"
    else
        if age < 60
            print "Adult"
        else
            print "Senior"
        endif
    endif
endif

; Multiple conditions
print ""
print "Special conditions:"
if score > 95
    print "Outstanding performance!"
endif

if age >= 18
    print "Eligible to vote"
endif

if score >= 80
    if age < 25
        print "Young high achiever!"
    endif
endif