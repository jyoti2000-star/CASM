; 10_comprehensive_demo.casm
; Comprehensive demonstration of all CASM features

; Variable declarations
@int user_age = 0
@int user_score = 0
@string user_name = ""
@int choice = 0
@int result = 0

print "=== CASM Comprehensive Feature Demo ==="
print "This program demonstrates all major CASM features:"
print "1. Variables and I/O"
print "2. Control flow (if/while/for)"
print "3. C code integration"
print "4. Raw assembly integration"
print "5. Complex calculations"
print ""

; === SECTION 1: Basic I/O and Variables ===
print "=== Section 1: User Input and Variables ==="
print "Please enter your name:"
scan user_name
print "Please enter your age:"
scan user_age
print "Please enter a test score (0-100):"
scan user_score

print ""
print "Hello"
print user_name
print "You are"
print user_age
print "years old with a score of"
print user_score

; === SECTION 2: Conditional Logic ===
print ""
print "=== Section 2: Conditional Analysis ==="

; Age category
if user_age < 18
    print "Category: Minor"
else
    if user_age < 65
        print "Category: Adult"
    else
        print "Category: Senior"
    endif
endif

; Score analysis
if user_score >= 90
    print "Grade: A - Excellent!"
    result = 4
else
    if user_score >= 80
        print "Grade: B - Good job!"
        result = 3
    else
        if user_score >= 70
            print "Grade: C - Satisfactory"
            result = 2
        else
            print "Grade: F - Needs improvement"
            result = 1
        endif
    endif
endif

; === SECTION 3: Loops ===
print ""
print "=== Section 3: Loop Demonstrations ==="

; While loop - countdown
print "Score countdown:"
@int countdown = user_score / 10
while countdown > 0
    print countdown
    countdown = countdown - 1
endwhile
print "Done!"

; For loop - multiplication table
print ""
print "Grade point table:"
for i in range(result)
    @int points = (i + 1) * result
    print "Level"
    print i + 1
    print "Points:"
    print points
endfor

; === SECTION 4: C Integration ===
print ""
print "=== Section 4: C Code Integration ==="

_c_
#include <stdio.h>
#include <math.h>
#include <string.h>

printf("=== C Analysis Section ===\n");
printf("User: %s\n", user_name);
printf("Age: %d\n", user_age);
printf("Score: %d\n", user_score);

// Calculate some statistics
double score_percentage = (double)user_score / 100.0 * 100.0;
double age_factor = sqrt(user_age);
double performance_index = score_percentage * age_factor / 10.0;

printf("\nStatistical Analysis:\n");
printf("Score percentage: %.2f%%\n", score_percentage);
printf("Age factor (sqrt): %.2f\n", age_factor);
printf("Performance index: %.2f\n", performance_index);

// String analysis
int name_length = strlen(user_name);
printf("Name length: %d characters\n", name_length);

// Age-based recommendations
printf("\nRecommendations:\n");
if (user_age < 25) {
    printf("- Focus on skill development\n");
    printf("- Build experience\n");
} else if (user_age < 50) {
    printf("- Leverage your experience\n");
    printf("- Consider leadership roles\n");
} else {
    printf("- Share your wisdom\n");
    printf("- Mentor others\n");
}

// Score improvement suggestions
if (user_score < 70) {
    printf("- Study fundamentals\n");
    printf("- Seek additional help\n");
} else if (user_score < 90) {
    printf("- Review challenging topics\n");
    printf("- Practice more problems\n");
} else {
    printf("- Excellent work!\n");
    printf("- Help others learn\n");
}
_endc_

; === SECTION 5: Assembly Integration ===
print ""
print "=== Section 5: Assembly Integration ==="

; Performance critical calculation using assembly
mov eax, dword [rel var_user_score]
mov ebx, dword [rel var_user_age]

; Calculate weighted score using assembly
imul eax, 3             ; Score weight = 3
add eax, ebx            ; Add age
mov ecx, 4
xor edx, edx
div ecx                 ; Divide by 4

mov dword [rel var_result], eax

print "Weighted calculation result:"
print result

; Bit manipulation demo
mov eax, dword [rel var_user_score]
mov ebx, eax
shl eax, 2              ; Multiply by 4 using bit shift
shr ebx, 1              ; Divide by 2 using bit shift
add eax, ebx            ; Combine results

mov dword [rel var_choice], eax

print "Bit manipulation result:"
print choice

; === SECTION 6: Final Analysis ===
print ""
print "=== Section 6: Final Summary ==="

; Use all features together
@int final_grade = 0
@int bonus_points = 0

; Assembly calculation for bonus
mov eax, dword [rel var_user_age]
cmp eax, 30
jl no_age_bonus
add dword [rel var_bonus_points], 5

no_age_bonus:
mov eax, dword [rel var_user_score]
cmp eax, 95
jl calculate_final
add dword [rel var_bonus_points], 10

calculate_final:
mov eax, dword [rel var_user_score]
add eax, dword [rel var_bonus_points]
mov dword [rel var_final_grade], eax

print "Final analysis for"
print user_name

if bonus_points > 0
    print "Bonus points earned:"
    print bonus_points
endif

print "Final grade:"
print final_grade

; Final C integration for report
_c_
printf("\n=== Final Report ===\n");
printf("Student: %s\n", user_name);
printf("Original Score: %d\n", user_score);
printf("Bonus Points: %d\n", bonus_points);
printf("Final Grade: %d\n", final_grade);

if (final_grade >= 95) {
    printf("Status: OUTSTANDING\n");
} else if (final_grade >= 85) {
    printf("Status: EXCELLENT\n");
} else if (final_grade >= 75) {
    printf("Status: GOOD\n");
} else {
    printf("Status: NEEDS IMPROVEMENT\n");
}

printf("\nThank you for using CASM!\n");
_endc_

print ""
print "=== Demo Complete ==="
print "All CASM features demonstrated successfully!"