; Simple Nested If Statements and Loop Testing
%var x 5
%var y 3
%var age 25
%var score 85
%var counter 0

%println "=== NESTED IF AND LOOP TESTING ==="
%println ""

; ========== NESTED IF STATEMENTS ==========
%println "1. NESTED IF STATEMENTS:"
%println "========================"

%println "Test 1A: Grade system (score = 85)"
%if [score] >= 90
    %println "Grade: A"
%else
    %if [score] >= 80
        %println "Grade: B"
    %else
        %if [score] >= 70
            %println "Grade: C"
        %else
            %println "Grade: F"
        %endif
    %endif
%endif
%println ""

%println "Test 1B: Age and score check (age=25, score=85)"
%if [age] >= 18
    %println "Age: Adult"
    %if [score] >= 75
        %println "Score: Good"
        %println "Result: QUALIFIED"
    %else
        %println "Score: Low"
        %println "Result: NOT QUALIFIED"
    %endif
%else
    %println "Age: Minor"
    %println "Result: NOT QUALIFIED"
%endif
%println ""

%println "Test 1C: Variable comparison (x=5, y=3)"
%if [x] > [y]
    %println "x > y: TRUE"
    %if [x] > 4
        %println "x > 4: TRUE"
        %println "Result: x is large"
    %else
        %println "x > 4: FALSE"
        %println "Result: x is medium"
    %endif
%else
    %println "x > y: FALSE"
    %println "Result: y is larger"
%endif
%println ""

; ========== NESTED LOOPS ==========
%println "2. NESTED LOOPS:"
%println "================"

%println "Test 2A: Simple nested for loops"
%for i in range 3
    %print "Row: "
    %for j in range 3
        %print "* "
    %endfor
    %println ""
%endfor
%println ""

%println "Test 2B: Nested loops with counters"
mov qword [counter], 0
%for i in range 2
    %print "Outer loop "
    mov rax, [i]
    add rax, 48
    %print rax
    %print ": "
    
    %for j in range 3
        mov rax, [counter]
        add rax, 1
        mov [counter], rax
        add rax, 48
        %print rax
        %print " "
    %endfor
    %println ""
%endfor
%println ""

; ========== FOR LOOP WITH NESTED IF ==========
%println "3. FOR LOOP WITH NESTED IF:"
%println "==========================="

%println "Test 3A: Number analysis (1-5)"
%for i in range 5
    mov rax, [i]
    add rax, 1
    mov [x], rax
    
    %print "Number: "
    add rax, 48
    %print rax
    %print " -> "
    
    %if [x] <= 2
        %println "SMALL"
    %else
        %if [x] <= 4
            %println "MEDIUM"
        %else
            %println "LARGE"
        %endif
    %endif
%endfor
%println ""

; ========== WHILE LOOP WITH NESTED IF ==========
%println "4. WHILE LOOP WITH NESTED IF:"
%println "============================="

%println "Test 4A: Countdown with conditions"
mov qword [counter], 5
%while [counter] > 0
    %print "Count: "
    mov rax, [counter]
    add rax, 48
    %print rax
    %print " -> "
    
    %if [counter] > 3
        %println "HIGH"
    %else
        %if [counter] > 1
            %println "MEDIUM"
        %else
            %println "LOW - STOPPING"
            mov qword [counter], 0
        %endif
    %endif
    
    %if [counter] > 0
        mov rax, [counter]
        sub rax, 1
        mov [counter], rax
    %endif
%endwhile
%println ""

%println "=== ALL NESTED TESTS COMPLETED ==="
%exit 0