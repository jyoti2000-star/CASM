; Minimal test to isolate relocation issues
%var test_var 42
%var result 0

%println "Testing basic operations"

; Simple variable access
mov rax, [test_var]
mov [result], rax

%println "Basic test complete"
%exit 0