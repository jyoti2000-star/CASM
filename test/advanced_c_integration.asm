; Advanced C-Assembly Integration Test
; Tests complex C integration capabilities including:
; - Structures and arrays
; - Function calls and recursion
; - Mathematical operations
; - String manipulation
; - Memory allocation concepts
; - Control flow integration

%set x 42
%set name "Advanced Test"
%set count 7
%set pi 3.14159
%set buffer_size 256

%array numbers int 10 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
%array messages string 3 {"Hello", "World", "Integration"}

%struct Person {
    name: string,
    age: int,
    salary: double
}

%println "=== Advanced C-Assembly Integration Test ==="

; Test 1: Basic arithmetic and printf integration
%! int sum = 0; for(int i = 0; i < 10; i++) { sum += numbers[i]; } printf("Sum of array elements: %d\n", sum);

; Test 2: String operations and struct-like behavior
%! struct Person person; person.name = name; person.age = x; person.salary = pi * 1000.0; printf("Person: %s, Age: %d, Salary: %.2f\n", person.name, person.age, person.salary);

; Test 3: Mathematical operations with C math library
%! double result = pow(x, 2) + sqrt(count * 4); double cosine = cos(pi / 4); printf("Math result: %.2f, cos(π/4): %.4f\n", result, cosine);

; Test 4: Conditional logic integration
%! if(x > count) { printf("x (%d) is greater than count (%d)\n", x, count); } else { printf("x (%d) is not greater than count (%d)\n", x, count); }

; Test 5: Loop with assembly variable modification
%println "Loop test with assembly-C interaction:"
%! for(int j = 0; j < count; j++) { printf("Iteration %d: x * j = %d * %d = %d\n", j, x, j, x * j); }

; Test 6: Array processing with C
%! printf("Processing string array:\n"); for(int k = 0; k < 3; k++) { printf("Message %d: %s (length: %d)\n", k, messages[k], strlen(messages[k])); }

; Test 7: Function-like behavior simulation
%! int factorial(int n) { if(n <= 1) return 1; return n * factorial(n - 1); } int fact_result = factorial(5); printf("Factorial of 5: %d\n", fact_result);

; Test 8: Memory and pointer concepts
%! char buffer[buffer_size]; sprintf(buffer, "Formatted: %s has value %d", name, x); printf("Buffer content: %s\n", buffer);

; Test 9: Advanced mathematical expressions
%! double complex_calc = (x * pi) / (count + 1.0); double power_series = pow(2, count) + pow(3, count/2); printf("Complex calculation: %.4f\n", complex_calc); printf("Power series result: %.2f\n", power_series);

; Test 10: Final integration test with multiple data types
%! printf("\n=== Final Integration Summary ===\n"); printf("Integer operations: %d + %d = %d\n", x, count, x + count); printf("Float operations: %.2f * %.2f = %.4f\n", pi, pi, pi * pi); printf("String operations: '%s' contains %d characters\n", name, strlen(name)); printf("Array sum: %d, Average: %.2f\n", sum, (double)sum / 10.0);

; Assembly-specific operations mixed with C
%println "Pure assembly message from within C integration"
%set result_var 999
%! printf("Assembly set variable: %d\n", result_var);

%println "Test completed successfully!"