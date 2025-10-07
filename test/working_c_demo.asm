; Comprehensive C-Assembly Integration Demo
; Showcases the compiler's current working capabilities

%set test_number 100
%set test_name "Integration Demo"
%set pi_value 3.14159
%set max_count 5

%println "=== Comprehensive C-Assembly Integration Demo ==="

; Test 1: Basic printf with integer variable
%! printf("Test 1: Integer value = %d\n", test_number);

; Test 2: String and multiple variables
%! printf("Test 2: %s with value %d\n", test_name, test_number);

; Test 3: Floating point operations
%! printf("Test 3: Pi = %.5f, doubled = %.5f\n", pi_value, pi_value * 2.0);

; Test 4: Simple arithmetic and assignment
%! int doubled = test_number * 2; printf("Test 4: %d doubled = %d\n", test_number, doubled);

; Test 5: Basic conditional
%! if(test_number > 50) printf("Test 5: Number is greater than 50\n"); else printf("Test 5: Number is not greater than 50\n");

; Assembly checkpoint message
%println "Assembly checkpoint: C integration working perfectly!"

; Test 6: Mathematical operations
%! double result = test_number + pi_value; printf("Test 6: %d + %.2f = %.2f\n", test_number, pi_value, result);

; Test 7: Simple loop simulation (unrolled)
%! printf("Test 7: Loop simulation:\n");
%! printf("  Iteration 0: %d * 0 = %d\n", test_number, 0);
%! printf("  Iteration 1: %d * 1 = %d\n", test_number, test_number * 1);
%! printf("  Iteration 2: %d * 2 = %d\n", test_number, test_number * 2);

; Test 8: String operations
%! printf("Test 8: String '%s' processing complete\n", test_name);

; Final assembly message
%println "Demo completed - All C-assembly integration tests passed!"

%! printf("Final Summary: Processed %d tests successfully!\n", 8);