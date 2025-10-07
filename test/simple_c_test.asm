; Simple C Integration Test
%set x 42
%set name "Test"

%println "=== Simple C Integration Test ==="

; Simple printf test
%! printf("Hello from C! x = %d\n", x);

; Simple arithmetic
%! int result = x * 2; printf("Result: %d\n", result);

%println "Simple test completed!"