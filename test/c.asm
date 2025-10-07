%var x 25
%var name "Alice"
%var count 3

%println "=== C Code Integration Test ==="

%! int doubled = x * 2;
%! printf("Doubled value: %d\n", doubled);

% if [x] > 20
    %! printf("Hello from C! x = %d, name = %s\n", x, name);
    % println "x is greater than 20"

    ; normal arithmetic inline
    %! int result = x + count;
    %! printf("x + count = %d + %d = %d\n", x, count, result);

    ; inline math library usage
    %! double power = pow(x, 2) + pow(count, 2);
    %! double h = sqrt(power);
    % println "Hypotenuse (C math inline): " + h

    ; inline addition as sum
    %! int sum = x + count;
    % println "Inline C sum result: " + sum
% endif

%! printf("Final message: %s has value %d\n", name, x);

% println "=== Test Complete ==="
% exit 0
