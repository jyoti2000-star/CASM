; 08_c_integration.casm
; Demonstrates C code integration with _c_ and _endc_ blocks

@int radius = 5
@int temperature = 75

print "=== C Integration Demo ==="
print "CASM variables:"
print "Radius:"
print radius
print "Temperature:"
print temperature

; Embed C code for mathematical calculations
_c_
#include <stdio.h>
#include <math.h>

printf("\n=== C Code Section ===\n");
printf("Accessing CASM variables from C:\n");
printf("Radius from CASM: %d\n", radius);
printf("Temperature from CASM: %d\n", temperature);

// Mathematical calculations using C library functions
double pi = 3.14159265359;
double area = pi * radius * radius;
double circumference = 2 * pi * radius;

printf("\nCircle calculations:\n");
printf("Area: %.2f\n", area);
printf("Circumference: %.2f\n", circumference);

// Temperature conversion
double celsius = (temperature - 32) * 5.0 / 9.0;
printf("\nTemperature conversion:\n");
printf("%d°F = %.2f°C\n", temperature, celsius);

// Define a C function
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

printf("\nFibonacci sequence (first 10 numbers):\n");
for (int i = 0; i < 10; i++) {
    printf("%d ", fibonacci(i));
}
printf("\n");

// More complex math
double power_result = pow(radius, 3);
double sqrt_result = sqrt(temperature);

printf("\nAdvanced math:\n");
printf("Radius cubed: %.2f\n", power_result);
printf("Square root of temperature: %.2f\n", sqrt_result);
_endc_

print ""
print "=== Back to CASM ==="
print "C code execution completed!"

; Continue with CASM logic
if radius > 3
    print "Large radius detected in CASM"
endif

if temperature > 70
    print "Warm temperature detected in CASM"
endif

print "Demo completed successfully!"