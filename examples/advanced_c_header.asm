extern <math.h>

var float angle = 1.57079632679
var float result = 0.0

print "Calling C (math.h): computing cos(angle)"
result = cos(angle);
printf("C computed cos(angle) = %f\n", result);