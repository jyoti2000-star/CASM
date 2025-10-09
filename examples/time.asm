; Advanced time example (POSIX)
; Demonstrates CASM control flow + implicit C usage with <time.h>

extern <stdio.h>
extern <time.h>
extern <string.h>

var int timestamp 0
var str formatted_time 128
var str new_time 128
var int add_seconds 3600
var int op_ok 0

print "=== Time Example ==="

; Get current time and format it
time_t now;
struct tm *t;
now = time(NULL);
timestamp = (int) now;
if timestamp != -1
    t = localtime(&now);
    op_ok = (t != NULL);
    if op_ok
        strftime(formatted_time, sizeof(formatted_time), "%Y-%m-%d %H:%M:%S", t);
        timestamp = (int)now;
        op_ok = 1;
    else
        op_ok = 0;
    endif
else
    op_ok = 0;
endif

if op_ok == 1
    print "Current local time:"
    print formatted_time

    print "Adding 1 hour (3600 seconds)..."
    timestamp = timestamp + add_seconds;

    time_t future = (time_t)timestamp;
    struct tm *ft = localtime(&future);
    op_ok = (ft != NULL);
    if op_ok
        strftime(new_time, sizeof(new_time), "%Y-%m-%d %H:%M:%S", ft);
        print "Future time:"
        print new_time
    else
        print "Error converting future time"
    endif
else
    print "Failed to get current time"
endif

print "=== Time Example Complete ==="

mov eax, 123
