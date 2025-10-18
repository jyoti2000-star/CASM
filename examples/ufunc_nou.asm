; Example using lowercase directives without leading 'U'
var str greeting "Hello, unified ASM!"

print "Start of program"

func foo
    ; simple function that returns 42
    mov eax, 42
    ret eax
endfunc

call foo

print "End of program"

exit 0
