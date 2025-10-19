extern <stdio.h>

; C interop test - forwards include and embedded C lines will be
; processed by the C processor when available. This file demonstrates
; how to forward an include and call into C from CASM.

func main
    ; The following lines are treated as C by the lexer/parser when
    ; written as standard C statements; they will be compiled and
    ; the resulting assembly inlined (if the C processor is available).
    puts("Hello from embedded C (via CASM)\n");
    exit 0
endfunc

call main
