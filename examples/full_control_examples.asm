; Comprehensive control flow examples for CASM
; Demonstrates UIF/UELSE/UENDIF, ULOOP (numeric, labeled, while-style, C-style for),
; UPRINT/UPRINTF, canonical registers (cax/ccx/csi), and nested loops.

; --- Data ---
str msg_hello "Hello from CASM!"
str msg_loop "Loop iteration"
str msg_done "Done"

; --- Simple UIF (if) ---
var int flag 1
if flag == 1
    print msg_hello
else
    print msg_done
endif

; --- Numeric counted loop ---
loop 3
    print msg_loop
endloop

; --- Labeled loop with explicit count ---
loop outer 2
    print msg_loop
    ; inner numeric loop
    loop 2
        print msg_loop
    endloop
endloop

; --- While-style loop ---
var int w 0
loop while w < 3
    print msg_loop
    ; increment
    w = w + 1
endloop

; --- C-style for loop ---
var int i 0
loop for int i = 0; i < 4; i++
    ; use canonical registers in generated instructions
    mov cax, i
    mov ccx, cax
    printf "Iteration %d\n" i
endloop

; --- Nested loops & condition inside ---
var int a 0
loop for int a = 0; a < 2; a += 1
    loop for int i = 0; i < 3; i++
        if i == 1
            printf "Inner hit at a=%d i=%d\n" a, i
        endif
    endloop
endloop

; End
print msg_done
