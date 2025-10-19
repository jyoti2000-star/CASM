func main
    ; Control-flow test: while, if, nested blocks
    int counter = 3;

    while counter
        ; simulate a loop body
        if counter == 2
            ; Do nothing special, just a branch test
            counter = counter - 1;
        else
            counter = counter - 1;
        endif
    endwhile

    exit 0
endfunc

call main
