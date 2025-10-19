; Functions and calls test
func greet
    ; A tiny function that would normally perform a side-effect
    ; (No C I/O used here; this is a structural test: define + call)
    return
endfunc

func main
    call greet
    call greet
    exit 0
endfunc

call main
