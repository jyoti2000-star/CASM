func main
    ; Variables test: ints, bools, floats and buffer
    int a = 10;
    int b = 5;
    float f = 3.14;
    bool flag = true;
    buffer buf[16];

    ; simple arithmetic and assignment
    a = a + b;
    b = b - 1;

    exit 0
endfunc

call main
