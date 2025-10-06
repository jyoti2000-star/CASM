%var x 10
%var y 20

%println "Before C code"
%! printf("First: %d\n", x);
%println "Between C commands"
%! printf("Second: %d\n", y);
%println "After C code"
%exit 0