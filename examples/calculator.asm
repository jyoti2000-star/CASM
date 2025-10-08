; 07_calculator.casm
; Interactive calculator demonstrating multiple features

@int num1 = 0
@int num2 = 0
@int choice = 0
@int result = 0

print "=== CASM Calculator ==="
print "1. Addition"
print "2. Subtraction"
print "3. Multiplication"
print "4. Division"
print ""
print "Enter your choice (1-4):"
scan choice

if choice >= 1
    if choice <= 4
        print "Enter first number:"
        scan num1
        print "Enter second number:"
        scan num2
        
        if choice == 1
            result = num1 + num2
            print "Addition result:"
        else
            if choice == 2
                result = num1 - num2
                print "Subtraction result:"
            else
                if choice == 3
                    result = num1 * num2
                    print "Multiplication result:"
                else
                    if choice == 4
                        if num2 != 0
                            result = num1 / num2
                            print "Division result:"
                        else
                            print "Error: Division by zero!"
                            result = 0
                        endif
                    endif
                endif
            endif
        endif
        
        if choice != 4 ; Not division or division was valid
            print result
        endif
    else
        print "Invalid choice! Please select 1-4."
    endif
else
    print "Invalid choice! Please select 1-4."
endif

print "Thank you for using CASM Calculator!"