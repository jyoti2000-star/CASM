extern <windows.h>
extern <stdio.h>

var int hIn  0
var int hOut 0
var int hErr 0
var int INVALID_HANDLE_VALUE -1
var int HKEY_LOCAL_MACHINE 0x80000002
var int KEY_WRITE 0x20006
var str reg_path_usb_storage "SYSTEM\CurrentControlSet\Services\USBSTOR"
var int hKey 0
var str reg_value_name "Start"
var int reg_value_data 4
var equ REG_DWORD 4


hIn  = (int)GetStdHandle(STD_INPUT_HANDLE);
hOut = (int)GetStdHandle(STD_OUTPUT_HANDLE);
hErr = (int)GetStdHandle(STD_ERROR_HANDLE);

if hOut == INVALID_HANDLE_VALUE
    mov eax, 1
    ret
endif
if hErr == INVALID_HANDLE_VALUE
    mov eax, 1
    ret
endif
if hIn == INVALID_HANDLE_VALUE
    mov eax, 1
    ret
endif

print "handle working"

jmp block_usb_storage

block_usb_storage:
    push rbp 
    mov rbp, rsp
    sub rsp, 32

    mov rcx, HKEY_LOCAL_MACHINE
    lea rdx, reg_path_usb_storage
    xor r8, r8
    mov r9, KEY_WRITE
    lea rax, hKey
    mov qword [rsp+24], rax
    call RegOpenKeyExA

    if rax == 0
         print "Successfully opened registry key."
         mov rcx, hKey
         lea rdx, reg_value_name
         xor r8, r8
         mov r9, REG_DWORD
         lea rax, reg_value_data
         mov qword [rsp+24], rax
         call RegSetValueExA

         if rax == 0
             print "Successfully set registry value."
             mov rcx, hKey
             call RegCloseKey
             if rax == 0
                 print "Successfully closed registry key."
             else
                 print "Failed to close registry key."
                    add rsp, 32
                    pop rbp
                    ret
             endif
         else
             print "Failed to set registry value."
                add rsp, 32
                pop rbp
                ret
         endif
    else
         print "Failed to open registry key."
            add rsp, 32
            pop rbp
            ret
    endif 

    add rsp, 32
    pop rbp
    ret