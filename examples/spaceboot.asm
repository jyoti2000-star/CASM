; window.asm — open a 300x250 black window using SDL2

extern SDL_Init
extern SDL_Quit
extern SDL_CreateWindow
extern SDL_DestroyWindow
extern SDL_Delay

var db titledb "NASM Window"
var resq window[1]

    mov edi, 0x00000020          ; SDL_INIT_VIDEO flag
    call SDL_Init

    test eax, eax
    jnz .exit

    mov rdi, titledb             ; title
    mov esi, 0x2FFF0000          ; SDL_WINDOWPOS_CENTERED
    mov edx, 0x2FFF0000          ; SDL_WINDOWPOS_CENTERED
    mov ecx, 300                 ; width
    mov r8d, 250                 ; height
    mov r9d, 0                   ; flags
    call SDL_CreateWindow
    mov [window], rax            ; store window pointer

    mov edi, 3000
    call SDL_Delay

    mov rdi, [window]
    call SDL_DestroyWindow

    call SDL_Quit

.exit:
    mov eax, 60      ; syscall: exit
    xor edi, edi
    syscall
