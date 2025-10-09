; Advanced network example mixing CASM control flow with small C blocks
; Control flow (if/while/for) is expressed in CASM; embedded C blocks perform
; the platform-specific networking calls and write results back into CASM
; variables.

extern <winsock2.h>
extern <ws2tcpip.h>

; CASM variables (these are exported as assembler symbols C code can use)
var int port = 27015
var buffer recbuf 512
var int ws_ready = 0
var int conn_ok = 0
var int recv_len = 0

print "Starting network test..."

; Initialize Winsock via a small C block that writes ws_ready (1 on success)
    WSADATA wsa;
    if WSAStartup(MAKEWORD(2,2), &wsa) == 0 
        ws_ready = 1;
    else 
        ws_ready = 0;
    endif

if ws_ready == 1
    print "Winsock initialized"

    ; Try connecting up to 3 times (CASM controls the retries)
    var int attempts = 0
    while attempts < 3
        ; Call a small C block that attempts connect/send/recv and sets conn_ok and recv_len
            SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if s == INVALID_SOCKET
                conn_ok = 0;
            else 
                struct sockaddr_in server;
                server.sin_family = AF_INET;
                server.sin_port = htons(port);
                server.sin_addr.s_addr = inet_addr("127.0.0.1");

                if connect(s, (struct sockaddr*)&server, sizeof(server)) == 0 
                    conn_ok = 1;
                    const char *msg = "Hello from CASM+C";
                    send(s, msg, (int)strlen(msg), 0);

                    int r = recv(s, recbuf, 511, 0);
                    if (r > 0)
                        recbuf[r] = 0;
                        recv_len = r;
                    else 
                        recv_len = r;
                    endif

                    closesocket(s);
                else 
                    conn_ok = 0;
                endif
            endif

        if conn_ok == 1
            print "Connected and exchanged data"
            attempts = 100
        else
            print "Connect failed, retrying..."
            attempts = attempts + 1
        endif
    endwhile

    if recv_len > 0
        print "Received (CASM sees buffer):"
        print recbuf
    else
        print "No data received from remote"
    endif

    WSACleanup();

else
    print "Winsock initialization failed"
endif

mov eax, 42 
