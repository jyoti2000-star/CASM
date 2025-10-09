; Advanced filesystem example (POSIX)
; Demonstrates CASM control flow + implicit C detection (lines ending with ';')

extern <stdio.h>
extern <dirent.h>
extern <string.h>

var str target_dir "./"
var int file_count 0
var str first_file 256
var str file_line 256
var int read_ok 0

print "=== Filesystem Test ==="

; Count files in directory and store name of first file
DIR *d;
struct dirent *dir;
d = opendir(target_dir);
if d
    file_count = 0;
    while (dir = readdir(d)) != NULL
        if strcmp(dir->d_name, ".") != 0 && strcmp(dir->d_name, "..") != 0
            file_count++;
            if file_count == 1
                strcpy(first_file, dir->d_name);
            endif
        endif
    endwhile
    closedir(d);
else
    file_count = -1;
endif

if file_count > 0
    print "Directory read successful"
    print "First file found:"
    print first_file

    FILE *fp = fopen(first_file, "r");
    if fp
        if fgets(file_line, sizeof(file_line), fp
            read_ok = 1;
        else
            read_ok = 0;
        endif
        fclose(fp);
    else
        read_ok = 0;
    endif

    if read_ok == 1
        print "First line of file:"
        print file_line
    else
        print "Failed to read file or it's empty"
    endif
else
    if file_count == -1
        print "Error: Could not open directory"
    else
        print "No files in directory"
    endif
endif

print "=== Filesystem test complete ==="

mov eax, 99
