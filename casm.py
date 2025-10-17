#!/usr/bin/env python3
import sys
import os
import time
import shutil
import io
import contextlib
from pathlib import Path
# Import the global compiler instance
from src.compiler import compiler
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Use questionary for interactive prompts
try:
    import questionary
    from questionary import Style
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False
    print("Warning: questionary not installed. Install with: pip install questionary")

console = Console()

# Allow a minimal UI mode via environment variable for compact output
from src.utils.colors import Colors
if os.environ.get('CASM_MINIMAL_UI') in ('1', 'true', 'yes', 'on'):
    Colors.MINIMAL = True

# Custom style for questionary to match CASM theme
custom_style = Style([
    ('qmark', 'fg:#9b59b6 bold'),           # Question mark - purple
    ('question', 'bold'),                    # Question text
    ('answer', 'fg:#9b59b6 bold'),          # Selected answer - purple
    ('pointer', 'fg:#9b59b6 bold'),         # Pointer - purple
    ('highlighted', 'fg:#9b59b6 bold'),     # Highlighted choice - purple
    ('selected', 'fg:#9b59b6'),             # Selected (but not highlighted) - purple
    ('separator', 'fg:#6c6c6c'),            # Separator
    ('instruction', 'fg:#858585'),          # Instructions
    ('text', ''),                            # Plain text
    ('disabled', 'fg:#858585 italic')       # Disabled choices
])

def print_header():
    """Print stylized header with rich"""
    console.print()
    console.print(Panel.fit(
        "[bold #9b59b6]CASM[/bold #9b59b6] [dim](C Assembly)[/dim]",
        border_style="#8e44ad",
        padding=(0, 2)
    ))
    console.print()

def validate_file(file_path: str) -> bool:
    """Validate input file with rich error messages"""
    if not os.path.exists(file_path):
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        return False
    
    if not file_path.endswith('.asm'):
        console.print(f"[yellow]Warning:[/yellow] Expected .asm extension (got: {file_path}), continuing anyway...")
    
    return True

def show_menu(file_name: str) -> str:
    """Display interactive menu and return selected option"""
    if not HAS_QUESTIONARY:
        console.print("[red]Error:[/red] questionary library required for interactive menu")
        console.print("[dim]Install with:[/dim] pip install questionary")
        sys.exit(1)
    # Do not print the full header here to avoid repeating it when the
    # main flow also prints the header. Keep the menu compact.
    console.print(f"[dim]File:[/dim] [#9b59b6]{file_name}[/#9b59b6]\n")
    
    choices = [
        {'name': 'Compile to Executable', 'value': 'exe'},
        {'name': 'Generate Assembly', 'value': 'asm'},
        {'name': 'Compile to Binary', 'value': 'bin'},
        {'name': 'Compile to Object File', 'value': 'obj'}
    ]
    
    try:
        result = questionary.select(
            "Select output format:",
            choices=choices,
            style=custom_style,
            use_shortcuts=False,
            use_arrow_keys=True,
            instruction="(Use arrow keys)"
        ).ask()
        
        if result is None:  # User cancelled with Ctrl+C
            raise KeyboardInterrupt
            
        return result
    except KeyboardInterrupt:
        raise

def compile_with_progress(output_type, input_file, debug_save=False, cflags=None, ldflags=None):
    """Execute compilation with progress indicators"""
    success = False
    
    # Show a bordered header before starting work, unless minimal UI is enabled
    from src.utils.colors import Colors as _Colors
    if not _Colors.MINIMAL:
        dot = "[#9b59b6]●[/#9b59b6]"
        title = f" {dot}  {Path(input_file).name}"
        console.print(Panel(title, border_style="#8e44ad", padding=(0,2)))

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    def print_error_summary(err_text=None, info_text=None):
        # Stop with a blank line then print concise info + error beneath the
        # progress display. Truncate long error output and suggest --debug-save
        console.print()
        if info_text:
            # Print only the first non-empty line of info
            first_info = next((ln for ln in (info_text or '').splitlines() if ln.strip()), None)
            if first_info:
                console.print(f"[#9b59b6]Info:[/#9b59b6] {first_info}")
        if err_text:
            lines = err_text.strip().splitlines()
            # Show up to 8 lines of the error, then ellipsize
            preview = '\n'.join(lines[:8])
            if len(lines) > 8:
                preview += '\n... (use --debug-save to see full output)'
            console.print(f"[red]Error:[/red] {preview}")
        else:
            console.print(f"[red]Error:[/red] See logs or re-run with --debug-save for details")

    # EXE path
    if output_type == 'exe':
        with Progress(
            SpinnerColumn(style="yellow"),
            TextColumn("[progress.description]{task.description} "),
            BarColumn(complete_style="yellow", finished_style="yellow3", pulse_style="gold3"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task1 = progress.add_task("[#9b59b6]Processing source...", total=100)
            progress.update(task1, advance=30)
            try:
                from src.utils.c_processor import c_processor
                c_processor.save_debug = bool(debug_save)
                if cflags:
                    c_processor.user_cflags = cflags
                if ldflags:
                    c_processor.user_ldflags = ldflags
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    success = compiler.compile_to_executable(input_file, run_after=False, quiet=True)
                captured = (buf_out.getvalue() or '') + '\n' + (buf_err.getvalue() or '')
                captured = captured.strip()
                if debug_save and captured:
                    try:
                        os.makedirs('output', exist_ok=True)
                        path = os.path.join('output', Path(input_file).stem + '.build.log')
                        with open(path, 'w', encoding='utf-8') as fh:
                            fh.write(captured)
                        console.print(f"[#9b59b6]Info:[/#9b59b6] Full build log saved: {path}")
                    except Exception:
                        pass
                progress.update(task1, completed=100)
            except Exception as e:
                progress.stop()
                print_error_summary(err_text=(captured if 'captured' in locals() and captured else str(e)))
                raise
            if not success:
                progress.stop()
                console.print()
                last_err = getattr(compiler, '_last_error', None)
                last_info = getattr(compiler, '_last_info', None)
                if last_info:
                    console.print(f"[#9b59b6]Info:[/#9b59b6] {last_info}")
                if last_err:
                    console.print(f"[red]Error:[/red] {last_err}")
                return False
        return True

    elif output_type in ('bin', 'obj'):
        # Generate assembly in temp dir, then run nasm to produce bin or obj in output
        import tempfile, subprocess
        tmpdir = tempfile.mkdtemp(prefix='casm_')
        try:
            tmp_asm = os.path.join(tmpdir, Path(input_file).stem + '.asm')
            # produce assembly
            from src.utils.c_processor import c_processor
            if cflags:
                c_processor.user_cflags = cflags
            if ldflags:
                c_processor.user_ldflags = ldflags

            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                ok = compiler.compile_to_assembly(input_file, tmp_asm, quiet=True)
            captured = (buf_out.getvalue() or '') + '\n' + (buf_err.getvalue() or '')
            captured = captured.strip()
            if not ok:
                try:
                    print_error_summary(err_text=captured or 'Assembly generation failed')
                except Exception:
                    # best-effort: fallback message
                    print_error_summary(err_text='Assembly generation failed')
                return False

            # run nasm to produce desired output
            nasm_fmt = compiler._nasm_format_for_host() if hasattr(compiler, '_nasm_format_for_host') else 'elf64'
            out_name = Path(input_file).stem + ('.bin' if output_type == 'bin' else '.o')
            out_path = os.path.join('output', out_name)

            cmd = ['nasm', '-f', nasm_fmt, tmp_asm, '-o', out_path]
            with Progress(
                SpinnerColumn(style="yellow"),
                TextColumn("[progress.description]{task.description} "),
                BarColumn(complete_style="yellow", finished_style="yellow3", pulse_style="gold3"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[#9b59b6]Assembling {out_name}...", total=100)
                try:
                    # Capture NASM output so any errors are printed after we
                    # stop the progress display. This keeps the progress bar
                    # visually above error/info messages.
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        progress.stop()
                        # Show concise summary of the assembler error
                        stderr = (result.stderr or '').strip()
                        stdout = (result.stdout or '').strip()
                        combined = (stdout + "\n" + stderr).strip()
                        print_error_summary(err_text=combined)
                        return False
                    progress.update(task, completed=100)
                    success = True
                except Exception:
                    # ensure we propagate after printing
                    raise
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
        return success

    elif output_type == 'asm':
        output_file = os.path.join("output", Path(input_file).stem + ".asm")

        with Progress(
            SpinnerColumn(style="yellow"),
            TextColumn("[progress.description]{task.description} "),
            BarColumn(complete_style="yellow", finished_style="yellow3", pulse_style="gold3"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task1 = progress.add_task("[#9b59b6]Processing source...", total=100)
            progress.update(task1, advance=50)

            task2 = progress.add_task("[#9b59b6]Generating assembly...", total=100)

            try:
                from src.utils.c_processor import c_processor
                if cflags:
                    c_processor.user_cflags = cflags
                if ldflags:
                    c_processor.user_ldflags = ldflags
                # Capture any stdout/stderr emitted by the C processor so it
                # is not printed over the progress UI. We'll show a concise
                # preview after stopping the progress display.
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    success = compiler.compile_to_assembly(input_file, output_file, quiet=True)
                captured = (buf_out.getvalue() or '') + '\n' + (buf_err.getvalue() or '')
                captured = captured.strip()
                # Save full log if requested
                if debug_save and captured:
                    try:
                        os.makedirs('output', exist_ok=True)
                        path = os.path.join('output', Path(input_file).stem + '.build.log')
                        with open(path, 'w', encoding='utf-8') as fh:
                            fh.write(captured)
                        # We'll show a short info line below the progress bar
                        info_line = f"Full build log saved: {path}"
                    except Exception:
                        info_line = None
                else:
                    info_line = None
                progress.update(task1, completed=100)
                progress.update(task2, completed=100)
            except Exception as e:
                progress.stop()
                # Prefer the captured output if available for concise preview
                err_preview = captured if 'captured' in locals() and captured else str(e)
                print_error_summary(err_text=err_preview)
                raise

            if not success:
                progress.stop()
                # Print concise preview (prefer c_processor status/output)
                last_info = getattr(compiler, '_last_info', None)
                proc = None
                try:
                    from src.utils.c_processor import c_processor as proc
                except Exception:
                    proc = None

                preview_text = ''
                info_text = info_line or last_info
                if proc and getattr(proc, '_last_status', None):
                    preview_text = proc._last_status
                if proc and getattr(proc, '_last_compile_output', None):
                    # prefer a short preview of the compile output
                    lines = proc._last_compile_output.splitlines()
                    preview_text = '\n'.join(lines[:8]) + (('\n... (use --debug-save to see full output)') if len(lines) > 8 else '')

                if not preview_text:
                    preview_text = captured if 'captured' in locals() and captured else ''

                print_error_summary(err_text=preview_text, info_text=info_text)
                return False

    return success


def prune_output(output_type: str, input_file: str):
    """Keep only the generated output file in the output/ directory.

    This will remove any other files produced by previous runs so the output
    folder contains only the artifact requested by the user.
    """
    out_dir = os.path.abspath('output')
    if not os.path.exists(out_dir):
        return

    stem = Path(input_file).stem
    # Map output type to expected filename
    if output_type == 'asm':
        expected = os.path.join(out_dir, f"{stem}.asm")
    elif output_type == 'exe':
        expected = os.path.join(out_dir, f"{stem}.exe")
    elif output_type == 'bin':
        expected = os.path.join(out_dir, f"{stem}.bin")
    elif output_type == 'obj':
        expected = os.path.join(out_dir, f"{stem}.o")
    else:
        # Unknown type: do nothing
        return

    # List files in output dir and remove anything not equal to expected
    for name in os.listdir(out_dir):
        full = os.path.join(out_dir, name)
        # skip directories
        if os.path.isdir(full):
            continue
        # if this is the expected file, keep it
        if os.path.abspath(full) == os.path.abspath(expected):
            continue
        try:
            os.remove(full)
        except Exception:
            # best-effort; ignore failures
            pass

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        console.print("[red]Error:[/red] File argument required")
        console.print("[dim]Usage:[/dim] casm.py <file.asm> [--debug-save] [--cflags FLAGS] [--ldflags FLAGS]")
        sys.exit(1)

    input_file = sys.argv[1]
    
    # Parse optional flags
    options = sys.argv[2:]
    debug_save = False
    cflags = None
    ldflags = None
    requested_type = None
    i = 0
    while i < len(options):
        opt = options[i]
        if opt in ('-d', '--debug-save'):
            debug_save = True
            i += 1
            continue
        if opt == '--cflags' and i + 1 < len(options):
            cflags = options[i+1]
            i += 2
            continue
        if opt == '--ldflags' and i + 1 < len(options):
            ldflags = options[i+1]
            i += 2
            continue
        if opt in ('-t', '--type') and i + 1 < len(options):
            requested_type = options[i+1]
            i += 2
            continue
        i += 1
    
    if not validate_file(input_file):
        sys.exit(1)
    
    try:
        # Show header once, then interactive menu. Avoid printing the header
        # again later so it doesn't repeat.
        print_header()
        # If a type was requested via CLI, use it non-interactively. Otherwise
        # show the interactive menu.
        if requested_type:
            allowed = ('asm', 'exe', 'bin', 'obj')
            if requested_type not in allowed:
                console.print(f"[red]Error:[/red] Unknown type: {requested_type}. Allowed: {', '.join(allowed)}")
                sys.exit(1)
            output_type = requested_type
        else:
            output_type = show_menu(Path(input_file).name)
        
        # Track time
        start_time = time.time()
        
        # Execute compilation with progress
        success = compile_with_progress(output_type, input_file, debug_save, cflags=cflags, ldflags=ldflags)
        
        if not success:
            sys.exit(1)
        # Prune output directory so only the requested output file remains
        try:
            prune_output(output_type, input_file)
        except Exception:
            # If pruning fails, just continue — don't treat as fatal
            pass
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Print final success output
        if output_type == 'asm':
            output_file = os.path.join("output", Path(input_file).stem + ".asm")
            console.print()
            console.print(f"[green]Success:[/green] Assembly: {output_file}  Time: [#9b59b6]{elapsed_time:.2f}s[/#9b59b6]")
            console.print()
        elif output_type in ('exe', 'bin', 'obj'):
            ext = ".exe" if output_type == 'exe' else f".{output_type}"
            output_file = os.path.join("output", Path(input_file).stem + ext)
            console.print()
            console.print(f"[green]Success:[/green] Output: {output_file}  Time: [#9b59b6]{elapsed_time:.2f}s[/#9b59b6]")
            console.print()
            
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cancelled by user[/yellow]")
        sys.exit(0)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()