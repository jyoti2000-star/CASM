#!/usr/bin/env python3
import sys
import os
import time
import shutil
import io
import contextlib
from pathlib import Path
from src.compiler import compiler

try:
    from src.utils.asm_transpiler import transpile_file as _transpile_asm_file
    HAS_ASM_TRANSPILER = True
except Exception:
    HAS_ASM_TRANSPILER = False

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.tree import Tree
from rich.text import Text
from rich import box
from rich.live import Live
from rich.layout import Layout

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

try:
    import questionary
    from questionary import Style
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False
    print("⚠ questionary not installed. Install with: pip install questionary")

console = Console()

from src.utils.colors import Colors
if os.environ.get('CASM_MINIMAL_UI') in ('1', 'true', 'yes', 'on'):
    Colors.MINIMAL = True

# Claude Code inspired style - clean purple/gray palette
custom_style = Style([
    ('qmark', 'fg:#9b59b6 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#9b59b6'),
    ('pointer', 'fg:#9b59b6 bold'),
    ('highlighted', 'fg:#9b59b6'),
    ('selected', 'fg:#7d3c98'),
    ('separator', 'fg:#4a4a4a'),
    ('instruction', 'fg:#888888'),
    ('text', ''),
    ('disabled', 'fg:#666666 italic')
])

def print_header():
    """Print Claude Code inspired header"""
    console.print()
    console.print("[bold #9b59b6]casm[/] [dim]C Assembly Compiler[/]")
    console.print()

def create_file_tree(file_name: str, file_path: str) -> Tree:
    """Create a tree view of file info similar to Claude Code"""
    file_size = os.path.getsize(file_path)
    file_size_kb = file_size / 1024
    
    tree = Tree(
        f"[bold #9b59b6]{file_name}[/]",
        guide_style="dim"
    )
    tree.add(f"[dim]path[/] {file_path}")
    tree.add(f"[dim]size[/] {file_size_kb:.2f} KB")
    tree.add(f"[dim]modified[/] {time.ctime(os.path.getmtime(file_path))}")
    
    return tree

def validate_file(file_path: str) -> bool:
    """Validate input file"""
    if not os.path.exists(file_path):
        console.print(f"[red]✗[/] File not found: [dim]{file_path}[/]")
        return False
    
    if not file_path.endswith('.asm'):
        console.print(f"[yellow]⚠[/] Expected .asm extension, got: [dim]{file_path}[/]")
        console.print("[dim]Continuing anyway...[/]")
        console.print()
    
    return True

def show_menu(file_name: str, platform: str, arch: str) -> str:
    """Display Claude Code style menu"""
    if not HAS_QUESTIONARY:
        console.print("[red]✗[/] questionary library required")
        console.print("[dim]Install with: pip install questionary[/]")
        sys.exit(1)
    
    # Show configuration in a clean table
    config = Table.grid(padding=(0, 2))
    config.add_column(style="dim", justify="right")
    config.add_column()
    config.add_row("target", f"[#9b59b6]{platform}[/] / [#9b59b6]{arch}[/]")
    
    console.print(config)
    console.print()
    
    choices = [
        questionary.Choice('Executable', value='exe'),
        questionary.Choice('Assembly', value='asm'),
        questionary.Choice('Object File', value='obj'),
    ]

    result = questionary.select(
        "Output format:",
        choices=choices,
        style=custom_style,
        use_shortcuts=False,
        use_arrow_keys=True,
        instruction=""
    ).ask()
    
    if result is None:
        raise KeyboardInterrupt
    return result

def compile_with_progress(output_type, input_file, debug_save=False, cflags=None, ldflags=None, platform_opt='linux', arch_opt='x86_64'):
    """Execute compilation with Claude Code style progress"""
    success = False
    
    if output_type == 'exe' and platform_opt == 'windows':
        try:
            import shutil
            if compiler.host != 'windows' and not shutil.which('x86_64-w64-mingw32-gcc'):
                console.print("[red]✗[/] MinGW cross-compiler not found")
                console.print("[dim]Install: brew install mingw-w64[/]")
                return False
        except Exception:
            pass
    
    os.makedirs('output', exist_ok=True)

    def print_error(err_text=None, info_text=None):
        console.print()
        if info_text:
            first = next((ln for ln in (info_text or '').splitlines() if ln.strip()), None)
            if first:
                console.print(f"[dim]{first}[/]")
        if err_text:
            lines = err_text.strip().splitlines()
            preview = '\n'.join(lines[:8])
            if len(lines) > 8:
                preview += '\n[dim]... (use --debug-save for full output)[/]'
            console.print(f"[red]✗[/] Compilation failed\n")
            console.print(preview)
        else:
            console.print("[red]✗[/] Build failed. Use --debug-save for details")

    # EXE compilation
    if output_type == 'exe':
        with Progress(
            SpinnerColumn(style="#9b59b6"),
            TextColumn("[dim]{task.description}[/]"),
            BarColumn(bar_width=40, style="#444444", complete_style="#9b59b6", finished_style="#7d3c98"),
            TextColumn("[#9b59b6]{task.percentage:>3.0f}%[/]"),
            console=console,
            transient=False
        ) as progress:
            task1 = progress.add_task("compiling", total=100)
            progress.update(task1, advance=30)
            
            try:
                from src.utils.c_processor import c_processor
                c_processor.save_debug = bool(debug_save)
                if cflags:
                    c_processor.user_cflags = cflags
                if ldflags:
                    c_processor.user_ldflags = ldflags
                try:
                    c_processor.set_target(platform_opt, arch_opt)
                except Exception:
                    pass
                
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    success = compiler.compile_to_executable(input_file, run_after=False, quiet=True, target=platform_opt, arch=arch_opt)
                
                captured = (buf_out.getvalue() or '') + '\n' + (buf_err.getvalue() or '')
                captured = captured.strip()
                
                if debug_save and captured:
                    try:
                        os.makedirs('output', exist_ok=True)
                        path = os.path.join('output', Path(input_file).stem + '.build.log')
                        with open(path, 'w', encoding='utf-8') as fh:
                            fh.write(captured)
                    except Exception:
                        pass
                
                progress.update(task1, completed=100)
            except Exception as e:
                progress.stop()
                print_error(err_text=(captured if 'captured' in locals() and captured else str(e)))
                raise
            
            if not success:
                progress.stop()
                console.print()
                last_err = getattr(compiler, '_last_error', None)
                last_info = getattr(compiler, '_last_info', None)
                print_error(err_text=last_err, info_text=last_info)
                return False
        return True

    elif output_type == 'obj':
        import tempfile, subprocess
        tmpdir = tempfile.mkdtemp(prefix='casm_')
        try:
            tmp_asm = os.path.join(tmpdir, Path(input_file).stem + '.asm')
            from src.utils.c_processor import c_processor
            if cflags:
                c_processor.user_cflags = cflags
            if ldflags:
                c_processor.user_ldflags = ldflags

            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                ok = compiler.compile_to_assembly(input_file, tmp_asm, quiet=True, target=platform_opt, arch=arch_opt)
            
            captured = (buf_out.getvalue() or '') + '\n' + (buf_err.getvalue() or '')
            captured = captured.strip()
            
            if not ok:
                print_error(err_text=captured or 'Assembly generation failed')
                return False

            if HAS_ASM_TRANSPILER:
                try:
                    _transpile_asm_file(tmp_asm, target=platform_opt, out_path=tmp_asm)
                except Exception as e:
                    console.print(f"[dim]warning: {e}[/]")

            def _nasm_format_for(platform_name: str, arch_name: str) -> str:
                p = (platform_name or 'linux').lower()
                a = (arch_name or 'x86_64').lower()
                if a in ('x86_64', 'amd64', 'x64'):
                    return 'win64' if p == 'windows' else 'elf64'
                else:
                    return 'win32' if p == 'windows' else 'elf32'

            nasm_fmt = _nasm_format_for(platform_opt, arch_opt)
            out_name = Path(input_file).stem + '.o'
            out_path = os.path.join('output', out_name)

            cmd = ['nasm', '-f', nasm_fmt, tmp_asm, '-o', out_path]
            
            with Progress(
                SpinnerColumn(style="#9b59b6"),
                TextColumn("[dim]{task.description}[/]"),
                BarColumn(bar_width=40, style="#444444", complete_style="#9b59b6"),
                TextColumn("[#9b59b6]{task.percentage:>3.0f}%[/]"),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task(f"assembling", total=100)
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        progress.stop()
                        stderr = (result.stderr or '').strip()
                        stdout = (result.stdout or '').strip()
                        combined = (stdout + "\n" + stderr).strip()
                        print_error(err_text=combined)
                        return False
                    progress.update(task, completed=100)
                    success = True
                except Exception:
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
            SpinnerColumn(style="#9b59b6"),
            TextColumn("[dim]{task.description}[/]"),
            BarColumn(bar_width=40, style="#444444", complete_style="#9b59b6"),
            TextColumn("[#9b59b6]{task.percentage:>3.0f}%[/]"),
            console=console,
            transient=False
        ) as progress:
            task1 = progress.add_task("processing", total=100)
            progress.update(task1, advance=50)

            task2 = progress.add_task("generating", total=100)

            try:
                from src.utils.c_processor import c_processor
                if cflags:
                    c_processor.user_cflags = cflags
                if ldflags:
                    c_processor.user_ldflags = ldflags
                
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    success = compiler.compile_to_assembly(input_file, output_file, quiet=True, target=platform_opt, arch=arch_opt)
                
                captured = (buf_out.getvalue() or '') + '\n' + (buf_err.getvalue() or '')
                captured = captured.strip()
                
                if debug_save and captured:
                    try:
                        os.makedirs('output', exist_ok=True)
                        path = os.path.join('output', Path(input_file).stem + '.build.log')
                        with open(path, 'w', encoding='utf-8') as fh:
                            fh.write(captured)
                        info_line = f"Build log: {path}"
                    except Exception:
                        info_line = None
                else:
                    info_line = None
                
                progress.update(task1, completed=100)
                progress.update(task2, completed=100)
            except Exception as e:
                progress.stop()
                err_preview = captured if 'captured' in locals() and captured else str(e)
                print_error(err_text=err_preview)
                raise

            if not success:
                progress.stop()
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
                    lines = proc._last_compile_output.splitlines()
                    preview_text = '\n'.join(lines[:8]) + (('\n[dim]... (use --debug-save)[/]') if len(lines) > 8 else '')

                if not preview_text:
                    preview_text = captured if 'captured' in locals() and captured else ''

                print_error(err_text=preview_text, info_text=info_text)
                return False

    return success

def prune_output(output_type: str, input_file: str):
    """Clean output directory"""
    out_dir = os.path.abspath('output')
    if not os.path.exists(out_dir):
        return

    stem = Path(input_file).stem
    if output_type == 'asm':
        expected = os.path.join(out_dir, f"{stem}.asm")
    elif output_type == 'exe':
        expected = os.path.join(out_dir, f"{stem}.exe")
    elif output_type == 'obj':
        expected = os.path.join(out_dir, f"{stem}.o")
    else:
        return

    for name in os.listdir(out_dir):
        full = os.path.join(out_dir, name)
        if os.path.isdir(full):
            continue
        if os.path.abspath(full) == os.path.abspath(expected):
            continue
        try:
            os.remove(full)
        except Exception:
            pass

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        console.print("[bold #9b59b6]casm[/] [dim]C Assembly Compiler[/]")
        console.print()
        console.print("[red]✗[/] No input file specified")
        console.print()
        console.print("[dim]usage:[/] casm.py <file.asm> [options]")
        console.print()
        console.print("[dim]options:[/]")
        console.print("  --debug-save         save detailed build logs")
        console.print("  --cflags FLAGS       custom C compiler flags")
        console.print("  --ldflags FLAGS      custom linker flags")
        console.print("  --type TYPE          output type: asm|exe|obj")
        console.print("  --platform OS        target: linux|windows")
        console.print("  --arch ARCH          architecture: x86_64|x86")
        sys.exit(1)

    input_file = sys.argv[1]
    
    # Parse options
    options = sys.argv[2:]
    debug_save = False
    cflags = None
    ldflags = None
    requested_type = None
    platform_opt = 'linux'
    arch_opt = 'x86_64'
    
    i = 0
    while i < len(options):
        opt = options[i]
        if opt in ('-d', '--debug-save'):
            debug_save = True
            i += 1
        elif opt == '--cflags' and i + 1 < len(options):
            cflags = options[i+1]
            i += 2
        elif opt == '--ldflags' and i + 1 < len(options):
            ldflags = options[i+1]
            i += 2
        elif opt in ('-t', '--type') and i + 1 < len(options):
            requested_type = options[i+1]
            i += 2
        elif opt == '--platform' and i + 1 < len(options):
            platform_opt = options[i+1]
            i += 2
        elif opt == '--arch' and i + 1 < len(options):
            arch_opt = options[i+1]
            i += 2
        else:
            i += 1
    
    if not validate_file(input_file):
        sys.exit(1)
    
    # Validate platform/arch
    if '--platform' in options and platform_opt.lower() not in ('linux', 'windows'):
        console.print(f"[red]✗[/] Unsupported platform: [dim]{platform_opt}[/]")
        console.print("[dim]Supported: linux, windows[/]")
        sys.exit(1)
    
    if '--arch' in options and arch_opt.lower() not in ('x86_64', 'x86'):
        console.print(f"[red]✗[/] Unsupported architecture: [dim]{arch_opt}[/]")
        console.print("[dim]Supported: x86_64, x86[/]")
        sys.exit(1)

    try:
        print_header()
        console.print(create_file_tree(Path(input_file).name, input_file))
        console.print()
        
        # Platform/arch selection
        try:
            default_platform = compiler._host_to_target()
        except Exception:
            default_platform = platform_opt
        
        try:
            default_arch = compiler.arch
        except Exception:
            default_arch = arch_opt

        if default_platform not in ('linux', 'windows'):
            default_platform = 'linux'
        if default_arch not in ('x86_64', 'x86'):
            default_arch = 'x86_64'

        # Interactive selection if not provided
        if '--platform' not in options and '--arch' not in options and not requested_type:
            if not HAS_QUESTIONARY:
                console.print("[red]✗[/] questionary library required")
                console.print("[dim]Install: pip install questionary[/]")
                sys.exit(1)

            if sys.stdin.isatty():
                try:
                    plat_choice = questionary.select(
                        "Target platform:",
                        choices=[
                            questionary.Choice('linux', value='linux'),
                            questionary.Choice('windows', value='windows'),
                        ],
                        default=default_platform,
                        style=custom_style,
                        instruction=""
                    ).ask()
                    if plat_choice:
                        platform_opt = plat_choice

                    arch_choice = questionary.select(
                        "Architecture:",
                        choices=[
                            questionary.Choice('x86_64', value='x86_64'),
                            questionary.Choice('x86', value='x86'),
                        ],
                        default=default_arch,
                        style=custom_style,
                        instruction=""
                    ).ask()
                    if arch_choice:
                        arch_opt = arch_choice
                except Exception:
                    pass
        
        if requested_type:
            if requested_type not in ('asm', 'exe', 'obj'):
                console.print(f"[red]✗[/] Unknown type: [dim]{requested_type}[/]")
                console.print("[dim]Valid types: asm, exe, obj[/]")
                sys.exit(1)
            output_type = requested_type
        else:
            output_type = show_menu(Path(input_file).name, platform_opt, arch_opt)
        
        start_time = time.time()
        
        # Override for exe if not specified
        if output_type == 'exe' and '--platform' not in options and '--arch' not in options:
            try:
                platform_opt = compiler._host_to_target()
                arch_opt = compiler.arch
            except Exception:
                pass
        
        console.print()
        success = compile_with_progress(output_type, input_file, debug_save, cflags=cflags, ldflags=ldflags, platform_opt=platform_opt, arch_opt=arch_opt)
        
        if not success:
            sys.exit(1)
        
        try:
            prune_output(output_type, input_file)
        except Exception:
            pass
        
        elapsed_time = time.time() - start_time
        
        # Success output - Claude Code style
        if output_type == 'asm':
            output_file = os.path.join("output", Path(input_file).stem + ".asm")
            type_name = "assembly"
        elif output_type == 'exe':
            output_file = os.path.join("output", Path(input_file).stem + ".exe")
            type_name = "executable"
        else:
            output_file = os.path.join("output", Path(input_file).stem + ".o")
            type_name = "object"
        
        file_size = os.path.getsize(output_file) / 1024
        
        console.print()
        console.print(f"[bold #9b59b6]✓[/] Build complete")
        console.print()
        
        # Create result tree
        result = Tree("[dim]output[/]", guide_style="dim")
        result.add(f"[#9b59b6]{Path(output_file).name}[/] [dim]({file_size:.2f} KB)[/]")
        result.add(f"[dim]type[/] {type_name}")
        result.add(f"[dim]time[/] {elapsed_time:.2f}s")
        
        console.print(result)
        console.print()
            
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]✗[/] Cancelled")
        console.print()
        sys.exit(0)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()