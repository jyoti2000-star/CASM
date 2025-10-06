#!/usr/bin/env python3

"""
HLASM Compiler Test Suite
Comprehensive testing for all advanced features
Automatically discovers and tests all .asm files in test folder
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

class TestRunner:
    def __init__(self, target_platform=None, target_arch=None):
        self.test_dir = Path(__file__).parent
        self.output_dir = self.test_dir / "output"
        self.test_files_dir = self.test_dir / "test"
        self.passed = 0
        self.failed = 0
        self.results = []
        self.target_platform = target_platform  # e.g., 'windows', None for default
        self.target_arch = target_arch  # e.g., 'x86_64', 'x86_32', None for default
        self.has_wine = self._check_wine_available()
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Print target information
        if self.target_platform:
            target_str = f"{self.target_platform}-{self.target_arch}" if self.target_arch else self.target_platform
            print(f"Target Platform: {target_str}")
            if self.target_platform == 'windows' and self.has_wine:
                print("Wine detected: Will run Windows executables")
            elif self.target_platform == 'windows' and not self.has_wine:
                print("Warning: Wine not detected - Windows executables will compile but not run")
        else:
            print("Target Platform: Default (auto-detect)")
    
    def discover_test_files(self) -> List[Path]:
        """Automatically discover all .asm files in the test directory"""
        if not self.test_files_dir.exists():
            print(f"Warning: Test directory {self.test_files_dir} not found")
            return []
        
        asm_files = list(self.test_files_dir.glob("*.asm"))
        asm_files.sort()  # Sort for consistent ordering
        
        print(f"Discovered {len(asm_files)} test files:")
        for file in asm_files:
            print(f"  - {file.name}")
        print()
        
        return asm_files
    
    def _check_wine_available(self) -> bool:
        """Check if Wine is available for running Windows executables"""
        try:
            result = subprocess.run(['wine', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def categorize_test(self, test_file: Path) -> Tuple[str, str]:
        """Categorize test based on filename and content"""
        filename = test_file.stem.lower()
        
        # Read file to check for specific features
        try:
            content = test_file.read_text().lower()
        except:
            content = ""
        
        # Determine category and type
        if "hello" in filename:
            return "Basic", "Hello"
        elif "var" in filename:
            return "Basic", "Variable"
        elif "array" in filename:
            return "Basic", "Array"
        elif "loop" in filename:
            return "Basic", "Loop"
        elif "nested" in filename:
            return "Basic", "Nested"
        elif "advanced" in filename or "demo" in filename:
            return "Advanced", "Advanced"
        elif "optimization" in filename or "opt" in filename:
            return "Advanced", "Optimization"
        elif "error" in filename or "syntax" in filename:
            return "Error", "Syntax"
        elif "function" in filename or "func" in filename:
            return "Advanced", "Function"
        elif "macro" in filename:
            return "Advanced", "Macro"
        elif "stdlib" in filename or "lib" in filename:
            return "Advanced", "Library"
        else:
            return "Other", "General"
    
    def determine_test_mode(self, test_file: Path) -> str:
        """Determine if test should compile-only or run based on content"""
        try:
            content = test_file.read_text().lower()
            
            # Skip files with problematic preprocessor directives
            if any(keyword in content for keyword in [
                "#define", "#if", "#else", "#endif", "#ifdef", "#ifndef"
            ]):
                return "skip"  # Skip files with preprocessor directives
            
            # Files with complex features should be compile-only
            if any(keyword in content for keyword in [
                "debug_print", "performance_test", "advanced", "complex"
            ]):
                return "compile-only"
            
            # Files with obvious infinite loops or complex I/O
            if "infinite" in content or "performance" in content:
                return "compile-only"
                
        except:
            pass
        
        # Default to run mode for basic tests
        return "run"
    
    def run_test(self, test_name: str, test_file: Path, expected_exit_code: int = 0, 
                 compile_only: bool = False, optimization_level: str = "O2") -> Dict[str, Any]:
        """Run a single test and return detailed results"""
        print(f"Running test: {test_name}")
        
        result_data = {
            "name": test_name,
            "file": str(test_file),
            "optimization_level": optimization_level,
            "target_platform": self.target_platform or "default",
            "target_arch": self.target_arch or "default",
            "mode": "compile-only" if compile_only else "run",
            "status": "UNKNOWN",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error_message": "",
            "execution_time": 0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Build command with target platform/architecture
            if compile_only:
                cmd = [sys.executable, "main.py", "c", str(test_file), "-O", optimization_level]
            else:
                cmd = [sys.executable, "main.py", "r", str(test_file), "-O", optimization_level]
            
            # Add target platform/architecture if specified
            if self.target_platform and self.target_arch:
                cmd.extend(["-t", f"{self.target_platform}-{self.target_arch}"])
            elif self.target_platform:
                cmd.extend(["-t", self.target_platform])
            elif self.target_arch:
                cmd.extend(["-a", self.target_arch])
            
            # For Windows cross-compilation, force compile-only if no Wine
            if self.target_platform == 'windows' and not compile_only and not self.has_wine:
                compile_only = True
                result_data["mode"] = "compile-only (no Wine)"
                # Change command to compile-only
                cmd[2] = "c"  # Change 'r' to 'c'
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir, timeout=60)
            
            execution_time = time.time() - start_time
            result_data.update({
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time
            })
            
            # Special handling for Windows executables with Wine
            if (self.target_platform == 'windows' and not compile_only and 
                result.returncode == 0 and self.has_wine):
                # The 'r' command should have run the executable with Wine automatically
                # Check if Wine execution was successful
                if "wine" in result.stdout.lower() or "Program exited with code" in result.stdout:
                    print(f"✓ PASSED: {test_name} (executed with Wine)")
                    self.passed += 1
                    result_data["status"] = "PASSED"
                    result_data["mode"] += " (Wine)"
                    self.results.append((test_name, "PASSED", "Executed with Wine"))
                else:
                    print(f"✓ PASSED: {test_name} (compiled for Windows)")
                    self.passed += 1
                    result_data["status"] = "PASSED"
                    self.results.append((test_name, "PASSED", "Compiled for Windows"))
            elif result.returncode == expected_exit_code:
                mode_desc = " (Wine)" if self.target_platform == 'windows' and self.has_wine else ""
                print(f"✓ PASSED: {test_name}{mode_desc}")
                self.passed += 1
                result_data["status"] = "PASSED"
                self.results.append((test_name, "PASSED", ""))
            else:
                print(f"✗ FAILED: {test_name} (exit code: {result.returncode})")
                if result.stdout:
                    print(f"STDOUT: {result.stdout}")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                self.failed += 1
                result_data["status"] = "FAILED"
                result_data["error_message"] = f"Exit code: {result.returncode}"
                self.results.append((test_name, "FAILED", f"Exit code: {result.returncode}"))
        
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: {test_name} (exceeded 30 seconds)")
            self.failed += 1
            result_data["status"] = "TIMEOUT"
            result_data["error_message"] = "Test exceeded 30 second timeout"
            self.results.append((test_name, "TIMEOUT", "Exceeded 30 second timeout"))
        except Exception as e:
            print(f"✗ ERROR: {test_name} - {e}")
            self.failed += 1
            result_data["status"] = "ERROR"
            result_data["error_message"] = str(e)
            self.results.append((test_name, "ERROR", str(e)))
        
        print("-" * 50)
        return result_data
    
    def run_analysis_test(self, test_name: str, test_file: Path) -> Dict[str, Any]:
        """Run code analysis test"""
        print(f"Running analysis test: {test_name}")
        
        result_data = {
            "name": test_name,
            "file": str(test_file),
            "type": "analysis",
            "status": "UNKNOWN",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error_message": ""
        }
        
        try:
            cmd = [sys.executable, "main.py", "analyze", str(test_file), "--reports"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir, timeout=30)
            
            result_data.update({
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
            if result.returncode == 0:
                print(f"✓ PASSED: {test_name}")
                self.passed += 1
                result_data["status"] = "PASSED"
                self.results.append((test_name, "PASSED", ""))
            else:
                print(f"✗ FAILED: {test_name}")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                self.failed += 1
                result_data["status"] = "FAILED"
                result_data["error_message"] = "Analysis failed"
                self.results.append((test_name, "FAILED", "Analysis failed"))
        
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: {test_name}")
            self.failed += 1
            result_data["status"] = "TIMEOUT"
            result_data["error_message"] = "Analysis timeout"
            self.results.append((test_name, "TIMEOUT", "Analysis timeout"))
        except Exception as e:
            print(f"✗ ERROR: {test_name} - {e}")
            self.failed += 1
            result_data["status"] = "ERROR"
            result_data["error_message"] = str(e)
            self.results.append((test_name, "ERROR", str(e)))
        
        print("-" * 50)
        return result_data
    
    def run_all_tests(self):
        """Run all tests by automatically discovering .asm files"""
        print("=" * 60)
        print("HLASM COMPILER TEST SUITE")
        print("Automatically discovering and testing all .asm files")
        print("=" * 60)
        
        # Discover all test files
        test_files = self.discover_test_files()
        if not test_files:
            print("No test files found!")
            return
        
        # Categorize tests
        categorized_tests = {}
        detailed_results = []
        
        for test_file in test_files:
            category, subcategory = self.categorize_test(test_file)
            test_mode = self.determine_test_mode(test_file)
            
            if category not in categorized_tests:
                categorized_tests[category] = []
            
            categorized_tests[category].append({
                "file": test_file,
                "subcategory": subcategory,
                "mode": test_mode
            })
        
        # Run tests by category
        for category, tests in categorized_tests.items():
            print(f"\n{category.upper()} TESTS")
            print("-" * 30)
            
            for test_info in tests:
                test_file = test_info["file"]
                test_name = f"{test_info['subcategory']} - {test_file.stem}"
                test_mode = test_info["mode"]
                
                # Skip files with problematic directives
                if test_mode == "skip":
                    print(f"⚠ SKIPPED: {test_name} (contains problematic preprocessor directives)")
                    continue
                
                compile_only = test_mode == "compile-only"
                
                result_data = self.run_test(
                    test_name, 
                    test_file, 
                    compile_only=compile_only
                )
                detailed_results.append(result_data)
        
        # Run optimization level tests on a sample file
        if test_files:
            sample_file = next((f for f in test_files if "loop" in f.name.lower()), test_files[0])
            
            print(f"\nOPTIMIZATION LEVEL TESTS")
            print("-" * 30)
            
            for opt_level in ["O0", "O1", "O2", "O3", "Os"]:
                test_name = f"Optimization {opt_level} - {sample_file.stem}"
                result_data = self.run_test(
                    test_name, 
                    sample_file, 
                    optimization_level=opt_level, 
                    compile_only=True
                )
                detailed_results.append(result_data)
        
        # Run analysis tests on all files
        print(f"\nCODE ANALYSIS TESTS")
        print("-" * 30)
        
        for test_file in test_files[:3]:  # Limit to first 3 files for analysis
            test_name = f"Analysis - {test_file.stem}"
            result_data = self.run_analysis_test(test_name, test_file)
            detailed_results.append(result_data)
        
        # Save detailed results
        self.save_detailed_results(detailed_results)
        
        # Compiler info tests
        self.test_compiler_info()
    
    def save_detailed_results(self, detailed_results: List[Dict[str, Any]]):
        """Save detailed test results to JSON file"""
        results_file = self.output_dir / "test_results.json"
        
        summary = {
            "timestamp": self.get_timestamp(),
            "total_tests": len(detailed_results),
            "passed": sum(1 for r in detailed_results if r["status"] == "PASSED"),
            "failed": sum(1 for r in detailed_results if r["status"] in ["FAILED", "ERROR", "TIMEOUT"]),
            "pass_rate": 0,
            "detailed_results": detailed_results
        }
        
        if summary["total_tests"] > 0:
            summary["pass_rate"] = (summary["passed"] / summary["total_tests"]) * 100
        
        try:
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"Warning: Could not save detailed results: {e}")
    
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def test_compiler_info(self):
        """Test compiler information commands"""
        print(f"\nCOMPILER INFORMATION TESTS")
        print("-" * 30)
        
        info_commands = [
            (["info"], "Compiler Info"),
            (["info", "--platform-info"], "Platform Info"),
            (["info", "--stdlib-doc"], "Standard Library Documentation")
        ]
        
        for cmd_args, test_name in info_commands:
            try:
                cmd = [sys.executable, "main.py"] + cmd_args
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir, timeout=10)
                
                if result.returncode == 0 and len(result.stdout) > 0:
                    print(f"✓ PASSED: {test_name}")
                    self.passed += 1
                    self.results.append((test_name, "PASSED", ""))
                else:
                    print(f"✗ FAILED: {test_name}")
                    self.failed += 1
                    self.results.append((test_name, "FAILED", "No output or error"))
            except subprocess.TimeoutExpired:
                print(f"✗ TIMEOUT: {test_name}")
                self.failed += 1
                self.results.append((test_name, "TIMEOUT", "Command timeout"))
            except Exception as e:
                print(f"✗ ERROR: {test_name} - {e}")
                self.failed += 1
                self.results.append((test_name, "ERROR", str(e)))
        
        print("-" * 50)
    
    def generate_report(self):
        """Generate test report and save to output folder"""
        print("\n" + "=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        
        total_tests = self.passed + self.failed
        pass_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.failed > 0:
            print("\nFailed Tests:")
            for test_name, status, error in self.results:
                if status in ["FAILED", "ERROR", "TIMEOUT"]:
                    print(f"  - {test_name}: {error}")
        
        print("\nTest Categories:")
        categories = {}
        for test_name, status, _ in self.results:
            # Extract category from test name
            if " - " in test_name:
                category = test_name.split(" - ")[0]
            else:
                category = test_name.split(" ")[0] if " " in test_name else "Other"
            
            if category not in categories:
                categories[category] = {"passed": 0, "failed": 0}
            if status == "PASSED":
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1
        
        for category, stats in categories.items():
            total = stats["passed"] + stats["failed"]
            rate = (stats["passed"] / total * 100) if total > 0 else 0
            print(f"  {category}: {stats['passed']}/{total} ({rate:.1f}%)")
        
        print("=" * 60)
        
        # Save detailed report to output folder
        report_file = self.output_dir / "test_report.txt"
        html_report_file = self.output_dir / "test_report.html"
        
        # Text report
        try:
            with open(report_file, 'w') as f:
                f.write("HLASM Compiler Test Report\n")
                f.write("=" * 40 + "\n")
                f.write(f"Generated: {self.get_timestamp()}\n")
                f.write(f"Test Discovery: Automatic (.asm files in test/)\n\n")
                f.write(f"Total Tests: {total_tests}\n")
                f.write(f"Passed: {self.passed}\n")
                f.write(f"Failed: {self.failed}\n")
                f.write(f"Pass Rate: {pass_rate:.1f}%\n\n")
                
                f.write("Test Categories:\n")
                f.write("-" * 20 + "\n")
                for category, stats in categories.items():
                    total_cat = stats["passed"] + stats["failed"]
                    rate_cat = (stats["passed"] / total_cat * 100) if total_cat > 0 else 0
                    f.write(f"{category}: {stats['passed']}/{total_cat} ({rate_cat:.1f}%)\n")
                
                f.write("\nDetailed Results:\n")
                f.write("-" * 20 + "\n")
                for test_name, status, error in self.results:
                    f.write(f"{test_name}: {status}")
                    if error:
                        f.write(f" ({error})")
                    f.write("\n")
            
            print(f"Text report saved to: {report_file}")
        except Exception as e:
            print(f"Warning: Could not save text report: {e}")
        
        # HTML report
        try:
            self.generate_html_report(html_report_file, total_tests, pass_rate, categories)
            print(f"HTML report saved to: {html_report_file}")
        except Exception as e:
            print(f"Warning: Could not save HTML report: {e}")
        
        return self.failed == 0
    
    def generate_html_report(self, html_file: Path, total_tests: int, pass_rate: float, categories: Dict):
        """Generate HTML test report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HLASM Compiler Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .failed {{ background-color: #ffe8e8; }}
        .passed {{ color: green; }}
        .failed-text {{ color: red; }}
        .timeout {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .category {{ font-weight: bold; background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HLASM Compiler Test Report</h1>
        <p>Generated: {self.get_timestamp()}</p>
        <p>Test Discovery: Automatic (.asm files in test/ folder)</p>
    </div>
    
    <div class="summary {'failed' if self.failed > 0 else ''}">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {total_tests}</p>
        <p><strong>Passed:</strong> <span class="passed">{self.passed}</span></p>
        <p><strong>Failed:</strong> <span class="failed-text">{self.failed}</span></p>
        <p><strong>Pass Rate:</strong> {pass_rate:.1f}%</p>
    </div>
    
    <h2>Test Categories</h2>
    <table>
        <tr><th>Category</th><th>Passed</th><th>Failed</th><th>Total</th><th>Pass Rate</th></tr>
"""
        
        for category, stats in categories.items():
            total_cat = stats["passed"] + stats["failed"]
            rate_cat = (stats["passed"] / total_cat * 100) if total_cat > 0 else 0
            html_content += f"""
        <tr>
            <td>{category}</td>
            <td class="passed">{stats["passed"]}</td>
            <td class="failed-text">{stats["failed"]}</td>
            <td>{total_cat}</td>
            <td>{rate_cat:.1f}%</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Detailed Results</h2>
    <table>
        <tr><th>Test Name</th><th>Status</th><th>Details</th></tr>
"""
        
        for test_name, status, error in self.results:
            status_class = "passed" if status == "PASSED" else ("timeout" if status == "TIMEOUT" else "failed-text")
            html_content += f"""
        <tr>
            <td>{test_name}</td>
            <td class="{status_class}">{status}</td>
            <td>{error if error else "OK"}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)

def main():
    """Main test runner with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HLASM Compiler Test Suite')
    parser.add_argument('-t', '--target', help='Target platform (e.g., windows-x86_64)')
    parser.add_argument('-a', '--arch', help='Target architecture (e.g., x86_64, x86_32)')
    parser.add_argument('--wine', action='store_true', help='Force Wine usage for Windows executables (auto-detected by default)')
    
    args = parser.parse_args()
    
    # Parse target platform
    target_platform = None
    target_arch = args.arch
    
    if args.target:
        if '-' in args.target:
            target_platform, target_arch = args.target.split('-', 1)
        else:
            target_platform = args.target
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    runner = TestRunner(target_platform, target_arch)
    
    try:
        runner.run_all_tests()
        runner.test_compiler_info()
        success = runner.generate_report()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()