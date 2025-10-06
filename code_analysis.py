#!/usr/bin/env python3

from typing import Dict, List, Optional, Set, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import ast
from enum import Enum

class AnalysisType(Enum):
    """Types of code analysis"""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STYLE = "style"
    COMPLEXITY = "complexity"
    DEPENDENCIES = "dependencies"

@dataclass
class AnalysisResult:
    """Result of code analysis"""
    analysis_type: AnalysisType
    severity: str  # "info", "warning", "error", "critical"
    message: str
    line_number: int = 0
    column: int = 0
    code: str = ""
    suggestion: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0

@dataclass
class CodeMetrics:
    """Code quality metrics"""
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    
    # Complexity metrics
    cyclomatic_complexity: int = 0
    nesting_depth: int = 0
    function_count: int = 0
    variable_count: int = 0
    
    # Assembly-specific metrics
    instruction_count: int = 0
    memory_references: int = 0
    register_usage: Dict[str, int] = field(default_factory=dict)
    jump_instructions: int = 0
    call_instructions: int = 0
    
    # Optimization metrics
    optimization_opportunities: int = 0
    code_duplication: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_lines': self.total_lines,
            'code_lines': self.code_lines,
            'comment_lines': self.comment_lines,
            'blank_lines': self.blank_lines,
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'nesting_depth': self.nesting_depth,
            'function_count': self.function_count,
            'variable_count': self.variable_count,
            'instruction_count': self.instruction_count,
            'memory_references': self.memory_references,
            'register_usage': self.register_usage,
            'jump_instructions': self.jump_instructions,
            'call_instructions': self.call_instructions,
            'optimization_opportunities': self.optimization_opportunities,
            'code_duplication': self.code_duplication
        }

class CodeAnalyzer(ABC):
    """Abstract base class for code analyzers"""
    
    @abstractmethod
    def analyze(self, code_lines: List[str]) -> List[AnalysisResult]:
        """Analyze code and return results"""
        pass
    
    @abstractmethod
    def get_analysis_type(self) -> AnalysisType:
        """Get the type of analysis performed"""
        pass

class SyntaxAnalyzer(CodeAnalyzer):
    """Syntax analysis for HLASM code"""
    
    def get_analysis_type(self) -> AnalysisType:
        return AnalysisType.SYNTAX
    
    def analyze(self, code_lines: List[str]) -> List[AnalysisResult]:
        """Analyze syntax issues"""
        results = []
        
        for line_num, line in enumerate(code_lines, 1):
            stripped = line.strip()
            
            # Skip comments and empty lines
            if not stripped or stripped.startswith(';'):
                continue
            
            # Check for common syntax issues
            results.extend(self._check_bracket_balance(line, line_num))
            results.extend(self._check_string_literals(line, line_num))
            results.extend(self._check_instruction_format(line, line_num))
            results.extend(self._check_label_format(line, line_num))
        
        return results
    
    def _check_bracket_balance(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check for balanced brackets"""
        results = []
        stack = []
        
        for i, char in enumerate(line):
            if char in '([{':
                stack.append((char, i))
            elif char in ')]}':
                if not stack:
                    results.append(AnalysisResult(
                        analysis_type=self.get_analysis_type(),
                        severity="error",
                        message=f"Unmatched closing bracket '{char}'",
                        line_number=line_num,
                        column=i + 1,
                        code=line.strip()
                    ))
                else:
                    open_char, _ = stack.pop()
                    expected = {'(': ')', '[': ']', '{': '}'}
                    if expected.get(open_char) != char:
                        results.append(AnalysisResult(
                            analysis_type=self.get_analysis_type(),
                            severity="error",
                            message=f"Mismatched brackets: '{open_char}' and '{char}'",
                            line_number=line_num,
                            column=i + 1,
                            code=line.strip()
                        ))
        
        if stack:
            for open_char, pos in stack:
                results.append(AnalysisResult(
                    analysis_type=self.get_analysis_type(),
                    severity="error",
                    message=f"Unmatched opening bracket '{open_char}'",
                    line_number=line_num,
                    column=pos + 1,
                    code=line.strip()
                ))
        
        return results
    
    def _check_string_literals(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check string literal syntax"""
        results = []
        
        # Find string literals
        in_string = False
        string_char = None
        escaped = False
        
        for i, char in enumerate(line):
            if not in_string:
                if char in '"\'':
                    in_string = True
                    string_char = char
                    start_pos = i
            else:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == string_char:
                    in_string = False
                    string_char = None
        
        if in_string:
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="error",
                message="Unterminated string literal",
                line_number=line_num,
                column=start_pos + 1,
                code=line.strip()
            ))
        
        return results
    
    def _check_instruction_format(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check assembly instruction format"""
        results = []
        stripped = line.strip()
        
        # Skip labels and directives
        if stripped.endswith(':') or stripped.startswith('%') or stripped.startswith('section'):
            return results
        
        # Check for invalid instruction patterns
        if re.match(r'^\s*\w+\s+,', stripped):
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="error",
                message="Invalid instruction format: missing destination",
                line_number=line_num,
                code=stripped,
                suggestion="Check instruction syntax"
            ))
        
        return results
    
    def _check_label_format(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check label format"""
        results = []
        stripped = line.strip()
        
        if stripped.endswith(':'):
            label = stripped[:-1]
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label):
                results.append(AnalysisResult(
                    analysis_type=self.get_analysis_type(),
                    severity="warning",
                    message=f"Invalid label format: '{label}'",
                    line_number=line_num,
                    code=stripped,
                    suggestion="Labels should start with letter or underscore"
                ))
        
        return results

class PerformanceAnalyzer(CodeAnalyzer):
    """Performance analysis for assembly code"""
    
    def get_analysis_type(self) -> AnalysisType:
        return AnalysisType.PERFORMANCE
    
    def analyze(self, code_lines: List[str]) -> List[AnalysisResult]:
        """Analyze performance issues"""
        results = []
        
        for line_num, line in enumerate(code_lines, 1):
            stripped = line.strip()
            
            if not stripped or stripped.startswith(';'):
                continue
            
            results.extend(self._check_inefficient_patterns(line, line_num))
            results.extend(self._check_register_usage(line, line_num))
            results.extend(self._check_memory_access(line, line_num))
        
        return results
    
    def _check_inefficient_patterns(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check for inefficient code patterns"""
        results = []
        stripped = line.strip().lower()
        
        # Inefficient multiplication by powers of 2
        if re.search(r'imul\s+\w+,\s*[248](?:\s|$)', stripped):
            power = re.search(r'imul\s+\w+,\s*([248])(?:\s|$)', stripped).group(1)
            shifts = {'2': '1', '4': '2', '8': '3'}
            if power in shifts:
                results.append(AnalysisResult(
                    analysis_type=self.get_analysis_type(),
                    severity="info",
                    message=f"Multiplication by {power} can be optimized",
                    line_number=line_num,
                    code=line.strip(),
                    suggestion=f"Use 'shl reg, {shifts[power]}' instead of 'imul reg, {power}'",
                    confidence=0.9
                ))
        
        # Inefficient division by powers of 2
        if re.search(r'idiv\s+.*[248](?:\s|$)', stripped):
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="info",
                message="Division by power of 2 can be optimized",
                line_number=line_num,
                code=line.strip(),
                suggestion="Consider using arithmetic right shift (sar) for signed division",
                confidence=0.8
            ))
        
        # Redundant moves
        mov_match = re.search(r'mov\s+(\w+),\s*(\w+)', stripped)
        if mov_match and mov_match.group(1) == mov_match.group(2):
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="warning",
                message="Redundant move instruction",
                line_number=line_num,
                code=line.strip(),
                suggestion="Remove redundant 'mov reg, reg' instruction"
            ))
        
        # Inefficient clearing
        if re.search(r'mov\s+\w+,\s*0(?:\s|$)', stripped):
            reg_match = re.search(r'mov\s+(\w+),\s*0(?:\s|$)', stripped)
            if reg_match:
                reg = reg_match.group(1)
                results.append(AnalysisResult(
                    analysis_type=self.get_analysis_type(),
                    severity="info",
                    message="Register clearing can be optimized",
                    line_number=line_num,
                    code=line.strip(),
                    suggestion=f"Use 'xor {reg}, {reg}' instead of 'mov {reg}, 0'",
                    confidence=0.9
                ))
        
        return results
    
    def _check_register_usage(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check register usage patterns"""
        results = []
        stripped = line.strip().lower()
        
        # Check for excessive register spilling indicators
        if 'push' in stripped or 'pop' in stripped:
            if line_num > 1:  # Context-dependent analysis would be better
                results.append(AnalysisResult(
                    analysis_type=self.get_analysis_type(),
                    severity="info",
                    message="Potential register pressure",
                    line_number=line_num,
                    code=line.strip(),
                    suggestion="Consider optimizing register allocation"
                ))
        
        return results
    
    def _check_memory_access(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check memory access patterns"""
        results = []
        stripped = line.strip()
        
        # Complex addressing modes
        if re.search(r'\[.*\+.*\*.*\+.*\]', stripped):
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="info",
                message="Complex addressing mode may impact performance",
                line_number=line_num,
                code=stripped,
                suggestion="Consider simplifying memory access pattern"
            ))
        
        return results

class SecurityAnalyzer(CodeAnalyzer):
    """Security analysis for assembly code"""
    
    def get_analysis_type(self) -> AnalysisType:
        return AnalysisType.SECURITY
    
    def analyze(self, code_lines: List[str]) -> List[AnalysisResult]:
        """Analyze security issues"""
        results = []
        
        for line_num, line in enumerate(code_lines, 1):
            stripped = line.strip()
            
            if not stripped or stripped.startswith(';'):
                continue
            
            results.extend(self._check_buffer_operations(line, line_num))
            results.extend(self._check_input_validation(line, line_num))
            results.extend(self._check_dangerous_operations(line, line_num))
        
        return results
    
    def _check_buffer_operations(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check for potential buffer overflow vulnerabilities"""
        results = []
        stripped = line.strip().lower()
        
        # Unbounded string operations
        dangerous_ops = ['strcpy', 'strcat', 'sprintf', 'gets']
        for op in dangerous_ops:
            if op in stripped:
                results.append(AnalysisResult(
                    analysis_type=self.get_analysis_type(),
                    severity="warning",
                    message=f"Potentially unsafe operation: {op}",
                    line_number=line_num,
                    code=line.strip(),
                    suggestion=f"Consider using safer alternative to {op}"
                ))
        
        return results
    
    def _check_input_validation(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check for input validation issues"""
        results = []
        stripped = line.strip()
        
        # Direct use of user input in syscalls
        if 'syscall' in stripped.lower() and '%input' in stripped:
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="warning",
                message="User input used directly in system call",
                line_number=line_num,
                code=stripped,
                suggestion="Validate and sanitize user input before use"
            ))
        
        return results
    
    def _check_dangerous_operations(self, line: str, line_num: int) -> List[AnalysisResult]:
        """Check for dangerous operations"""
        results = []
        stripped = line.strip().lower()
        
        # Shellcode-like patterns
        if 'int 0x80' in stripped or 'int 80h' in stripped:
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="info",
                message="Legacy system call interface detected",
                line_number=line_num,
                code=line.strip(),
                suggestion="Consider using modern syscall instruction"
            ))
        
        return results

class ComplexityAnalyzer(CodeAnalyzer):
    """Code complexity analysis"""
    
    def get_analysis_type(self) -> AnalysisType:
        return AnalysisType.COMPLEXITY
    
    def analyze(self, code_lines: List[str]) -> List[AnalysisResult]:
        """Analyze code complexity"""
        results = []
        
        # Calculate cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(code_lines)
        
        if complexity > 10:
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="warning",
                message=f"High cyclomatic complexity: {complexity}",
                line_number=0,
                code="",
                suggestion="Consider breaking down into smaller functions"
            ))
        
        # Check nesting depth
        max_depth = self._calculate_nesting_depth(code_lines)
        if max_depth > 4:
            results.append(AnalysisResult(
                analysis_type=self.get_analysis_type(),
                severity="warning",
                message=f"Deep nesting detected: {max_depth} levels",
                line_number=0,
                code="",
                suggestion="Consider reducing nesting depth"
            ))
        
        return results
    
    def _calculate_cyclomatic_complexity(self, code_lines: List[str]) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for line in code_lines:
            stripped = line.strip().lower()
            
            # Decision points increase complexity
            if any(keyword in stripped for keyword in ['%if', '%while', '%for', 'je', 'jne', 'jl', 'jg', 'jle', 'jge']):
                complexity += 1
            elif '%case' in stripped:
                complexity += 1
        
        return complexity
    
    def _calculate_nesting_depth(self, code_lines: List[str]) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for line in code_lines:
            stripped = line.strip().lower()
            
            # Opening constructs
            if any(keyword in stripped for keyword in ['%if', '%while', '%for', '%function']):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            # Closing constructs
            elif any(keyword in stripped for keyword in ['%endif', '%endwhile', '%endfor', '%endfunction']):
                current_depth = max(0, current_depth - 1)
        
        return max_depth

class CodeQualityEngine:
    """Main code quality analysis engine"""
    
    def __init__(self):
        self.analyzers: List[CodeAnalyzer] = [
            SyntaxAnalyzer(),
            PerformanceAnalyzer(),
            SecurityAnalyzer(),
            ComplexityAnalyzer()
        ]
        self.results: List[AnalysisResult] = []
        self.metrics: Optional[CodeMetrics] = None
    
    def analyze_code(self, code_lines: List[str], 
                    enabled_analyzers: Optional[Set[AnalysisType]] = None) -> List[AnalysisResult]:
        """Run all enabled analyzers on the code"""
        self.results.clear()
        
        for analyzer in self.analyzers:
            if enabled_analyzers is None or analyzer.get_analysis_type() in enabled_analyzers:
                analyzer_results = analyzer.analyze(code_lines)
                self.results.extend(analyzer_results)
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(code_lines)
        
        return self.results
    
    def _calculate_metrics(self, code_lines: List[str]) -> CodeMetrics:
        """Calculate comprehensive code metrics"""
        metrics = CodeMetrics()
        
        metrics.total_lines = len(code_lines)
        
        for line in code_lines:
            stripped = line.strip()
            
            if not stripped:
                metrics.blank_lines += 1
            elif stripped.startswith(';'):
                metrics.comment_lines += 1
            else:
                metrics.code_lines += 1
                
                # Count assembly instructions
                if not (stripped.startswith('%') or stripped.endswith(':') or 
                       stripped.startswith('section') or stripped.startswith('global')):
                    metrics.instruction_count += 1
                    
                    # Count specific instruction types
                    if any(jmp in stripped.lower() for jmp in ['jmp', 'je', 'jne', 'jl', 'jg', 'jle', 'jge', 'jz', 'jnz']):
                        metrics.jump_instructions += 1
                    
                    if 'call' in stripped.lower():
                        metrics.call_instructions += 1
                    
                    # Count memory references
                    if '[' in stripped and ']' in stripped:
                        metrics.memory_references += 1
                    
                    # Count register usage
                    registers = re.findall(r'\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r[89]|r1[0-5])\b', 
                                         stripped, re.IGNORECASE)
                    for reg in registers:
                        reg_lower = reg.lower()
                        metrics.register_usage[reg_lower] = metrics.register_usage.get(reg_lower, 0) + 1
            
            # Count HLASM constructs
            if '%function' in stripped:
                metrics.function_count += 1
            elif '%var' in stripped:
                metrics.variable_count += 1
        
        # Calculate complexity
        metrics.cyclomatic_complexity = ComplexityAnalyzer()._calculate_cyclomatic_complexity(code_lines)
        metrics.nesting_depth = ComplexityAnalyzer()._calculate_nesting_depth(code_lines)
        
        return metrics
    
    def get_results_by_severity(self, severity: str) -> List[AnalysisResult]:
        """Get results filtered by severity"""
        return [result for result in self.results if result.severity == severity]
    
    def get_results_by_type(self, analysis_type: AnalysisType) -> List[AnalysisResult]:
        """Get results filtered by analysis type"""
        return [result for result in self.results if result.analysis_type == analysis_type]
    
    def generate_quality_report(self) -> str:
        """Generate comprehensive quality report"""
        report = []
        report.append("=" * 80)
        report.append("CODE QUALITY ANALYSIS REPORT")
        report.append("=" * 80)
        
        if self.metrics:
            report.append("CODE METRICS:")
            report.append("-" * 40)
            report.append(f"Total Lines: {self.metrics.total_lines}")
            report.append(f"Code Lines: {self.metrics.code_lines}")
            report.append(f"Comment Lines: {self.metrics.comment_lines}")
            report.append(f"Blank Lines: {self.metrics.blank_lines}")
            report.append(f"Instructions: {self.metrics.instruction_count}")
            report.append(f"Functions: {self.metrics.function_count}")
            report.append(f"Variables: {self.metrics.variable_count}")
            report.append(f"Cyclomatic Complexity: {self.metrics.cyclomatic_complexity}")
            report.append(f"Max Nesting Depth: {self.metrics.nesting_depth}")
            report.append("")
            
            if self.metrics.register_usage:
                report.append("REGISTER USAGE:")
                report.append("-" * 40)
                for reg, count in sorted(self.metrics.register_usage.items(), 
                                       key=lambda x: x[1], reverse=True):
                    report.append(f"  {reg}: {count}")
                report.append("")
        
        # Analysis results summary
        if self.results:
            severity_counts = {}
            type_counts = {}
            
            for result in self.results:
                severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
                type_counts[result.analysis_type.value] = type_counts.get(result.analysis_type.value, 0) + 1
            
            report.append("ANALYSIS SUMMARY:")
            report.append("-" * 40)
            for severity, count in sorted(severity_counts.items()):
                report.append(f"{severity.title()}: {count}")
            report.append("")
            
            report.append("BY ANALYSIS TYPE:")
            report.append("-" * 40)
            for analysis_type, count in sorted(type_counts.items()):
                report.append(f"{analysis_type.title()}: {count}")
            report.append("")
            
            # Detailed results
            report.append("DETAILED RESULTS:")
            report.append("-" * 40)
            
            for result in sorted(self.results, key=lambda x: (x.severity, x.line_number)):
                report.append(f"[{result.severity.upper()}] Line {result.line_number}: {result.message}")
                if result.code:
                    report.append(f"  Code: {result.code}")
                if result.suggestion:
                    report.append(f"  Suggestion: {result.suggestion}")
                report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def export_results_json(self) -> Dict[str, Any]:
        """Export results as JSON-serializable dictionary"""
        return {
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'results': [
                {
                    'analysis_type': result.analysis_type.value,
                    'severity': result.severity,
                    'message': result.message,
                    'line_number': result.line_number,
                    'column': result.column,
                    'code': result.code,
                    'suggestion': result.suggestion,
                    'confidence': result.confidence
                }
                for result in self.results
            ]
        }