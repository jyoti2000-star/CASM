#!/usr/bin/env python3

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import copy
from abc import ABC, abstractmethod

class OptimizationLevel(Enum):
    """Optimization levels"""
    O0 = 0  # No optimization
    O1 = 1  # Basic optimization
    O2 = 2  # Advanced optimization
    O3 = 3  # Aggressive optimization
    Os = 4  # Size optimization

class OptimizationType(Enum):
    """Types of optimizations"""
    DEAD_CODE_ELIMINATION = "dead_code"
    CONSTANT_FOLDING = "constant_folding"
    CONSTANT_PROPAGATION = "constant_propagation"
    COMMON_SUBEXPRESSION = "common_subexpression"
    LOOP_OPTIMIZATION = "loop_optimization"
    REGISTER_ALLOCATION = "register_allocation"
    INSTRUCTION_SCHEDULING = "instruction_scheduling"
    PEEPHOLE = "peephole"
    TAIL_CALL = "tail_call"
    INLINING = "inlining"

@dataclass
class OptimizationPass:
    """Represents an optimization pass"""
    name: str
    description: str
    optimization_type: OptimizationType
    level_required: OptimizationLevel
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True

@dataclass
class BasicBlock:
    """Represents a basic block in control flow"""
    id: int
    instructions: List[str] = field(default_factory=list)
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    labels: Set[str] = field(default_factory=set)
    is_entry: bool = False
    is_exit: bool = False
    live_in: Set[str] = field(default_factory=set)
    live_out: Set[str] = field(default_factory=set)

@dataclass
class ControlFlowGraph:
    """Control flow graph for optimization analysis"""
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_block: Optional[int] = None
    exit_blocks: Set[int] = field(default_factory=set)
    
    def add_block(self, block_id: int) -> BasicBlock:
        """Add a new basic block"""
        if block_id not in self.blocks:
            self.blocks[block_id] = BasicBlock(block_id)
        return self.blocks[block_id]
    
    def add_edge(self, from_block: int, to_block: int):
        """Add edge between blocks"""
        if from_block in self.blocks and to_block in self.blocks:
            self.blocks[from_block].successors.add(to_block)
            self.blocks[to_block].predecessors.add(from_block)

class OptimizationEngine:
    """Advanced optimization engine for HLASM"""
    
    def __init__(self, level: OptimizationLevel = OptimizationLevel.O2):
        self.level = level
        self.passes: List[OptimizationPass] = []
        self.statistics: Dict[str, int] = {}
        self.enabled_optimizations: Set[OptimizationType] = set()
        
        # Initialize optimization passes
        self._register_optimization_passes()
        self._configure_optimizations()
    
    def _register_optimization_passes(self):
        """Register all available optimization passes"""
        self.passes = [
            OptimizationPass(
                "dead_code_elimination",
                "Remove unreachable and unused code",
                OptimizationType.DEAD_CODE_ELIMINATION,
                OptimizationLevel.O1
            ),
            OptimizationPass(
                "constant_folding",
                "Evaluate constant expressions at compile time",
                OptimizationType.CONSTANT_FOLDING,
                OptimizationLevel.O1
            ),
            OptimizationPass(
                "constant_propagation",
                "Replace variables with their constant values",
                OptimizationType.CONSTANT_PROPAGATION,
                OptimizationLevel.O1
            ),
            OptimizationPass(
                "common_subexpression",
                "Eliminate redundant calculations",
                OptimizationType.COMMON_SUBEXPRESSION,
                OptimizationLevel.O2
            ),
            OptimizationPass(
                "loop_optimization",
                "Optimize loop structures",
                OptimizationType.LOOP_OPTIMIZATION,
                OptimizationLevel.O2
            ),
            OptimizationPass(
                "register_allocation",
                "Optimize register usage",
                OptimizationType.REGISTER_ALLOCATION,
                OptimizationLevel.O2
            ),
            OptimizationPass(
                "peephole",
                "Local instruction sequence optimization",
                OptimizationType.PEEPHOLE,
                OptimizationLevel.O1
            ),
            OptimizationPass(
                "tail_call",
                "Optimize tail recursive calls",
                OptimizationType.TAIL_CALL,
                OptimizationLevel.O2
            ),
            OptimizationPass(
                "inlining",
                "Inline small functions",
                OptimizationType.INLINING,
                OptimizationLevel.O3
            ),
        ]
    
    def _configure_optimizations(self):
        """Configure which optimizations to enable based on level"""
        for pass_obj in self.passes:
            if pass_obj.level_required.value <= self.level.value:
                self.enabled_optimizations.add(pass_obj.optimization_type)
    
    def optimize(self, code_lines: List[str]) -> List[str]:
        """Apply all enabled optimizations to the code"""
        optimized_code = code_lines.copy()
        
        # Initialize statistics
        self.statistics = {opt.value: 0 for opt in OptimizationType}
        
        # Apply optimization passes in order
        if OptimizationType.DEAD_CODE_ELIMINATION in self.enabled_optimizations:
            optimized_code = self._eliminate_dead_code(optimized_code)
        
        if OptimizationType.CONSTANT_FOLDING in self.enabled_optimizations:
            optimized_code = self._fold_constants(optimized_code)
        
        if OptimizationType.CONSTANT_PROPAGATION in self.enabled_optimizations:
            optimized_code = self._propagate_constants(optimized_code)
        
        if OptimizationType.PEEPHOLE in self.enabled_optimizations:
            optimized_code = self._peephole_optimize(optimized_code)
        
        if OptimizationType.LOOP_OPTIMIZATION in self.enabled_optimizations:
            optimized_code = self._optimize_loops(optimized_code)
        
        if OptimizationType.REGISTER_ALLOCATION in self.enabled_optimizations:
            optimized_code = self._optimize_registers(optimized_code)
        
        if OptimizationType.COMMON_SUBEXPRESSION in self.enabled_optimizations:
            optimized_code = self._eliminate_common_subexpressions(optimized_code)
        
        return optimized_code
    
    def _eliminate_dead_code(self, code_lines: List[str]) -> List[str]:
        """Remove unreachable and unused code - DISABLED to prevent code removal"""
        # Dead code elimination disabled - return original code unchanged
        return code_lines
    
    def _fold_constants(self, code_lines: List[str]) -> List[str]:
        """Evaluate constant expressions at compile time"""
        optimized = []
        
        for line in code_lines:
            optimized_line = self._fold_constants_in_line(line)
            optimized.append(optimized_line)
            
            if optimized_line != line:
                self.statistics[OptimizationType.CONSTANT_FOLDING.value] += 1
        
        return optimized
    
    def _fold_constants_in_line(self, line: str) -> str:
        """Fold constants in a single line"""
        # Pattern for simple arithmetic: mov reg, num1 + num2
        pattern = r'mov\s+(\w+),\s*(\d+)\s*([+\-*/])\s*(\d+)'
        match = re.search(pattern, line)
        
        if match:
            reg, num1, op, num2 = match.groups()
            val1, val2 = int(num1), int(num2)
            
            if op == '+':
                result = val1 + val2
            elif op == '-':
                result = val1 - val2
            elif op == '*':
                result = val1 * val2
            elif op == '/':
                result = val1 // val2 if val2 != 0 else val1
            else:
                return line
            
            return f"    mov {reg}, {result}  ; folded: {num1} {op} {num2}"
        
        return line
    
    def _propagate_constants(self, code_lines: List[str]) -> List[str]:
        """Replace variables with their constant values"""
        constants = {}
        optimized = []
        
        for line in code_lines:
            stripped = line.strip()
            
            # Track constant assignments: mov [var], immediate
            const_assign = re.match(r'mov\s+\[(\w+)\],\s*(\d+)', stripped)
            if const_assign:
                var_name, value = const_assign.groups()
                constants[var_name] = value
                optimized.append(line)
                continue
            
            # Replace constant variable uses
            optimized_line = line
            for var_name, value in constants.items():
                pattern = rf'\[{re.escape(var_name)}\]'
                if re.search(pattern, optimized_line):
                    optimized_line = re.sub(pattern, value, optimized_line)
                    self.statistics[OptimizationType.CONSTANT_PROPAGATION.value] += 1
            
            # Invalidate constants if variable is modified
            if stripped.startswith('mov') and any(f'[{var}]' in stripped.split(',')[0] for var in constants.keys()):
                var_match = re.search(r'\[(\w+)\]', stripped.split(',')[0])
                if var_match:
                    invalidated_var = var_match.group(1)
                    if invalidated_var in constants:
                        del constants[invalidated_var]
            
            optimized.append(optimized_line)
        
        return optimized
    
    def _peephole_optimize(self, code_lines: List[str]) -> List[str]:
        """Apply peephole optimizations"""
        optimized = []
        i = 0
        
        while i < len(code_lines):
            current_line = code_lines[i]
            next_line = code_lines[i + 1] if i + 1 < len(code_lines) else ""
            
            # Optimization: mov reg, val; mov reg, val2 -> mov reg, val2
            if (self._is_mov_instruction(current_line) and 
                self._is_mov_instruction(next_line)):
                
                curr_reg = self._get_mov_destination(current_line)
                next_reg = self._get_mov_destination(next_line)
                
                if curr_reg and next_reg and curr_reg == next_reg:
                    # Skip first mov, keep second
                    optimized.append(f"{next_line}  ; optimized: removed redundant mov")
                    self.statistics[OptimizationType.PEEPHOLE.value] += 1
                    i += 2
                    continue
            
            # Optimization: add reg, 0 -> remove
            if re.search(r'add\s+\w+,\s*0', current_line.strip()):
                optimized.append(f"    ; removed: {current_line.strip()}")
                self.statistics[OptimizationType.PEEPHOLE.value] += 1
                i += 1
                continue
            
            # Optimization: mul reg, 1 -> remove
            if re.search(r'mul\s+\w+,\s*1', current_line.strip()):
                optimized.append(f"    ; removed: {current_line.strip()}")
                self.statistics[OptimizationType.PEEPHOLE.value] += 1
                i += 1
                continue
            
            # Optimization: mul reg, 2 -> add reg, reg
            mul_by_2 = re.search(r'mul\s+(\w+),\s*2', current_line.strip())
            if mul_by_2:
                reg = mul_by_2.group(1)
                optimized.append(f"    add {reg}, {reg}  ; optimized: mul by 2")
                self.statistics[OptimizationType.PEEPHOLE.value] += 1
                i += 1
                continue
            
            optimized.append(current_line)
            i += 1
        
        return optimized
    
    def _optimize_loops(self, code_lines: List[str]) -> List[str]:
        """Optimize loop structures - DISABLED due to incorrect unrolling"""
        # Loop optimization disabled - return original code unchanged
        return code_lines
    
    def _optimize_loop_block(self, code_lines: List[str], start_index: int) -> Optional[Dict[str, Any]]:
        """Optimize a specific loop block"""
        # Find loop end
        end_index = start_index + 1
        while end_index < len(code_lines):
            if 'while_end' in code_lines[end_index]:
                break
            end_index += 1
        
        if end_index >= len(code_lines):
            return None
        
        loop_lines = code_lines[start_index:end_index + 1]
        
        # Simple optimization: loop unrolling for small, known iteration counts
        if self._can_unroll_loop(loop_lines):
            unrolled = self._unroll_loop(loop_lines)
            return {
                'optimized_lines': unrolled,
                'next_index': end_index + 1
            }
        
        return None
    
    def _optimize_registers(self, code_lines: List[str]) -> List[str]:
        """Optimize register allocation and usage"""
        optimized = []
        register_usage = {}
        
        for line in code_lines:
            # Track register usage
            registers = self._extract_registers(line)
            for reg in registers:
                register_usage[reg] = register_usage.get(reg, 0) + 1
            
            # Simple optimization: prefer commonly used registers
            optimized_line = self._optimize_register_usage(line, register_usage)
            optimized.append(optimized_line)
            
            if optimized_line != line:
                self.statistics[OptimizationType.REGISTER_ALLOCATION.value] += 1
        
        return optimized
    
    def _eliminate_common_subexpressions(self, code_lines: List[str]) -> List[str]:
        """Eliminate common subexpressions"""
        optimized = []
        expression_cache = {}
        
        for line in code_lines:
            # Look for computation patterns
            computation = self._extract_computation(line)
            if computation:
                if computation in expression_cache:
                    # Use cached result
                    cached_reg = expression_cache[computation]
                    dest_reg = self._get_mov_destination(line)
                    if dest_reg and dest_reg != cached_reg:
                        optimized.append(f"    mov {dest_reg}, {cached_reg}  ; CSE: reused computation")
                        self.statistics[OptimizationType.COMMON_SUBEXPRESSION.value] += 1
                        continue
                else:
                    # Cache this computation
                    dest_reg = self._get_mov_destination(line)
                    if dest_reg:
                        expression_cache[computation] = dest_reg
            
            optimized.append(line)
        
        return optimized
    
    # Helper methods
    def _build_control_flow_graph(self, code_lines: List[str]) -> ControlFlowGraph:
        """Build control flow graph from code"""
        cfg = ControlFlowGraph()
        current_block = 0
        entry_found = False
        
        for i, line in enumerate(code_lines):
            stripped = line.strip()
            
            # Identify entry points (main labels, global directives, section starts)
            if not entry_found and (stripped.endswith(':') or 
                                  'global _main' in stripped or 
                                  'global main' in stripped or
                                  'section .text' in stripped):
                if current_block not in cfg.blocks:
                    cfg.add_block(current_block)
                if cfg.entry_block is None:
                    cfg.entry_block = current_block
                    cfg.blocks[current_block].is_entry = True
                    entry_found = True
            
            if stripped.endswith(':'):
                # New basic block for labels
                if current_block in cfg.blocks and cfg.blocks[current_block].instructions:
                    current_block = len(cfg.blocks)
                if current_block not in cfg.blocks:
                    cfg.add_block(current_block)
                cfg.blocks[current_block].labels.add(stripped[:-1])
                
                # First label is entry if no other entry found
                if not entry_found:
                    cfg.entry_block = current_block
                    cfg.blocks[current_block].is_entry = True
                    entry_found = True
            
            elif current_block not in cfg.blocks:
                cfg.add_block(current_block)
            
            if current_block in cfg.blocks:
                cfg.blocks[current_block].instructions.append(line)
            
            # Check for control flow instructions
            if any(instr in stripped for instr in ['jmp', 'je', 'jne', 'jl', 'jg', 'call', 'ret']):
                if 'ret' in stripped or 'syscall' in stripped:
                    cfg.exit_blocks.add(current_block)
                    cfg.blocks[current_block].is_exit = True
        
        # If no entry found, make first block the entry
        if cfg.entry_block is None and cfg.blocks:
            cfg.entry_block = 0
            cfg.blocks[0].is_entry = True
        
        return cfg
    
    def _find_reachable_blocks(self, cfg: ControlFlowGraph) -> Set[int]:
        """Find all reachable basic blocks"""
        if cfg.entry_block is None:
            return set()
        
        reachable = set()
        to_visit = [cfg.entry_block]
        
        while to_visit:
            block_id = to_visit.pop()
            if block_id in reachable:
                continue
            
            reachable.add(block_id)
            
            if block_id in cfg.blocks:
                for successor in cfg.blocks[block_id].successors:
                    if successor not in reachable:
                        to_visit.append(successor)
        
        return reachable
    
    def _is_mov_instruction(self, line: str) -> bool:
        """Check if line is a mov instruction"""
        return line.strip().startswith('mov ')
    
    def _get_mov_destination(self, line: str) -> Optional[str]:
        """Get destination register from mov instruction"""
        match = re.search(r'mov\s+(\w+)', line.strip())
        return match.group(1) if match else None
    
    def _extract_registers(self, line: str) -> List[str]:
        """Extract all register names from a line"""
        registers = []
        reg_pattern = r'\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r[89]|r1[0-5])\b'
        matches = re.findall(reg_pattern, line, re.IGNORECASE)
        return matches
    
    def _extract_computation(self, line: str) -> Optional[str]:
        """Extract computation pattern from line for CSE"""
        # Look for patterns like: add reg, [memory]
        pattern = r'(add|sub|mul|div)\s+\w+,\s*(.+)'
        match = re.search(pattern, line.strip())
        if match:
            operation, operand = match.groups()
            return f"{operation} {operand.strip()}"
        return None
    
    def _can_unroll_loop(self, loop_lines: List[str]) -> bool:
        """Check if loop can be unrolled"""
        # Simple heuristic: small loops with constant bounds
        return len(loop_lines) < 10
    
    def _unroll_loop(self, loop_lines: List[str]) -> List[str]:
        """Unroll a loop (simplified implementation)"""
        # This is a simplified version - real loop unrolling is much more complex
        unrolled = []
        unrolled.append("; Loop unrolled for optimization")
        
        # Remove loop control structures and duplicate body
        for line in loop_lines:
            if not any(ctrl in line for ctrl in ['while_start', 'while_end', 'jmp', 'cmp']):
                unrolled.append(line)
        
        return unrolled
    
    def _optimize_register_usage(self, line: str, usage_stats: Dict[str, int]) -> str:
        """Optimize register usage based on statistics"""
        # Simple optimization: suggest better register choices
        # This is a placeholder for more sophisticated register allocation
        return line
    
    def get_optimization_report(self) -> str:
        """Generate optimization report"""
        report = []
        report.append("=" * 60)
        report.append("OPTIMIZATION REPORT")
        report.append("=" * 60)
        report.append(f"Optimization Level: {self.level.name}")
        report.append("")
        
        total_optimizations = sum(self.statistics.values())
        report.append(f"Total Optimizations Applied: {total_optimizations}")
        report.append("")
        
        if total_optimizations > 0:
            report.append("Optimization Breakdown:")
            for opt_type, count in self.statistics.items():
                if count > 0:
                    percentage = (count / total_optimizations) * 100
                    report.append(f"  {opt_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        report.append("")
        report.append("Enabled Optimizations:")
        for opt in self.enabled_optimizations:
            report.append(f"  - {opt.value.replace('_', ' ').title()}")
        
        report.append("=" * 60)
        return "\n".join(report)