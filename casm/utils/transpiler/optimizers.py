from typing import List, Tuple, Set, Dict
from .data_structures import Instruction, Loop
from .simd import SIMDTranslator
from .cfg import ControlFlowGraph
from collections import defaultdict, deque
from .enums import Architecture

class PeepholeOptimizer:
    @staticmethod
    def optimize(instructions: List[str]) -> List[str]:
        if not instructions:
            return instructions
        optimized = []
        i = 0
        while i < len(instructions):
            current = instructions[i].strip()
            if current.startswith('mov '):
                parts = current.split(',')
                if len(parts) == 2:
                    src = parts[1].strip()
                    dst = parts[0].replace('mov', '').strip()
                    if src == dst:
                        i += 1
                        continue
            if i + 1 < len(instructions):
                next_inst = instructions[i + 1].strip()
                if current.startswith('mov ') and next_inst.startswith('add '):
                    optimized.append(current)
                    i += 2
                    continue
            if i + 1 < len(instructions):
                next_inst = instructions[i + 1].strip()
                if current.startswith('push ') and next_inst.startswith('pop '):
                    push_reg = current.replace('push', '').strip()
                    pop_reg = next_inst.replace('pop', '').strip()
                    if push_reg == pop_reg:
                        i += 2
                        continue
            optimized.append(instructions[i])
            i += 1
        return optimized

class LoopOptimizer:
    @staticmethod
    def unroll_loop(loop: Loop, instructions: List[str], factor: int = 4) -> List[str]:
        if factor <= 1 or not loop.trip_count:
            return instructions
        unrolled = []
        body = []
        in_loop = False
        for inst in instructions:
            if loop.start_label in inst:
                in_loop = True
                unrolled.append(inst)
                continue
            if loop.end_label in inst:
                in_loop = False
                for _ in range(factor):
                    unrolled.extend(body)
                unrolled.append(inst)
                body = []
                continue
            if in_loop:
                body.append(inst)
            else:
                unrolled.append(inst)
        return unrolled
    
    @staticmethod
    def vectorize_loop(loop: Loop, instructions: List[str]) -> Tuple[List[str], bool]:
        if not loop.is_vectorizable:
            return instructions, False
        vectorized = []
        for inst in instructions:
            if 'addps' in inst or 'mulps' in inst:
                vectorized.append(SIMDTranslator.translate_sse_to_avx(inst))
            else:
                vectorized.append(inst)
        return vectorized, True

class DeadCodeEliminator:
    @staticmethod
    def eliminate(cfg: ControlFlowGraph) -> Set[str]:
        dead_blocks = set()
        reachable = set()
        if cfg.entry_block:
            queue = deque([cfg.entry_block])
            reachable.add(cfg.entry_block)
            while queue:
                current = queue.popleft()
                block = cfg.blocks.get(current)
                if block:
                    for succ in block.successors:
                        if succ not in reachable:
                            reachable.add(succ)
                            queue.append(succ)
        for label in cfg.blocks:
            if label not in reachable:
                dead_blocks.add(label)
        return dead_blocks
    
    @staticmethod
    def eliminate_unused_stores(instructions: List[Instruction]) -> List[Instruction]:
        used_vars = set()
        for inst in instructions:
            used_vars.update(inst.reads)
        filtered = []
        for inst in instructions:
            if not inst.writes or inst.writes & used_vars:
                filtered.append(inst)
        return filtered

class RegisterAllocator:
    def __init__(self, arch: Architecture):
        self.arch = arch
        self.available_regs = self._get_available_registers()
        self.allocation: Dict[str, str] = {}
        self.spills: Set[str] = set()
    
    def _get_available_registers(self) -> List[str]:
        if self.arch == Architecture.X86_64:
            return ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11']
        elif self.arch == Architecture.ARM64:
            return ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
        return []
    
    def allocate(self, live_ranges: Dict[str, Tuple[int, int]]) -> Dict[str, str]:
        sorted_vars = sorted(live_ranges.items(), key=lambda x: x[1][0])
        active = []
        free_regs = self.available_regs.copy()
        for var, (start, end) in sorted_vars:
            active = [(v, e) for v, e in active if e > start]
            for expired_var, _ in [(v, e) for v, e in active if e <= start]:
                if expired_var in self.allocation:
                    reg = self.allocation[expired_var]
                    if reg not in free_regs:
                        free_regs.append(reg)
            if free_regs:
                reg = free_regs.pop(0)
                self.allocation[var] = reg
                active.append((var, end))
            else:
                self.spills.add(var)
        return self.allocation

class InstructionScheduler:
    @staticmethod
    def schedule(instructions: List[Instruction]) -> List[Instruction]:
        if len(instructions) <= 1:
            return instructions
        scheduled = []
        ready = []
        waiting = instructions.copy()
        cycle = 0
        deps = defaultdict(set)
        for i, inst in enumerate(instructions):
            for j in range(i + 1, len(instructions)):
                if inst.writes & instructions[j].reads:
                    deps[j].add(i)
        completed = set()
        while waiting or ready:
            for i, inst in enumerate(waiting):
                if i not in deps or deps[i].issubset(completed):
                    ready.append((i, inst))
            waiting = [inst for i, inst in enumerate(waiting) if i not in [idx for idx, _ in ready]]
            if ready:
                ready.sort(key=lambda x: x[1].latency, reverse=True)
                idx, inst = ready.pop(0)
                scheduled.append(inst)
                completed.add(idx)
                cycle += inst.latency
        return scheduled
