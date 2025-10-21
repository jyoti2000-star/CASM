from typing import Dict, Optional, Set, List
from collections import deque
from .data_structures import BasicBlock, Instruction

class ControlFlowGraph:
    def __init__(self):
        self.blocks: Dict[str, BasicBlock] = {}
        self.entry_block: Optional[str] = None
        self.exit_blocks: Set[str] = set()
    
    def build_from_instructions(self, instructions: List[Instruction]):
        current_block = BasicBlock(label="entry")
        self.entry_block = "entry"
        self.blocks["entry"] = current_block
        
        for inst in instructions:
            if inst.original_line.strip().endswith(':'):
                label = inst.original_line.strip()[:-1]
                
                if current_block.instructions:
                    self.blocks[current_block.label] = current_block
                
                if label in self.blocks:
                    current_block = self.blocks[label]
                else:
                    current_block = BasicBlock(label=label)
                    self.blocks[label] = current_block
                
                continue
            
            current_block.instructions.append(inst.original_line)
            
            if inst.is_branch:
                if inst.operands:
                    target = inst.operands[0]
                    current_block.successors.add(target)
                    
                    if target not in self.blocks:
                        self.blocks[target] = BasicBlock(label=target)
                    self.blocks[target].predecessors.add(current_block.label)
            
            if inst.is_return:
                self.exit_blocks.add(current_block.label)
        
        if current_block.instructions and current_block.label not in self.blocks:
            self.blocks[current_block.label] = current_block
    
    def compute_dominators(self):
        if not self.entry_block:
            return
        all_blocks = set(self.blocks.keys())
        self.blocks[self.entry_block].dominators = {self.entry_block}
        
        for label in all_blocks - {self.entry_block}:
            self.blocks[label].dominators = all_blocks.copy()
        
        changed = True
        while changed:
            changed = False
            
            for label in all_blocks - {self.entry_block}:
                block = self.blocks[label]
                new_dom = {label}
                
                if block.predecessors:
                    pred_doms = [self.blocks[pred].dominators for pred in block.predecessors]
                    if pred_doms:
                        new_dom |= set.intersection(*pred_doms)
                
                if new_dom != block.dominators:
                    block.dominators = new_dom
                    changed = True
    
    def identify_loops(self):
        loops = []
        from .data_structures import Loop
        for label, block in self.blocks.items():
            for succ in block.successors:
                if succ in block.dominators:
                    loop = Loop(start_label=succ, end_label=label)
                    loops.append(loop)
        return loops
