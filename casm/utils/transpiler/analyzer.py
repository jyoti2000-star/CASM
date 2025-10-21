import re
from typing import List, Tuple, Set, Optional
from .data_structures import Instruction
from .enums import InstructionType

class InstructionAnalyzer:
    X86_64_REGS = {
        'integer': ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                   'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15'],
        'float': ['xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7',
                 'xmm8', 'xmm9', 'xmm10', 'xmm11', 'xmm12', 'xmm13', 'xmm14', 'xmm15'],
        'vector': ['ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7',
                  'ymm8', 'ymm9', 'ymm10', 'ymm11', 'ymm12', 'ymm13', 'ymm14', 'ymm15'],
    }
    
    LATENCIES = {
        'mov': 1, 'add': 1, 'sub': 1, 'xor': 1, 'and': 1, 'or': 1,
        'imul': 3, 'idiv': 20, 'mul': 3, 'div': 20,
        'call': 5, 'ret': 5, 'jmp': 1,
        'addps': 3, 'mulps': 5, 'divps': 14,
    }
    
    @staticmethod
    def parse_instruction(line: str, line_num: int) -> Optional[Instruction]:
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('#'):
            return None
        if line.endswith(':'):
            return None
        parts = re.split(r'[\s,]+', line, maxsplit=1)
        if not parts:
            return None
        mnemonic = parts[0].lower()
        operands = []
        if len(parts) > 1:
            operands = [op.strip() for op in re.split(r',', parts[1])]
        inst = Instruction(
            mnemonic=mnemonic,
            operands=operands,
            line_number=line_num,
            original_line=line
        )
        inst.type = InstructionAnalyzer._classify_instruction(mnemonic)
        inst.is_branch = mnemonic in ['jmp', 'je', 'jne', 'jz', 'jnz', 'jl', 'jle', 'jg', 'jge', 'call']
        inst.is_call = mnemonic == 'call'
        inst.is_return = mnemonic in ['ret', 'retn']
        inst.reads, inst.writes = InstructionAnalyzer._analyze_operands(mnemonic, operands)
        inst.latency = InstructionAnalyzer.LATENCIES.get(mnemonic, 1)
        return inst
    
    @staticmethod
    def _classify_instruction(mnemonic: str) -> InstructionType:
        if mnemonic in ['movaps', 'movups', 'addps', 'mulps', 'vmovaps']:
            return InstructionType.VECTOR
        elif mnemonic in ['lock', 'xchg', 'cmpxchg']:
            return InstructionType.ATOMIC
        elif mnemonic in ['mov', 'lea', 'push', 'pop']:
            return InstructionType.MEMORY
        elif mnemonic in ['jmp', 'je', 'jne', 'call', 'ret']:
            return InstructionType.CONTROL
        elif mnemonic in ['add', 'sub', 'mul', 'imul', 'div', 'idiv']:
            return InstructionType.ARITHMETIC
        elif mnemonic in ['and', 'or', 'xor', 'not', 'shl', 'shr']:
            return InstructionType.LOGICAL
        elif mnemonic in ['cmp', 'test']:
            return InstructionType.COMPARISON
        else:
            return InstructionType.SCALAR
    
    @staticmethod
    def _analyze_operands(mnemonic: str, operands: List[str]):
        reads = set()
        writes = set()
        if not operands:
            return reads, writes
        for op in operands:
            regs = InstructionAnalyzer._extract_registers(op)
            reads.update(regs)
        if mnemonic in ['mov', 'lea', 'add', 'sub', 'xor', 'and', 'or', 'mul', 'imul']:
            dest_regs = InstructionAnalyzer._extract_registers(operands[0])
            writes.update(dest_regs)
        return reads, writes
    
    @staticmethod
    def _extract_registers(operand: str) -> Set[str]:
        regs = set()
        for reg_class in InstructionAnalyzer.X86_64_REGS.values():
            for reg in reg_class:
                if reg in operand.lower():
                    regs.add(reg)
        return regs
