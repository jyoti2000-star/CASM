from typing import Dict

class SIMDTranslator:
    SSE_TO_AVX: Dict[str, str] = {
        'movaps': 'vmovaps', 'movups': 'vmovups',
        'movapd': 'vmovapd', 'movupd': 'vmovupd',
        'addps': 'vaddps', 'addpd': 'vaddpd',
        'subps': 'vsubps', 'subpd': 'vsubpd',
        'mulps': 'vmulps', 'mulpd': 'vmulpd',
        'divps': 'vdivps', 'divpd': 'vdivpd',
        'andps': 'vandps', 'andpd': 'vandpd',
        'orps': 'vorps', 'orpd': 'vorpd',
        'xorps': 'vxorps', 'xorpd': 'vxorpd',
    }
    
    X86_TO_NEON: Dict[str, str] = {
        'movaps': 'vld1.32', 'movups': 'vld1.32',
        'addps': 'vadd.f32', 'subps': 'vsub.f32',
        'mulps': 'vmul.f32', 'divps': 'vdiv.f32',
    }
    
    @staticmethod
    def translate_sse_to_avx(instruction: str) -> str:
        parts = instruction.split()
        if not parts:
            return instruction
        mnemonic = parts[0]
        if mnemonic in SIMDTranslator.SSE_TO_AVX:
            parts[0] = SIMDTranslator.SSE_TO_AVX[mnemonic]
            return ' '.join(parts)
        return instruction
    
    @staticmethod
    def translate_x86_to_arm(instruction: str) -> str:
        parts = instruction.split()
        if not parts:
            return instruction
        mnemonic = parts[0]
        if mnemonic in SIMDTranslator.X86_TO_NEON:
            neon_mnemonic = SIMDTranslator.X86_TO_NEON[mnemonic]
            return f"    {neon_mnemonic} {{d0}}, {{d0}}, {{d1}}"
        return instruction
