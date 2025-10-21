import re
from typing import List, Optional
from .data import CVariable, CFunction
from .enums import VariableScope

class VariableAnalyzer:
    """Analyze C variables and their types"""
    
    # C type patterns
    TYPE_PATTERNS = [
        r'(unsigned\s+)?(char|short|int|long|long\s+long)',
        r'(signed\s+)?(char|short|int|long)',
        r'float|double|long\s+double',
        r'void',
        r'size_t|ssize_t|ptrdiff_t',
        r'int8_t|int16_t|int32_t|int64_t',
        r'uint8_t|uint16_t|uint32_t|uint64_t',
        r'intptr_t|uintptr_t',
        r'struct\s+\w+',
        r'union\s+\w+',
        r'enum\s+\w+',
        r'\w+_t',  # typedef'd types
    ]
    
    @staticmethod
    def parse_declaration(declaration: str) -> Optional[CVariable]:
        """Parse a C variable declaration"""
        declaration = declaration.strip().rstrip(';')
        
        # Handle const and volatile
        is_const = 'const' in declaration
        is_volatile = 'volatile' in declaration
        is_static = 'static' in declaration
        
        # Remove qualifiers
        decl = re.sub(r'\b(const|volatile|static|extern|register|auto)\b', '', declaration)
        decl = ' '.join(decl.split())  # normalize whitespace
        
        # Extract type and name
        match = re.match(r'(.+?)\s+(\**)(\w+)(\[(\d+)\])?\s*(=\s*(.+))?', decl)
        
        if not match:
            return None
        
        type_, pointers, name, _, array_size, _, value = match.groups()
        type_ = type_.strip()
        pointer_depth = len(pointers)
        
        # Parse array size
        array_sz = int(array_size) if array_size else None
        
        # Parse value
        val = value.strip() if value else None
        
        return CVariable(
            name=name,
            type_=type_,
            value=val,
            is_const=is_const,
            is_volatile=is_volatile,
            is_static=is_static,
            array_size=array_sz,
            pointer_depth=pointer_depth,
            scope=VariableScope.STATIC if is_static else VariableScope.LOCAL
        )
    
    @staticmethod
    def extract_variables(code: str) -> List[CVariable]:
        """Extract all variable declarations from C code"""
        variables = []
        
        var_pattern = r'(?:const|volatile|static|extern|register)?\s*' + \
                     r'(?:unsigned|signed)?\s*' + \
                     r'(?:char|short|int|long|float|double|void|\w+_t|\w+)\s*' + \
                     r'\*?\s*\w+\s*(?:\[[\d\w]+\])?\s*(?:=\s*[^;]+)?\s*;'
        
        for match in re.finditer(var_pattern, code):
            var = VariableAnalyzer.parse_declaration(match.group(0))
            if var:
                variables.append(var)
        
        return variables
    
    @staticmethod
    def infer_type(value: str) -> str:
        """Infer C type from value"""
        value = value.strip()
        
        # String literal
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            if value.startswith('"'):
                return "const char*"
            return "char"
        
        # Null pointer
        if value.lower() in ('null', 'nullptr', '0'):
            return "void*"
        
        # Hex number
        if value.startswith('0x') or value.startswith('0X'):
            return "unsigned int"
        
        # Float/double
        if '.' in value or 'e' in value.lower():
            if value.endswith('f') or value.endswith('F'):
                return "float"
            return "double"
        
        # Integer
        try:
            int(value)
            return "int"
        except ValueError:
            pass
        
        # Boolean
        if value.lower() in ('true', 'false'):
            return "bool"
        
        return "int"  # default


class FunctionAnalyzer:
    """Analyze C functions"""
    
    @staticmethod
    def extract_functions(code: str) -> List[CFunction]:
        """Extract function definitions from C code"""
        functions = []
        
        func_pattern = r'((?:static|inline|extern)\s+)?(\w+(?:\s+\w+)?)\s+(\**)(\w+)\s*\(([^)]*)\)\s*\{'
        
        for match in re.finditer(func_pattern, code):
            qualifiers, return_type, pointers, name, params = match.groups()
            
            is_static = 'static' in (qualifiers or '')
            is_inline = 'inline' in (qualifiers or '')
            is_extern = 'extern' in (qualifiers or '')
            
            param_list = []
            if params.strip():
                for param in params.split(','):
                    param = param.strip()
                    if param and param != 'void':
                        parts = param.split()
                        if len(parts) >= 2:
                            param_type = ' '.join(parts[:-1])
                            param_name = parts[-1].strip('*')
                            param_list.append((param_name, param_type))
            
            start = match.end() - 1
            body = FunctionAnalyzer._extract_function_body(code, start)
            
            func = CFunction(
                name=name,
                return_type=return_type.strip() + pointers,
                parameters=param_list,
                body=body,
                is_inline=is_inline,
                is_static=is_static,
                is_extern=is_extern,
                line_number=code[:match.start()].count('\n') + 1
            )
            
            functions.append(func)
        
        return functions
    
    @staticmethod
    def _extract_function_body(code: str, start: int) -> str:
        """Extract function body using brace matching"""
        depth = 0
        i = start
        
        while i < len(code):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    return code[start:i+1]
            i += 1
        
        return code[start:]
    
    @staticmethod
    def calculate_complexity(func: CFunction) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        decision_keywords = ['if', 'else', 'while', 'for', 'case', 'default', '&&', '||', '?']
        
        for keyword in decision_keywords:
            complexity += func.body.count(keyword)
        
        return complexity
