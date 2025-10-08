#!/usr/bin/env python3
from typing import Dict, List, Set

class MinimalStdLib:
    """Minimal standard library with only essential I/O functions"""
    
    def __init__(self):
        self.available_functions = {
            'println': self._println_info,
            'scanf': self._scanf_info,
        }
    
    def _println_info(self) -> Dict:
        """Information about println function"""
        return {
            'name': 'println',
            'description': 'Print a line to console with newline',
            'parameters': ['message'],
            'example': '%println "Hello World"',
            'external_deps': ['printf']
        }
    
    def _scanf_info(self) -> Dict:
        """Information about scanf function"""
        return {
            'name': 'scanf',
            'description': 'Read formatted input from console',
            'parameters': ['format_string', 'variable'],
            'example': '%scanf "%d" number',
            'external_deps': ['scanf']
        }
    
    def get_function_info(self, name: str) -> Dict:
        """Get information about a function"""
        if name in self.available_functions:
            return self.available_functions[name]()
        return {}
    
    def has_function(self, name: str) -> bool:
        """Check if function is available"""
        return name in self.available_functions
    
    def get_all_functions(self) -> List[str]:
        """Get list of all available functions"""
        return list(self.available_functions.keys())
    
    def get_external_dependencies(self, functions: Set[str]) -> Set[str]:
        """Get external dependencies for a set of functions"""
        deps = set()
        for func in functions:
            if func in self.available_functions:
                info = self.available_functions[func]()
                deps.update(info.get('external_deps', []))
        return deps

# Global instance
stdlib = MinimalStdLib()