from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class TokenType(Enum):
    # Control Flow Keywords
    IF = "IF"
    ELIF = "ELIF"
    ELSE = "ELSE"
    ENDIF = "ENDIF"
    WHILE = "WHILE"
    ENDWHILE = "ENDWHILE"
    DO = "DO"
    FOR = "FOR"
    ENDFOR = "ENDFOR"
    SWITCH = "SWITCH"
    CASE = "CASE"
    DEFAULT = "DEFAULT"
    ENDSWITCH = "ENDSWITCH"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    RETURN = "RETURN"
    GOTO = "GOTO"
    
    # Declaration Keywords
    VAR = "VAR"
    CONST = "CONST"
    STATIC = "STATIC"
    EXTERN = "EXTERN"
    VOLATILE = "VOLATILE"
    REGISTER = "REGISTER"
    INLINE = "INLINE"
    FUNCTION = "FUNCTION"
    PROCEDURE = "PROCEDURE"
    STRUCT = "STRUCT"
    UNION = "UNION"
    ENUM = "ENUM"
    TYPEDEF = "TYPEDEF"
    MACRO = "MACRO"
    ENDMACRO = "ENDMACRO"
    
    # ... (rest omitted in favor of importing only necessary parts)

@dataclass
class SourceLocation:
    line: int
    column: int
    file: str = "<stdin>"
    raw_line: str = ""
    end_column: Optional[int] = None
    
    def __str__(self):
        return f"{self.file}:{self.line}:{self.column}"

@dataclass
class Token:
    type: TokenType
    value: str
    location: SourceLocation
    leading_whitespace: str = ""
    trailing_whitespace: str = ""
    
    def __str__(self):
        return f"{self.type.name}('{self.value}') @ {self.location}"
