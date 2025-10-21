class TranspileError(Exception):
    def __init__(self, message: str, line: int = 0, context: str = ""):
        self.message = message
        self.line = line
        self.context = context
        super().__init__(f"Line {line}: {message}" if line else message)
