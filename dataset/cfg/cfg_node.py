class CFGNode:

    def __init__(
        self,
        node_id,
        lineno,
        node_type,
        end_lineno=None,
        text=""
    ):
        self.node_id = node_id
        self.lineno = lineno
        self.node_type = node_type
        self.text = text
        self.end_lineno = end_lineno

    def __repr__(self):
        return (
            f"{self.node_id}:"
            f"{self.node_type}"
            f"@{self.lineno}"
        )