class CFGNode:

    def __init__(
        self,
        node_id,
        lineno,
        node_type,
        text,
        end_lineno=None
    ):

        self.node_id = node_id

        self.lineno = lineno

        self.end_lineno = (
            end_lineno
            if end_lineno is not None
            else lineno
        )

        self.node_type = node_type

        self.text = text

    def __repr__(
        self
    ):

        return (
            f"{self.node_id}:"
            f"{self.node_type}@"
            f"{self.lineno}"
        )