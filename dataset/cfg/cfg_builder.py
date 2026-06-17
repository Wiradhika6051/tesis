import ast

from tesis.dataset.cfg.cfg_node import CFGNode

class CFGBuilder:

    def __init__(self):

        self.nodes = []
        self.edges = []

        self.counter = 0

    def add_node(
        self,
        ast_node
    ):

        node = CFGNode(
            node_id=self.counter,
            lineno=getattr(
                ast_node,
                "lineno",
                -1
            ),
            node_type=type(
                ast_node
            ).__name__,
            text=ast.dump(
                ast_node
            )
        )

        self.counter += 1

        self.nodes.append(node)

        return node.node_id
    
    def process_block(
        self,
        statements
    ):

        previous = None

        for stmt in statements:

            current = self.process_stmt(
                stmt
            )

            if (
                previous is not None
                and
                current is not None
            ):
                self.edges.append(
                    (
                        previous,
                        current
                    )
                )

            previous = current

    def process_if(
        self,
        stmt
    ):

        if_id = self.add_node(
            stmt
        )

        if stmt.body:

            first_true = self.process_stmt(
                stmt.body[0]
            )

            self.edges.append(
                (
                    if_id,
                    first_true
                )
            )

        if stmt.orelse:

            first_false = self.process_stmt(
                stmt.orelse[0]
            )

            self.edges.append(
                (
                    if_id,
                    first_false
                )
            )

        return if_id
    
    def process_while(
        self,
        stmt
    ):

        while_id = self.add_node(
            stmt
        )

        if stmt.body:

            first_body = self.process_stmt(
                stmt.body[0]
            )

            self.edges.append(
                (
                    while_id,
                    first_body
                )
            )

            self.edges.append(
                (
                    first_body,
                    while_id
                )
            )

        return while_id
    
    def process_for(
        self,
        stmt
    ):

        for_id = self.add_node(
            stmt
        )

        if stmt.body:

            first_body = self.process_stmt(
                stmt.body[0]
            )

            self.edges.append(
                (
                    for_id,
                    first_body
                )
            )

            self.edges.append(
                (
                    first_body,
                    for_id
                )
            )

        return for_id
    
    def process_stmt(
        self,
        stmt
    ):

        if isinstance(
            stmt,
            ast.If
        ):
            return self.process_if(
                stmt
            )

        if isinstance(
            stmt,
            ast.For
        ):
            return self.process_for(
                stmt
            )

        if isinstance(
            stmt,
            ast.While
        ):
            return self.process_while(
                stmt
            )

        return self.add_node(
            stmt
        )
    
    def build(
        self,
        source
    ):

        try:
        
            tree = ast.parse(
                source
            )
    
        except Exception as e:
        
            print(
                f"CFG parse failed: {e}"
            )
    
            return None
    
        self.process_block(
            tree.body
        )
    
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }