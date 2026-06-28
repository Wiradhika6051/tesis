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

        first = None
        previous = None

        for stmt in statements:

            current = self.process_stmt(stmt)

            if current is None:
                continue

            if first is None:
                first = current

            if previous is not None:

                self.edges.append(
                    (
                        previous,
                        current
                    )
                )

            previous = current

        return first, previous
    
    def old_process_block(
        self,
        statements
    ):
    
        previous = None
        first = None
    
        for stmt in statements:
        
            current = self.process_stmt(
                stmt
            )
    
            if current is None:
                continue
            
            if first is None:
                first = current
    
            if previous is not None:
            
                self.edges.append(
                    (
                        previous,
                        current
                    )
                )
    
            previous = current
    
        return first
    
    def process_if(
        self,
        stmt
    ):

        if_node = self.add_node(stmt)

        merge_node = self.add_virtual_node(
            "MERGE"
        )

        #
        # True branch
        #
        if stmt.body:

            body_start, body_end = self.process_block(
                stmt.body
            )

            self.edges.append(
                (
                    if_node,
                    body_start
                )
            )

            if body_end is not None:

                self.edges.append(
                    (
                        body_end,
                        merge_node
                    )
                )

        else:

            self.edges.append(
                (
                    if_node,
                    merge_node
                )
            )

        #
        # False branch
        #
        if stmt.orelse:

            else_start, else_end = self.process_block(
                stmt.orelse
            )

            self.edges.append(
                (
                    if_node,
                    else_start
                )
            )

            if else_end is not None:

                self.edges.append(
                    (
                        else_end,
                        merge_node
                    )
                )

        else:

            self.edges.append(
                (
                    if_node,
                    merge_node
                )
            )

        return merge_node
    
    def old_process_if(
        self,
        stmt
    ):

        if_id = self.add_node(
            stmt
        )

        if stmt.body:

            body_start = self.process_block(
                stmt.body
            )

            self.edges.append(
                (
                    if_id,
                    body_start
                )
            )

        if stmt.orelse:

            else_start = self.process_block(
                stmt.orelse
            )

            self.edges.append(
                (
                    if_id,
                    else_start
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

        loop = self.add_node(stmt)

        exit_node = self.add_virtual_node(
            "LOOP_EXIT"
        )

        if stmt.body:

            start, end = self.process_block(
                stmt.body
            )

            self.edges.append(
                (
                    loop,
                    start
                )
            )

            self.edges.append(
                (
                    end,
                    loop
                )
            )

        #
        # loop exit
        #
        self.edges.append(
            (
                loop,
                exit_node
            )
        )

        return exit_node

    def old_process_for(
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

        if isinstance(stmt, ast.If):
            return self.process_if(stmt)

        if isinstance(stmt, ast.For):
            return self.process_for(stmt)

        if isinstance(stmt, ast.While):
            return self.process_while(stmt)

        if isinstance(stmt, ast.Try):
            return self.process_try(stmt)

        if isinstance(stmt, ast.Return):
            return self.process_return(stmt)

        if isinstance(stmt, ast.Break):
            return self.process_break(stmt)

        if isinstance(stmt, ast.Continue):
            return self.process_continue(stmt)

        return self.add_node(stmt)

    def old_process_stmt(
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
    
    def process_return(
        self,
        stmt
    ):

        return self.add_node(stmt)

    def process_break(
        self,
        stmt
    ):

        return self.add_node(stmt)
    
    def process_continue(
        self,
        stmt
    ):

        return self.add_node(stmt)
    
    def process_try(
        self,
        stmt
    ):

        try_node = self.add_node(stmt)

        if stmt.body:

            start, _ = self.process_block(
                stmt.body
            )

            self.edges.append(
                (
                    try_node,
                    start
                )
            )

        for handler in stmt.handlers:

            start, _ = self.process_block(
                handler.body
            )

            self.edges.append(
                (
                    try_node,
                    start
                )
            )

        return try_node
    
    def add_virtual_node(
        self,
        node_type
    ):

        node = CFGNode(
            node_id=self.counter,
            lineno=-1,
            node_type=node_type,
            text=node_type
        )

        self.counter += 1

        self.nodes.append(node)

        return node.node_id

    def old_build(
        self,
        source
    ):
    
        self.nodes = []
        self.edges = []
        self.counter = 0
    
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
    
    def build(
        self,
        source
    ):
    
        self.nodes = []
        self.edges = []
        self.counter = 0
    
        try:
        
            tree = ast.parse(source)
    
        except SyntaxError as e:
        
            print(f"CFG parse failed: {e}")
    
            return None
    
        entry = self.add_virtual_node("ENTRY")
    
        start, end = self.process_block(
            tree.body
        )
    
        exit_node = self.add_virtual_node("EXIT")
    
        if start is not None:
        
            self.edges.append(
                (
                    entry,
                    start
                )
            )
    
        if end is not None:
        
            self.edges.append(
                (
                    end,
                    exit_node
                )
            )
    
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }