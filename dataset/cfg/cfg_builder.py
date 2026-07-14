import ast
import json
from platform import node

from tesis.dataset.cfg.cfg_node import CFGNode

class CFGBuilder:

    def __init__(self):

        self.nodes = []
        self.edges = []

        self.counter = 0

        self.reset_statistics()

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

        first_entry = None
        previous_exit = None
        
        for stmt in statements:
        
            entry, exit = self.process_stmt(stmt)
        
            if entry is None:
                continue
            
            if first_entry is None:
                first_entry = entry
        
            if previous_exit is not None:
                self.edges.append(
                    (
                        previous_exit,
                        entry
                    )
                )
        
            previous_exit = exit
        
        return first_entry, previous_exit
    
    def process_function(
        self,
        stmt
    ):

        function = self.add_node(stmt)

        if stmt.body:

            start, end = self.process_block(
                stmt.body
            )

            if start is not None:

                self.edges.append(
                    (
                        function,
                        start
                    )
                )

            return function, end

        return function, function
        
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

        return if_node, merge_node

    def process_while(
        self,
        stmt
    ):
        while_node = self.add_node(stmt)
        exit_node = self.add_virtual_node(
            "LOOP_EXIT"
        )
        if stmt.body:
        
            body_start, body_end = self.process_block(
                stmt.body
            )
            self.edges.append(
                (
                    while_node,
                    body_start
                )
            )
            self.edges.append(
                (
                    body_end,
                    while_node
                )
            )
        #
        # Loop exit
        #
        self.edges.append(
            (
                while_node,
                exit_node
            )
        )
        return while_node, exit_node
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

        return loop, exit_node
   
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
        if isinstance(stmt, ast.FunctionDef) or isinstance(stmt, ast.AsyncFunctionDef):
            return self.process_function(
                stmt
            )

        node = self.add_node(stmt)

        return node, node
   
    def process_return(
        self,
        stmt
    ):

        node = self.add_node(stmt)

        return node, node

    def process_break(
        self,
        stmt
    ):

        node = self.add_node(stmt)

        return node, node
    
    def process_continue(
        self,
        stmt
    ):

        node = self.add_node(stmt)

        return node, node
    
    def process_try(
        self,
        stmt
    ):
    
        try_node = self.add_node(stmt)
    
        merge_node = self.add_virtual_node(
            "MERGE"
        )
    
        if stmt.body:
        
            start, end = self.process_block(
                stmt.body
            )
    
            self.edges.append(
                (
                    try_node,
                    start
                )
            )
    
            if end is not None:
            
                self.edges.append(
                    (
                        end,
                        merge_node
                    )
                )
    
        for handler in stmt.handlers:
        
            start, end = self.process_block(
                handler.body
            )
    
            self.edges.append(
                (
                    try_node,
                    start
                )
            )
    
            if end is not None:
            
                self.edges.append(
                    (
                        end,
                        merge_node
                    )
                )
    
        return try_node, merge_node

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
   
    def build(
        self,
        source
    ):

        self.nodes = []
        self.edges = []
        self.counter = 0
        self.quality_stats["total_files"] += 1

        try:

            tree = ast.parse(source)

        except SyntaxError as e:
            print(f"CFG parse failed: {e}")
            self.quality_stats["parse_fail"] += 1
            return None
        self.quality_stats["parse_success"] += 1
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

        if len(self.nodes) == 0:
            self.quality_stats["empty_cfg"] += 1

        self.update_graph_statistics()

        return {
            "nodes": self.nodes,
            "edges": self.edges
        }
    
    def reset_statistics(self):

        #
        # CFG quality
        #
        self.quality_stats = {

            "total_files": 0,

            "parse_success": 0,

            "parse_fail": 0,

            "empty_cfg": 0,
        }

        #
        # Graph statistics
        #
        self.graph_stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "max_nodes": 0,
            "max_edges": 0,
            "min_nodes": float("inf"),
            "min_edges": float("inf"),
            "total_target_nodes": 0,
            "target_detection_fail": 0,
            "total_pruned_nodes": 0,
            "total_pruned_edges": 0
        }

    def update_graph_statistics(
            self
        ):

        n_nodes = len(self.nodes)

        n_edges = len(self.edges)

        self.graph_stats["total_nodes"] += n_nodes

        self.graph_stats["total_edges"] += n_edges

        self.graph_stats["max_nodes"] = max(
            self.graph_stats["max_nodes"],
            n_nodes
        )

        self.graph_stats["max_edges"] = max(
            self.graph_stats["max_edges"],
            n_edges
        )

        self.graph_stats["min_nodes"] = min(
            self.graph_stats["min_nodes"],
            n_nodes
        )

        self.graph_stats["min_edges"] = min(
            self.graph_stats["min_edges"],
            n_edges
        )   

    def get_graph_stats(self):
    
        success = max(
            self.quality_stats["parse_success"],
            1
        )
    
        avg_nodes = (
            self.graph_stats["total_nodes"]
            / success
        )
    
        avg_edges = (
            self.graph_stats["total_edges"]
            / success
        )
    
        avg_pruned_nodes = (
            self.graph_stats["total_pruned_nodes"]
            / success
        )
    
        avg_pruned_edges = (
            self.graph_stats["total_pruned_edges"]
            / success
        )
    
        return {
        
            **self.graph_stats,
    
            "avg_nodes": avg_nodes,
    
            "avg_edges": avg_edges,
    
            "avg_pruned_nodes": avg_pruned_nodes,
    
            "avg_pruned_edges": avg_pruned_edges,
    
            "avg_target_nodes":
                self.graph_stats["total_target_nodes"]
                / success,
    
            "target_detection_rate":
                1
                -
                (
                    self.graph_stats["target_detection_fail"]
                    / success
                ),
    
            "node_reduction_ratio":
                avg_pruned_nodes
                / max(avg_nodes, 1),
    
            "edge_reduction_ratio":
                avg_pruned_edges
                / max(avg_edges, 1)
        }
     
    def get_quality_stats(self):
        total = max(self.quality_stats["total_files"], 1)

        return {
            **self.quality_stats,
            "parse_success_rate":
                self.quality_stats["parse_success"] / total,
            "parse_failure_rate":
                self.quality_stats["parse_fail"] / total
        }
    
    def print_report(
        self,
        use_pruning=True
    ):
    
        q = self.get_quality_stats()
        g = self.get_graph_stats()
    
        print("=" * 50)
        print("CFG Report")
        print("=" * 50)
    
        print("\nCFG Quality")
        print("-" * 30)
    
        for k, v in q.items():
            print(f"{k}: {v}")
    
        print("\nGraph Statistics")
        print("-" * 30)
    
        graph_keys = [
        
            "total_nodes",
            "total_edges",
    
            "avg_nodes",
            "avg_edges",
    
            "max_nodes",
            "max_edges",
    
            "min_nodes",
            "min_edges"
        ]
    
        for key in graph_keys:
        
            print(
                f"{key}: {g[key]}"
            )
    
        if use_pruning:
        
            print("\nTarget Detection")
            print("-" * 30)
    
            print(
                "total_target_nodes:",
                g["total_target_nodes"]
            )
    
            print(
                "target_detection_fail:",
                g["target_detection_fail"]
            )
    
            print(
                "target_detection_rate:",
                g["target_detection_rate"]
            )
    
            print("\nPruning")
            print("-" * 30)
    
            print(
                "total_pruned_nodes:",
                g["total_pruned_nodes"]
            )
    
            print(
                "total_pruned_edges:",
                g["total_pruned_edges"]
            )
    
            print(
                "avg_pruned_nodes:",
                g["avg_pruned_nodes"]
            )
    
            print(
                "avg_pruned_edges:",
                g["avg_pruned_edges"]
            )
    
            print(
                "node_reduction_ratio:",
                g["node_reduction_ratio"]
            )
    
            print(
                "edge_reduction_ratio:",
                g["edge_reduction_ratio"]
            )

    def save_report(
        self,
        output_file,
        use_pruning=True
    ):

        graph = self.get_graph_stats()

        report = {

            "quality": self.get_quality_stats(),

            "graph": {

                "total_nodes": graph["total_nodes"],
                "total_edges": graph["total_edges"],

                "avg_nodes": graph["avg_nodes"],
                "avg_edges": graph["avg_edges"],

                "max_nodes": graph["max_nodes"],
                "max_edges": graph["max_edges"],

                "min_nodes": graph["min_nodes"],
                "min_edges": graph["min_edges"]

            }

        }

        if use_pruning:

            report["target_detection"] = {

                "total_target_nodes":
                    graph["total_target_nodes"],

                "target_detection_fail":
                    graph["target_detection_fail"],

                "target_detection_rate":
                    graph["target_detection_rate"]

            }

            report["pruning"] = {

                "total_pruned_nodes":
                    graph["total_pruned_nodes"],

                "total_pruned_edges":
                    graph["total_pruned_edges"],

                "avg_pruned_nodes":
                    graph["avg_pruned_nodes"],

                "avg_pruned_edges":
                    graph["avg_pruned_edges"],

                "node_reduction_ratio":
                    graph["node_reduction_ratio"],

                "edge_reduction_ratio":
                    graph["edge_reduction_ratio"]

            }

        with open(
            output_file,
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

        print(
            f"CFG report saved to {output_file}"
        )

    def update_target_statistics(
        self,
        target_nodes
    ):

        self.graph_stats[
            "total_target_nodes"
        ] += len(target_nodes)

        if len(target_nodes) == 0:

            self.graph_stats[
                "target_detection_fail"
            ] += 1

    def update_pruning_statistics(
        self,
        before_nodes,
        after_nodes,
        before_edges,
        after_edges
    ):

        self.graph_stats[
            "total_pruned_nodes"
        ] += (
            before_nodes - after_nodes
        )

        self.graph_stats[
            "total_pruned_edges"
        ] += (
            before_edges - after_edges
        )

    def visualize(self,cfg):

        print("=" * 60)
        print("Nodes")
        print("=" * 60)

        for node in cfg["nodes"]:

            print(
                f"[{node.node_id}] "
                f"{node.node_type:<15}"
                f" line={node.lineno}"
            )

        print()

        print("=" * 60)
        print("Edges")
        print("=" * 60)

        for src, dst in cfg["edges"]:

            print(f"{src} -> {dst}")

    def to_dot(self,cfg):

        lines = []

        lines.append("digraph CFG {")

        for node in cfg["nodes"]:

            label = (
                f"{node.node_id}\\n"
                f"{node.node_type}\\n"
                f"L{node.lineno}"
            )

            lines.append(
                f'{node.node_id} [label="{label}"];'
            )

        for src, dst in cfg["edges"]:

            lines.append(
                f"{src} -> {dst};"
            )

        lines.append("}")

        return "\n".join(lines)

if __name__ == "__main__":

    source_sequential = """
def login(username):

    query = "SELECT * FROM user WHERE name='" + username + "'"

    cursor.execute(query)

    return query
"""
    source_if = """
def foo(x):

    a = 1

    if x:
        b = 2
    else:
        c = 3

    d = 4

    return d
"""
    source_while = """
def foo():

    while cond:
        a = 1

    return a
"""

    source_for = """
def foo():

    for i in x:
        a = 1
        b = 2

    return b
"""

    source = """
def foo():

    try:
        a = 1
    except:
        b = 2

    return 0
"""

    builder = CFGBuilder()

    cfg = builder.build(source)

    builder.visualize(cfg)

    # save to graphvix
    dot = builder.to_dot(cfg)

    with open("cfg.dot", "w") as f:

        f.write(dot)