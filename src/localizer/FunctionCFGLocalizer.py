import ast

from src.localizer.CFGLocalizer import CFGLocalizer


class FunctionCFGLocalizer(CFGLocalizer):

    def localize(
        self,
        sample
    ):

        tree = ast.parse(
            sample.source
        )

        target_function = None

        #
        # Find the function containing
        # one of the changed lines.
        #
        for node in ast.walk(tree):

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef
                )
            ):
                continue

            start = node.lineno

            end = getattr(
                node,
                "end_lineno",
                start
            )

            for line in sample.seed_lines:

                if start <= line <= end:

                    target_function = (
                        start,
                        end
                    )

                    break

            if target_function is not None:
                break

        #
        # Module-level change.
        #
        if target_function is None:

            sample.function_nodes = [

                node.node_id

                for node in sample.cfg["nodes"]

            ]

            return sample

        start, end = target_function

        sample.function_nodes = [

            node.node_id

            for node in sample.cfg["nodes"]

            if start <= node.lineno <= end

        ]

        return sample