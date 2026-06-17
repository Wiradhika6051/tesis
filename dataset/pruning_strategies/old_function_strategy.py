from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy
from tesis.dataset.pruning_strategies.source_strategy import reconstruct_fixed_source

class FunctionStrategy(
    DatasetStrategy
):

    def extract(
        self,
        file,
        repo
    ):

        source = file.get(
            "sourceWithComments",
            file.get("source", "")
        )

        if not source:
            return []

        result = []

        for change in file.get(
            "changes",
            []
        ):

            badparts = change.get(
                "badparts",
                []
            )

            goodparts = change.get(
                "goodparts",
                []
            )

            for bad in badparts:

                function = find_function_containing_snippet(
                    source,
                    bad
                )

                if function:

                    result.append({
                        "code": function,
                        "label": 1,
                        "repo": repo
                    })

            fixed_source = reconstruct_fixed_source(
                source,
                [change]
            )

            for good in goodparts:

                function = find_function_containing_snippet(
                    fixed_source,
                    good
                )

                if function:

                    result.append({
                        "code": function,
                        "label": 0,
                        "repo": repo
                    })

        return result
    

import ast


def find_function_containing_snippet(
    source,
    snippet
):
    try:
        tree = ast.parse(source)
    except:
        return None

    lines = source.splitlines()

    for node in ast.walk(tree):

        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue

        start = node.lineno - 1

        end = getattr(
            node,
            "end_lineno",
            start + 1
        )

        function_source = "\n".join(
            lines[start:end]
        )

        if snippet.strip() in function_source:
            return function_source

    return None