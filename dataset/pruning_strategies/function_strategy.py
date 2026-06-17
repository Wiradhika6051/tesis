import re
import ast
from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy
from tesis.dataset.pruning_strategies.source_strategy import reconstruct_fixed_source

FUNCTION_STATS = {
    "parse_fail": 0,
    "snippet_not_found": 0,
    "function_not_found": 0,
    "success": 0
}

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

                function = extract_function(
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

                function = extract_function(
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

def extract_function(
    source,
    snippet
):

    line_no = find_snippet_line(
        source,
        snippet
    )

    if line_no is None:

        FUNCTION_STATS[
            "snippet_not_found"
        ] += 1

        return None

    function = find_function_by_line(
        source,
        line_no
    )

    if function is None:

        FUNCTION_STATS[
            "function_not_found"
        ] += 1

        return None

    FUNCTION_STATS[
        "success"
    ] += 1

    return function

def find_function_by_line(
    source,
    line_no
):

    try:

        tree = ast.parse(
            source
        )

    except Exception:

        FUNCTION_STATS[
            "parse_fail"
        ] += 1

        return None

    lines = source.splitlines()

    best_function = None

    best_size = None

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

        if start <= line_no <= end:

            size = end - start

            if (
                best_size is None
                or
                size < best_size
            ):

                best_size = size

                best_function = "\n".join(
                    lines[start - 1:end]
                )

    return best_function

def find_snippet_line(
    source,
    snippet
):

    source_lines = source.splitlines()

    normalized_snippet = re.sub(
        r"\s+",
        "",
        snippet
    )

    for idx, line in enumerate(source_lines):

        normalized_line = re.sub(
            r"\s+",
            "",
            line
        )

        if normalized_snippet in normalized_line:

            return idx + 1

    return None