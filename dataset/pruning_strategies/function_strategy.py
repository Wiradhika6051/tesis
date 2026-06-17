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
            file.get(
                "source",
                ""
            )
        )

        if not source:
            return []

        result = []

        for change in file.get(
            "changes",
            []
        ):

            fixed_source = reconstruct_fixed_source(
                source,
                [change]
            )

            for bad in change.get(
                "badparts",
                []
            ):

                result.append({
                    "source": source,
                    "snippet": bad,
                    "label": 1,
                    "repo": repo
                })

            for good in change.get(
                "goodparts",
                []
            ):

                result.append({
                    "source": fixed_source,
                    "snippet": good,
                    "label": 0,
                    "repo": repo
                })

        return result
