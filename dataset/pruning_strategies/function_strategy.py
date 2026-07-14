import re
import ast
from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy

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
        repo,
        diff
    ):

        current_source = file.get(
            "sourceWithComments",
            file.get(
                "source",
                ""
            )
        )

        previous_source = file.get(
            "previousSource"
        )

        if (
            not current_source
            or
            not previous_source
        ):
            return []

        result = []

        for change in file.get(
            "changes",
            []
        ):

            for bad in change.get(
                "badparts",
                []
            ):
                result.append({
                    "source": previous_source,
                    "snippet": bad,
                    "label": 1,
                    "repo": repo,
                    "diff": diff,
                })

            for good in change.get(
                "goodparts",
                []
            ):
                result.append({
                    "source": current_source,
                    "snippet": good,
                    "label": 0,
                    "repo": repo,
                    "diff": diff,
                })

        return result
