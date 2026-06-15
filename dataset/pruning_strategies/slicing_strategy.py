from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy
from tesis.dataset.pruning_strategies.source_strategy import reconstruct_fixed_source

class SlicingStrategy(
    DatasetStrategy
):

    def extract(
        self,
        file
    ):

        source = file.get(
            "sourceWithComments",
            file.get("source", "")
        )

        if not source:
            return []

        result = []

        fixed_source = reconstruct_fixed_source(
            source,
            file.get(
                "changes",
                []
            )
        )

        for change in file.get(
            "changes",
            []
        ):

            for bad in change.get(
                "badparts",
                []
            ):

                snippet = extract_slice(
                    source,
                    bad,
                    context=5
                )

                if snippet:

                    result.append({
                        "code": snippet,
                        "label": 1
                    })

            for good in change.get(
                "goodparts",
                []
            ):

                snippet = extract_slice(
                    fixed_source,
                    good,
                    context=5
                )

                if snippet:

                    result.append({
                        "code": snippet,
                        "label": 0
                    })

        return result


def extract_slice(
    source,
    snippet,
    context=5
):

    lines = source.splitlines()

    target = snippet.strip()

    for i, line in enumerate(lines):

        if target in line:

            start = max(
                0,
                i - context
            )

            end = min(
                len(lines),
                i + context
            )

            return "\n".join(
                lines[start:end]
            )

    return None