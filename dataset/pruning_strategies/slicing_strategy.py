from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy
from tesis.dataset.pruning_strategies.source_strategy import reconstruct_fixed_source
from tesis.dataset.tokenizer import tokenize_python


class SlicingStrategy(
    DatasetStrategy
):

    def __init__(
        self,
        context=100
    ):
        self.context = context

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

                snippet = extract_token_slice(
                    source,
                    bad,
                    context=self.context
                )

                if snippet:

                    result.append({
                        "code": snippet,
                        "label": 1,
                        "repo": repo,
                        "diff": file["diff"],
                    })

            for good in change.get(
                "goodparts",
                []
            ):

                snippet = extract_token_slice(
                    fixed_source,
                    good,
                    context=self.context
                )

                if snippet:

                    result.append({
                        "code": snippet,
                        "label": 0,
                        "repo": repo
                    })

        return result


def extract_token_slice(
    source,
    snippet,
    context=100
):
    """
    context = tokens before/after
    """

    source_tokens = tokenize_python(
        source
    )

    snippet_tokens = tokenize_python(
        snippet
    )

    if not snippet_tokens:
        return None

    target_len = len(
        snippet_tokens
    )

    for i in range(
        len(source_tokens)
    ):

        if (
            source_tokens[i:i+target_len]
            ==
            snippet_tokens
        ):

            start = max(
                0,
                i - context
            )

            end = min(
                len(source_tokens),
                i + target_len + context
            )

            return " ".join(
                source_tokens[start:end]
            )

    return None