from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy

class SourceStrategy(
    DatasetStrategy
):

    def extract(
        self,
        file
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

        return [
            {
                "code": source,
                "label": 1
            }
        ]