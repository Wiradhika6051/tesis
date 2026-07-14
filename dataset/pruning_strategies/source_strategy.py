from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy


class SourceStrategy(
    DatasetStrategy
):

    def extract(
        self,
        file,
        repo
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

        return [

            {
                "source": previous_source,

                "snippet": previous_source,

                "label": 1,
                "diff": file["diff"],
                "repo": repo
            },

            {
                "source": current_source,

                "snippet": current_source,

                "label": 0,
                "diff": file["diff"],
                "repo": repo
            }

        ]