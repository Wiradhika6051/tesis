from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy


class DiffStrategy(
    DatasetStrategy
):

    def extract(
        self,
        file,
        repo,
        diff
    ):

        samples = []

        for change in file.get(
            "changes",
            []
        ):

            #
            # Vulnerable snippet
            #
            for snippet in change.get(
                "badparts",
                []
            ):

                samples.append({

                    "source": snippet,

                    "snippet": snippet,
                    "diff": diff,
                    "label": 1,

                    "repo": repo

                })

            #
            # Fixed snippet
            #
            for snippet in change.get(
                "goodparts",
                []
            ):

                samples.append({

                    "source": snippet,

                    "snippet": snippet,
                    "diff": diff,
                    "label": 0,

                    "repo": repo

                })

        return samples