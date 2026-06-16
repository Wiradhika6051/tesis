from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy

class DiffStrategy(
    DatasetStrategy
):

    def extract(
        self,
        file,
        repo
    ):

        samples = []

        for change in file.get(
            "changes",
            []
        ):

            for snippet in change.get(
                "badparts",
                []
            ):

                samples.append({
                    "code": snippet,
                    "label": 1,
                    "repo": repo
                })

            for snippet in change.get(
                "goodparts",
                []
            ):

                samples.append({
                    "code": snippet,
                    "label": 0,
                    "repo": repo
                })

        return samples