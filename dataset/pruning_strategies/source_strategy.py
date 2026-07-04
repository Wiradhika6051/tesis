from tesis.dataset.pruning_strategies.base_strategy import DatasetStrategy

class SourceStrategy(DatasetStrategy):

    def extract(self, file, repo):
        current_source = file.get(
            "sourceWithComments"
        )

        previous_source = file.get(
            "previousSource"
        )

        if previous_source is None:
            return []

        return [
            {
                "code": previous_source,
                "label": 1,
                "repo": repo
            },
            {
                "code": current_source,
                "label": 0,
                "repo": repo
            }
        ]
    
def reconstruct_fixed_source(source, changes):
    """
    Reconstruct fixed source code by replacing
    vulnerable snippets with patched snippets.
    """
    fixed_source = source

    for change in changes:
    
        badparts = change.get(
            "badparts",
            []
        )

        goodparts = change.get(
            "goodparts",
            []
        )

        pairs = min(
            len(badparts),
            len(goodparts)
        )

        for bad, good in zip(
            badparts,
            goodparts
        ):

            bad = bad.strip()
            good = good.strip()

            if not bad:
                continue
            
            fixed_source = fixed_source.replace(
                bad,
                good,
                1
            )

    return fixed_source