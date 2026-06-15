import json

def load_samples(jsonl_path, strategy):

    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line in f:

            data = json.loads(line)

            for repo in data.values():

                for commit in repo.values():
                
                    files = commit.get("files", {})

                    for file in files.values():
                        samples.extend(strategy.extract(file))

    return samples

def reconstruct_fixed_source(source, changes):
    """
    Reconstruct fixed source code by replacing
    vulnerable snippets with patched snippets.
    """

    fixed_source = source

    for change in changes:

        badparts = change.get("badparts", [])
        goodparts = change.get("goodparts", [])

        pairs = min(
            len(badparts),
            len(goodparts)
        )

        for i in range(pairs):

            bad = badparts[i].strip()
            good = goodparts[i].strip()

            if bad and bad in fixed_source:
                fixed_source = fixed_source.replace(
                    bad,
                    good,
                    1
                )

    return fixed_source