import json

def load_samples(jsonl_path):

    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line in f:

            data = json.loads(line)

            for repo in data.values():

                for commit in repo.values():
                
                    files = commit.get("files", {})

                    for file in files.values():
                    
                        source = file.get(
                            "sourceWithComments",
                            file.get("source", "")
                        )

                        if not source:
                            continue
                        
                        changes = file.get(
                            "changes",
                            []
                        )

                        fixed_source = reconstruct_fixed_source(
                            source,
                            changes
                        )

                        if fixed_source != source:
                            # positive
                            samples.append({
                                "code": source,
                                "label": 1
                            })
    
                            # negative
                            samples.append({
                                "code": fixed_source,
                                "label": 0
                            })

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