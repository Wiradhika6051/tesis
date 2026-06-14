import json


def load_samples(jsonl_path):

    samples = []

    with open(
        jsonl_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            data = json.loads(line)

            for repo in data.values():

                for commit in repo.values():

                    files = commit.get(
                        "files",
                        {}
                    )

                    for file in files.values():

                        for change in file.get(
                            "changes",
                            []
                        ):

                            for code in change.get(
                                "badparts",
                                []
                            ):

                                samples.append({
                                    "code": code,
                                    "label": 1
                                })

                            for code in change.get(
                                "goodparts",
                                []
                            ):

                                samples.append({
                                    "code": code,
                                    "label": 0
                                })

    return samples