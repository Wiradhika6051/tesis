import json
import pickle


def load_samples(
    jsonl_path,
    strategy,
    previous_source_path=None
):

    if previous_source_path:

        with open(previous_source_path, "rb") as f:
            previous_lookup = pickle.load(f)

    else:

        previous_lookup = {}

    samples = []

    with open(
        jsonl_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            data = json.loads(line)

            for repo_url, repo in data.items():

                for commit in repo.values():

                    files = commit.get(
                        "files",
                        {}
                    )

                    for file in files.values():

                        filename = file[
                            "filename"
                        ].lstrip("/")

                        key = (
                            repo_url,
                            commit["sha"],
                            filename
                        )

                        previous = previous_lookup.get(
                            key
                        )

                        file["previousSource"] = (
                            previous["source"]
                            if previous is not None
                            else None
                        )

                        samples.extend(

                            strategy.extract(
                                file,
                                repo_url
                            )

                        )

    return samples