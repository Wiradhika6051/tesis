import json

def load_samples(jsonl_path):

    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line in f:

            data = json.loads(line)

            for repo in data.values():

                for commit in repo.values():

                    files = commit.get(
                        "files",
                        {}
                    )

                    for file_name, file_data in files.items():

                        source = file_data.get(
                            "sourceWithComments",
                            file_data.get("source", "")
                        )

                        if not source.strip():
                            continue

                        samples.append({
                            "code": source,
                            "label": 1,
                            "file": file_name
                        })

    return samples