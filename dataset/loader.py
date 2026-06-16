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
                        samples.extend(strategy.extract(file,next(iter(data))))

    return samples