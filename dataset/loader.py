import json
import pickle



def load_samples(jsonl_path, strategy):
    with open(
        "dataset/previous_sources.pkl",
        "rb"
    ) as f:
        previous_lookup = pickle.load(f)

    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line in f:

            data = json.loads(line)

            for repo in data.values():

                for commit in repo.values():
                
                    files = commit.get("files", {})

                    for file in files.values():
                        file["previousSource"] = previous_lookup.get(                    
                            (
                                repo,
                                commit["sha"],
                                file["filename"]
                            )
                        )
                        samples.extend(strategy.extract(file,next(iter(data))))

    return samples