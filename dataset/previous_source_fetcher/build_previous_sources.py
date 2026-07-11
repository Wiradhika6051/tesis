import json

from previous_source_fetcher import (
    fetch_previous_sources
)

from checkpoint import (
    CheckpointManager
)


DATASET = "../data/plain_sql.jsonl"

REPOSITORY_ROOT = "../data/repositories"

CHECKPOINT_DIR = "../data/checkpoints"

OUTPUT_FILE = "../data/previous_sources.pkl"

REPORT_FILE = "../data/previous_sources_report.json"


def load_dataset(
    jsonl_path
):

    dataset = []

    with open(
        jsonl_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            data = json.loads(
                line
            )

            for repo_url, repo in data.items():

                commits = []

                for commit in repo.values():

                    commits.append({

                        "sha": commit["sha"],

                        "files": commit.get(
                            "files",
                            {}
                        )

                    })

                dataset.append({

                    "repo": repo_url,

                    "commits": commits

                })

    return dataset


def main():

    print("=" * 60)

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET
    )

    print(
        f"Repositories : {len(dataset)}"
    )

    checkpoint = CheckpointManager(
        CHECKPOINT_DIR
    )

    fetch_previous_sources(

        dataset=dataset,

        repo_root=REPOSITORY_ROOT,

        checkpoint=checkpoint,

        report_file=REPORT_FILE

    )

    print()

    print("=" * 60)

    print("Merging repository checkpoints...")

    merged = checkpoint.merge(
        OUTPUT_FILE
    )

    print(
        f"Saved {len(merged)} previous sources."
    )

    print()

    print("=" * 60)

    print("Finished!")

    print(f"Output : {OUTPUT_FILE}")

    print(f"Report : {REPORT_FILE}")


if __name__ == "__main__":

    main()