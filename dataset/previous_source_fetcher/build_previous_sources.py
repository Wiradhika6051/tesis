import json

from previous_source_fetcher import (
    fetch_previous_sources
)

from checkpoint import (
    CheckpointManager
)


DATASET = "../../data/plain_xss.jsonl"

REPOSITORY_ROOT = "../../data/repositories"

CHECKPOINT_DIR = "../../data/checkpoints"

OUTPUT_FILE = "../../data/previous_sources.pkl"

REPORT_FILE = "../../data/previous_sources_report.json"



def load_dataset(jsonl_path):
    repositories = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            data = json.loads(line)

            repo_url = data["repo_url"]
            commit_sha = data.get("commit_sha") or data["sha"]

            if repo_url not in repositories:
                repositories[repo_url] = {
                    "repo": repo_url,
                    "commits": [],
                    "_seen_commits": set(),
                }

            repo = repositories[repo_url]

            # Avoid adding the same commit twice.
            if commit_sha in repo["_seen_commits"]:
                continue

            repo["_seen_commits"].add(commit_sha)

            repo["commits"].append({
                "sha": commit_sha,
                "files": data.get("files", {}),
            })

    # Remove internal deduplication metadata.
    dataset = []

    for repo in repositories.values():
        repo.pop("_seen_commits")
        dataset.append(repo)

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