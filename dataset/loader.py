import json
import pickle
import os


def load_samples(
    jsonl_path,
    strategy,
    checkpoint_dir=None
):

    samples = []

    total_repositories = 0
    used_repositories = 0
    skipped_repositories = 0

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

                total_repositories += 1

                #
                # Load repository checkpoint
                #
                previous_lookup = {}

                if checkpoint_dir is not None:

                    repo_name = (
                        repo_url
                        .rstrip("/")
                        .split("/")[-1]
                    )

                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"{repo_name}.pkl"
                    )

                    if not os.path.exists(
                        checkpoint_path
                    ):

                        skipped_repositories += 1

                        print(
                            f"Skipping {repo_name} "
                            "(checkpoint not found)"
                        )

                        continue

                    with open(
                        checkpoint_path,
                        "rb"
                    ) as pf:

                        previous_lookup = pickle.load(
                            pf
                        )

                used_repositories += 1

                #
                # Process commits
                #
                for commit in repo.values():

                    files = commit.get(
                        "files",
                        {}
                    )

                    for filename, file in files.items():

                        filename = filename.lstrip("/")

                        #
                        # Some preprocessing may already
                        # have inserted filename.
                        #
                        file["filename"] = filename

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

    print("=" * 50)

    print("Dataset Loading Report")

    print("=" * 50)

    print(
        f"Repositories total   : {total_repositories}"
    )

    print(
        f"Repositories used    : {used_repositories}"
    )

    print(
        f"Repositories skipped : {skipped_repositories}"
    )

    print(
        f"Samples generated    : {len(samples)}"
    )

    return samples