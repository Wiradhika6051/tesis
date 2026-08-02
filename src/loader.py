from typing import List
import json
import pickle
import os

from src.type.Sample import Sample
from src.type.GitChange import GitChange


def load_samples(
    jsonl_path,
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

            data = json.loads(line)

            for repo_url, repo in data.items():

                total_repositories += 1

                #
                # Load repository checkpoint
                #
                lookup = {}

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

                        lookup = pickle.load(pf)

                used_repositories += 1

                #
                # Process commits
                #
                for commit in repo.values():

                    for filename in commit.get(
                        "files",
                        {}
                    ):

                        filename = filename.lstrip("/")

                        key = (
                            repo_url,
                            commit["sha"],
                            filename
                        )

                        record = lookup.get(key)

                        if record is None:
                            continue

                        if (
                            record["vulnerable_source"] is None
                            or
                            record["fixed_source"] is None
                        ):
                            continue

                        change = GitChange(

                            repo=repo_url,

                            parent_commit=record["parent"],

                            commit_hash=record["commit"],

                            file_path=filename,

                            previous_source=record["vulnerable_source"],

                            current_source=record["fixed_source"],

                            diff=commit["diff"]

                        )

                        samples.extend(
                            build_samples(change)
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


def build_samples(
    change: GitChange
) -> List[Sample]:

    vulnerable = Sample(

        repo=change.repo,

        parent_commit=change.parent_commit,

        commit_hash=change.parent_commit,

        file_path=change.file_path,

        source=change.previous_source,

        diff=change.diff,

        label=1,

    )

    fixed = Sample(

        repo=change.repo,

        parent_commit=change.parent_commit,

        commit_hash=change.commit_hash,

        file_path=change.file_path,

        source=change.current_source,

        diff=change.diff,

        label=0,

    )

    return [
        vulnerable,
        fixed
    ]