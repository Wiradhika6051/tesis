import os
import shutil
import pickle
import subprocess

from git import Repo
from tqdm import tqdm


CHECKPOINT_INTERVAL = 500


def clone_repo(
    repo_url,
    repo_root
):

    repo_name = repo_url.rstrip("/").split("/")[-1]

    local_path = os.path.join(
        repo_root,
        repo_name
    )

    if os.path.exists(local_path):
        return local_path

    print(f"Cloning {repo_name}...")

    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            repo_url,
            local_path
        ],
        check=True
    )

    return local_path


def save_checkpoint(
    lookup,
    output_file
):

    with open(
        output_file,
        "wb"
    ) as f:

        pickle.dump(
            lookup,
            f
        )


def fetch_previous_sources(
    dataset,
    repo_root,
    output_file
):

    os.makedirs(
        repo_root,
        exist_ok=True
    )

    #
    # Resume if checkpoint exists
    #
    if os.path.exists(output_file):

        with open(
            output_file,
            "rb"
        ) as f:

            lookup = pickle.load(f)

        print(
            f"Loaded checkpoint with {len(lookup)} entries."
        )

    else:

        lookup = {}

    stats = {

        "repos": 0,
        "commits": 0,
        "files": 0,

        "success": 0,
        "missing": 0,

        "clone_fail": 0,
        "commit_fail": 0,
        "file_fail": 0

    }

    since_checkpoint = 0

    for repo_data in tqdm(
        dataset,
        desc="Repositories"
    ):

        repo_url = repo_data["repo"]

        repo_name = repo_url.rstrip("/").split("/")[-1]

        #
        # clone
        #
        try:

            repo_path = clone_repo(
                repo_url,
                repo_root
            )

            git_repo = Repo(
                repo_path
            )

            stats["repos"] += 1

        except Exception as e:

            print(
                f"Clone failed: {repo_name}"
            )

            print(e)

            stats["clone_fail"] += 1

            continue

        #
        # process commits
        #
        for commit in tqdm(
            repo_data["commits"],
            leave=False,
            desc=repo_name
        ):

            stats["commits"] += 1

            sha = commit["sha"]

            try:

                git_commit = git_repo.commit(
                    sha
                )

                if len(
                    git_commit.parents
                ) == 0:
                    continue

                parent = git_commit.parents[0]

            except Exception:

                stats["commit_fail"] += 1

                continue

            for file in commit["files"]:

                stats["files"] += 1

                filename = file[
                    "filename"
                ].lstrip("/")

                key = (
                    repo_url,
                    sha,
                    filename
                )

                #
                # Already fetched
                #
                if key in lookup:
                    continue

                try:

                    source = git_repo.git.show(
                        f"{parent.hexsha}:{filename}"
                    )

                    lookup[key] = {

                        "repo": repo_url,

                        "commit": sha,

                        "parent": parent.hexsha,

                        "filename": filename,

                        "source": source

                    }

                    stats["success"] += 1

                except Exception:

                    lookup[key] = None

                    stats["missing"] += 1

                    stats["file_fail"] += 1

                since_checkpoint += 1

                #
                # checkpoint
                #
                if since_checkpoint >= CHECKPOINT_INTERVAL:

                    save_checkpoint(
                        lookup,
                        output_file
                    )

                    print(
                        f"\nCheckpoint saved ({len(lookup)} files)"
                    )

                    since_checkpoint = 0

        #
        # remove repository after processing
        #
        try:

            shutil.rmtree(
                repo_path
            )

        except Exception:

            print(
                f"Unable to delete {repo_name}"
            )

    #
    # final save
    #
    save_checkpoint(
        lookup,
        output_file
    )

    print()

    print("=" * 60)

    print("Fetch Previous Source Report")

    print("=" * 60)

    for k, v in stats.items():

        print(
            f"{k:15}: {v}"
        )

    print()

    print(
        f"Lookup Size : {len(lookup)}"
    )

    print(
        f"Checkpoint  : {output_file}"
    )