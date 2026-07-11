from tqdm import tqdm

from git_utils import GitRepository
from statistics import Statistics


def fetch_previous_sources(
    dataset,
    repo_root,
    checkpoint,
    report_file
):

    stats = Statistics()

    for repo_data in tqdm(
        dataset,
        desc="Repositories"
    ):

        repo_url = repo_data["repo"]

        repo_name = (
            repo_url
            .rstrip("/")
            .split("/")[-1]
        )

        #
        # Skip if already processed
        #
        if checkpoint.repository_finished(
            repo_name
        ):

            stats.increment(
                "repositories_skipped"
            )

            continue

        repo_lookup = {}

        git = GitRepository(
            repo_url,
            repo_root
        )

        try:

            git.clone()

            stats.increment(
                "repositories"
            )

        except Exception as e:

            print(
                f"Clone failed: {repo_name}"
            )

            print(e)

            stats.increment(
                "repositories_failed"
            )

            stats.increment(
                "clone_failures"
            )

            continue

        #
        # Process commits
        #
        for commit in tqdm(
            repo_data["commits"],
            leave=False,
            desc=repo_name
        ):

            stats.increment(
                "commits"
            )

            sha = commit["sha"]

            parent_sha = git.get_parent_sha(
                sha
            )

            if parent_sha is None:

                stats.increment(
                    "commit_failures"
                )

                continue

            #
            # Process files
            #
            for filename in commit.get(
                "files",
                {}
            ):

                stats.increment(
                    "files"
                )

                filename = filename.lstrip(
                    "/"
                )

                key = (
                    repo_url,
                    sha,
                    filename
                )

                #
                # Already fetched
                #
                if key in repo_lookup:

                    continue

                source = git.get_previous_source(
                    parent_sha,
                    filename
                )

                if source is None:

                    repo_lookup[key] = None

                    stats.increment(
                        "files_missing"
                    )

                    stats.increment(
                        "file_failures"
                    )

                    continue

                repo_lookup[key] = {

                    "repo": repo_url,

                    "commit": sha,

                    "parent": parent_sha,

                    "filename": filename,

                    "source": source

                }

                stats.increment(
                    "files_success"
                )

        #
        # Save checkpoint
        #
        checkpoint.save_repository(
            repo_name,
            repo_lookup
        )

        #
        # Cleanup
        #
        git.clear_cache()

        git.delete()

    #
    # Save statistics
    #
    stats.save(
        report_file
    )

    stats.print_report()