import os
import shutil
import subprocess

from git import Repo


class GitRepository:

    def __init__(
        self,
        repo_url,
        repo_root
    ):

        self.repo_url = repo_url

        self.repo_name = (
            repo_url
            .rstrip("/")
            .split("/")[-1]
        )

        self.repo_root = repo_root

        self.repo_path = os.path.join(
            repo_root,
            self.repo_name
        )

        self.repo = None

        #
        # Cache
        #
        self.parent_cache = {}

        self.file_cache = {}

    def clone(self):

        if not os.path.exists(
            self.repo_path
        ):

            print(
                f"Cloning {self.repo_name}"
            )

            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    self.repo_url,
                    self.repo_path
                ],
                check=True
            )

        self.repo = Repo(
            self.repo_path
        )

    def delete(self):

        if os.path.exists(
            self.repo_path
        ):

            shutil.rmtree(
                self.repo_path
            )

    def get_parent_sha(
        self,
        sha
    ):

        if sha in self.parent_cache:

            return self.parent_cache[
                sha
            ]

        try:

            commit = self.repo.commit(
                sha
            )

        except Exception:

            self.parent_cache[
                sha
            ] = None

            return None

        if len(
            commit.parents
        ) == 0:

            self.parent_cache[
                sha
            ] = None

            return None

        parent_sha = (
            commit.parents[0]
            .hexsha
        )

        self.parent_cache[
            sha
        ] = parent_sha

        return parent_sha

    def get_previous_source(
        self,
        parent_sha,
        filename
    ):

        filename = filename.lstrip("/")

        key = (
            parent_sha,
            filename
        )

        if key in self.file_cache:

            return self.file_cache[
                key
            ]

        try:

            source = subprocess.check_output(
                [
                    "git",
                    "-C",
                    self.repo_path,
                    "cat-file",
                    "-p",
                    f"{parent_sha}:{filename}"
                ],
                text=True,
                errors="ignore"
            )

        except Exception:

            source = None

        self.file_cache[
            key
        ] = source

        return source

    def clear_cache(self):

        self.parent_cache.clear()

        self.file_cache.clear()