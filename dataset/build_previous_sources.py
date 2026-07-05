import os
import pickle

from git import Repo
from tqdm import tqdm
import json

def fetch_previous_sources(
    dataset,
    repo_root,
    output_file
):

    lookup = {}

    repo_cache = {}

    for repo_data in tqdm(dataset):

        repo_url = repo_data["repo"]

        repo_name = repo_url.rstrip("/").split("/")[-1]

        if repo_name not in repo_cache:

            repo_cache[repo_name] = Repo(
                os.path.join(
                    repo_root,
                    repo_name
                )
            )

        git_repo = repo_cache[repo_name]

        for commit in repo_data["commits"]:

            sha = commit["sha"]

            try:

                git_commit = git_repo.commit(
                    sha
                )

                if len(git_commit.parents) == 0:
                    continue

                parent = git_commit.parents[0]

            except Exception:
                continue

            for file in commit["files"]:

                filename = file["filename"]

                key = (
                    repo_url,
                    sha,
                    filename.lstrip("/")
                )

                if key in lookup:
                    continue

                try:

                    blob = parent.tree / filename

                    source = blob.data_stream.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

                    lookup[key] = {
                                        
                        "repo": repo_url,
                    
                        "commit": sha,
                    
                        "filename": filename,
                    
                        "source": source
                    
                    }

                except Exception:

                    lookup[key] = None

    with open(
        output_file,
        "wb"
    ) as f:

        pickle.dump(
            lookup,
            f
        )

    print(
        "Saved",
        len(lookup),
        "previous sources."
    )

if __name__ == "__main__":
    with open(
        "dataset/python_sql.json",
        "r"
    ) as f:

        dataset = json.load(f)

    fetch_previous_sources(
        dataset,
        repo_root="repositories",
        output_file="dataset/previous_sources.pkl"
    )