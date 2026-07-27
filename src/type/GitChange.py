from dataclasses import dataclass

@dataclass
class GitChange:
    repo: str

    parent_commit: str
    commit_hash: str

    file_path: str

    # Unified diff (git diff -U...)
    diff: str

    previous_source: str = ""
    current_source: str = ""