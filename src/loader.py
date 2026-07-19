from typing import List
from type.Sample import Sample
from type.GitChange import GitChange

def build_samples(change: GitChange) -> List[Sample]:
    """
    Build two training samples from a vulnerability-fixing commit.

    Returns:
        Sample(label=1): Parent (vulnerable) revision
        Sample(label=0): Current (fixed) revision
    """

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

    return [vulnerable, fixed]