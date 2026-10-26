
from types import SimpleNamespace

# Use the same import paths as your existing unit tests.
from src.localizer.GitDiffLocalizer import GitDiffLocalizer
from src.localizer.LineCFGLocalizer import LineCFGLocalizer
from src.localizer.FunctionScopeLocalizer import FunctionScopeLocalizer
from src.dataset.function_sample_builder import FunctionSampleBuilder


def make_diff_sample(diff, label=1):
    return SimpleNamespace(
        diff=diff,
        file_path="app.py",
        label=label,
        repo="test-repo",
        commit_hash="test-commit",
        seed_lines=[],
        seed_nodes=[],
    )


def make_node(node_id, lineno, end_lineno=None):
    return SimpleNamespace(
        node_id=node_id,
        lineno=lineno,
        end_lineno=lineno if end_lineno is None else end_lineno,
    )


def test_unrelated_function_changes_do_not_seed_eligible_function():
    """
    Integration test:
    - Vulnerable samples use deleted lines from the parent revision.
    - Changed lines are assigned to their respective functions.
    - Unrelated function changes do not become seeds for the eligible function.
    - Final seed nodes remain within the eligible function's scope.
    """

    # Parent revision: source code before the fix.
    parent_source = """\
def get_user(user_id):
    query = build_query(user_id)
    return query

def format_name(name):
    return name.strip()
"""

    # One commit changes two functions.
    diff = """\
diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -1,6 +1,6 @@
 def get_user(user_id):
-    query = build_query(user_id)
+    query = build_safe_query(user_id)
     return query

 def format_name(name):
-    return name.strip()
+    return name.strip().title()
"""

    # Stage 1: localize changed lines from the diff.
    parent_sample = make_diff_sample(diff, label=1)
    parent_sample.seed_lines = GitDiffLocalizer().localize(parent_sample)

    # Both deleted lines are valid candidates at the file level.
    assert parent_sample.seed_lines == [2, 6]

    # Stage 2: assign changed lines to function samples.
    # Use the parent revision for this vulnerable sample.
    parent_sample.source = parent_source
    function_samples = FunctionSampleBuilder().build(parent_sample)

    samples_by_function = {
        sample.function_name: sample
        for sample in function_samples
    }

    assert set(samples_by_function) == {"get_user", "format_name"}

    # Each changed line belongs to its own function.
    assert samples_by_function["get_user"].seed_lines == [2]
    assert samples_by_function["format_name"].seed_lines == [6]

    # Stage 3: simulate the full-file CFG's source-line coordinates.
    # The CFG includes nodes from both functions, as the real builder does.
    cfg_nodes = [
        make_node(0, 1),  # get_user declaration
        make_node(1, 2),  # changed query statement
        make_node(2, 3),  # get_user return
        make_node(3, 5),  # format_name declaration
        make_node(4, 6),  # unrelated changed return
    ]

    eligible_sample = samples_by_function["get_user"]
    eligible_sample.cfg = {
        "nodes": cfg_nodes,
        "edges": [],
    }

    # Stage 4: map changed lines to CFG nodes.
    eligible_sample = LineCFGLocalizer().localize(eligible_sample)

    assert eligible_sample.seed_nodes == [1]
    assert 4 not in eligible_sample.seed_nodes

    # Stage 5: enforce the eligible function's scope.
    eligible_sample = FunctionScopeLocalizer().localize(eligible_sample)

    valid_seeds = (
        set(eligible_sample.seed_nodes)
        & set(eligible_sample.function_nodes)
    )

    eligible_sample.seed_nodes = sorted(valid_seeds)

    # Final guarantee: only the changed line in get_user seeds this sample.
    assert eligible_sample.function_start == 1
    assert eligible_sample.function_end == 3
    assert eligible_sample.function_nodes == [0, 1, 2]
    assert eligible_sample.seed_nodes == [1]

    # The unrelated change remains associated with format_name.
    assert samples_by_function["format_name"].seed_lines == [6]
