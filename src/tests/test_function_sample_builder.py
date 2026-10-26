
from types import SimpleNamespace

from src.dataset.function_sample_builder import FunctionSampleBuilder


def make_sample(source, seed_lines):
    return SimpleNamespace(
        source=source,
        seed_lines=seed_lines,
        seed_nodes=[],
        repo="test-repo",
        commit_hash="test-commit",
        file_path="app.py",
    )


def test_changed_lines_are_assigned_to_correct_function():
    source = """\
def get_user(user_id):
    query = build_query(user_id)
    return query

def format_name(name):
    return name.strip()
"""

    sample = make_sample(source, [2, 6])

    builder = FunctionSampleBuilder()
    functions = builder.build(sample)

    seeds_by_function = {
        function.function_name: function.seed_lines
        for function in functions
    }

    assert seeds_by_function == {
        "get_user": [2],
        "format_name": [6],
    }


def test_function_without_changed_lines_is_retained():
    source = """\
def get_user(user_id):
    query = build_query(user_id)
    return query

def format_name(name):
    return name.strip()
"""

    sample = make_sample(source, [2])

    builder = FunctionSampleBuilder()
    functions = builder.build(sample)

    seeds_by_function = {
        function.function_name: function.seed_lines
        for function in functions
    }

    assert seeds_by_function == {
        "get_user": [2],
        "format_name": [],
    }
