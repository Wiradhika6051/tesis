
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


from types import SimpleNamespace

from src.dataset.function_sample_builder import FunctionSampleBuilder


def test_builder_emits_all_functions_with_changed_lines():
    source = """\
def get_user(user_id):
    user = load_user(user_id)
    return user

def format_name(first, last):
    full_name = first + " " + last
    return full_name
"""

    sample = SimpleNamespace(
        source=source,
        seed_lines=[3, 7],
        repo="test-repo",
        commit_hash="abc123",
        file_path="example.py",
    )

    builder = FunctionSampleBuilder()
    candidates = builder.build(sample)

    by_name = {candidate.function_name: candidate for candidate in candidates}

    assert "get_user" in by_name
    assert "format_name" in by_name

    assert by_name["get_user"].seed_lines == [3]
    assert by_name["format_name"].seed_lines == [7]

    assert (
        by_name["get_user"].function_start
        < by_name["format_name"].function_start
    )

    assert by_name["get_user"].function_source
    assert by_name["format_name"].function_source


from types import SimpleNamespace

from src.dataset.function_sample_builder import FunctionSampleBuilder


def test_builder_excludes_nested_function_lines_from_outer_scope():
    source = """\
def outer():
    x = 1

    def inner():
        y = 2
        return y

    return x
"""

    sample = SimpleNamespace(
        source=source,
        seed_lines=[2, 5, 6, 8],
        repo="test-repo",
        commit_hash="abc123",
        file_path="example.py",
    )

    candidates = FunctionSampleBuilder().build(sample)
    by_name = {
        candidate.function_name: candidate
        for candidate in candidates
    }

    outer = by_name["outer"]
    inner = by_name["inner"]

    assert set(outer.function_body_lines) == {1, 2, 3, 7, 8}
    assert set(inner.function_body_lines) == {4, 5, 6}

    assert outer.seed_lines == [2, 8]
    assert inner.seed_lines == [5, 6]
