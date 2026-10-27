
from types import SimpleNamespace

from src.localizer.FunctionScopeLocalizer import FunctionScopeLocalizer


def make_node(node_id, lineno):
    return SimpleNamespace(
        node_id=node_id,
        lineno=lineno,
    )


def make_sample(nodes, **attributes):
    sample = SimpleNamespace(
        cfg={"nodes": nodes},
        seed_nodes=[],
    )

    for name, value in attributes.items():
        setattr(sample, name, value)

    return sample


def test_localizes_single_function():
    sample = make_sample(
        [
            make_node(1, 1),
            make_node(2, 3),
            make_node(3, 8),
            make_node(4, 12),
        ],
        function_start=2,
        function_end=8,
        seed_nodes=[2, 4],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == [2, 3]
    assert result.seed_nodes == [2]
    assert result.function_scope_status == "success"


def test_localizes_all_eligible_function_ranges():
    sample = make_sample(
        [
            make_node(1, 2),
            make_node(2, 5),
            make_node(3, 15),
            make_node(4, 20),
            make_node(5, 30),
        ],
        function_scopes=[
            {"start": 1, "end": 6},
            {"start": 14, "end": 21},
        ],
        seed_nodes=[2, 4, 5],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == [1, 2, 3, 4]
    assert result.seed_nodes == [2, 4]
    assert result.function_scope_status == "success"


def test_excludes_nodes_outside_matched_functions():
    sample = make_sample(
        [
            make_node(1, 2),
            make_node(2, 12),
            make_node(3, 25),
        ],
        function_scopes=[
            {"start": 10, "end": 15},
        ],
        seed_nodes=[2],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == [2]
    assert result.function_scope_status == "success"


def test_missing_scope_does_not_include_entire_file():
    sample = make_sample(
        [
            make_node(1, 1),
            make_node(2, 10),
            make_node(3, 20),
        ],
        seed_nodes=[1, 2, 3],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == []
    assert result.function_scope_status == "failed"
    assert result.function_scope_reason == "no_function_scopes"


def test_invalid_range_does_not_include_entire_file():
    sample = make_sample(
        [
            make_node(1, 1),
            make_node(2, 10),
        ],
        function_scopes=[
            {"start": 10, "end": 5},
        ],
        seed_nodes=[1, 2],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == []
    assert result.function_scope_status == "failed"
    assert result.function_scope_reason == "invalid_function_range"


def test_virtual_nodes_are_excluded():
    sample = make_sample(
        [
            make_node(1, -1),
            make_node(2, 5),
            make_node(3, 8),
        ],
        function_start=1,
        function_end=10,
        seed_nodes=[2],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == [2, 3]


def test_empty_cfg_fails_explicitly():
    sample = make_sample(
        [],
        function_start=1,
        function_end=10,
        seed_nodes=[],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == []
    assert result.function_scope_status == "failed"
    assert result.function_scope_reason == "empty_cfg"


def test_duplicate_node_ids_are_removed():
    sample = make_sample(
        [
            make_node(1, 5),
            make_node(1, 6),
            make_node(2, 7),
        ],
        function_start=1,
        function_end=10,
        seed_nodes=[1],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == [1, 2]


def test_no_seed_in_scope_fails():
    sample = make_sample(
        [
            make_node(1, 5),
            make_node(2, 15),
        ],
        function_start=1,
        function_end=10,
        seed_nodes=[2],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == [1]
    assert result.seed_nodes == []
    assert result.function_scope_status == "failed"
    assert (
        result.function_scope_reason
        == "no_seed_nodes_in_function_scope"
    )


from types import SimpleNamespace

from src.localizer.FunctionScopeLocalizer import FunctionScopeLocalizer


def make_node(node_id, lineno):
    return {
        "node_id": node_id,
        "lineno": lineno,
    }


def test_nested_function_nodes_are_excluded_from_outer_scope():
    sample = SimpleNamespace(
        cfg={
            "nodes": [
                make_node("outer_start", 1),
                make_node("outer_assignment", 2),
                make_node("inner_start", 3),
                make_node("inner_assignment", 4),
                make_node("inner_return", 5),
                make_node("outer_return", 6),
            ]
        },
        function_start=1,
        function_end=6,
        function_body_lines=[1, 2, 6],
        seed_nodes=["outer_assignment", "inner_assignment"],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == [
        "outer_start",
        "outer_assignment",
        "outer_return",
    ]
    assert result.seed_nodes == ["outer_assignment"]
    assert result.function_scope_status == "success"


def test_inner_function_can_keep_its_own_nodes():
    sample = SimpleNamespace(
        cfg={
            "nodes": [
                make_node("outer_assignment", 2),
                make_node("inner_start", 3),
                make_node("inner_assignment", 4),
                make_node("inner_return", 5),
            ]
        },
        function_start=3,
        function_end=5,
        function_body_lines=[3, 4, 5],
        seed_nodes=["inner_assignment"],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == [
        "inner_start",
        "inner_assignment",
        "inner_return",
    ]
    assert result.seed_nodes == ["inner_assignment"]
    assert result.function_scope_status == "success"


def test_missing_body_lines_uses_legacy_range():
    sample = SimpleNamespace(
        cfg={
            "nodes": [
                make_node("outside", 2),
                make_node("inside", 4),
                make_node("outside_again", 8),
            ]
        },
        function_start=3,
        function_end=5,
        seed_nodes=["inside"],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == ["inside"]
    assert result.seed_nodes == ["inside"]
    assert result.function_scope_status == "success"


def test_missing_scope_does_not_include_entire_file():
    sample = SimpleNamespace(
        cfg={
            "nodes": [
                make_node("node_1", 1),
                make_node("node_2", 2),
            ]
        },
        seed_nodes=["node_1"],
    )

    result = FunctionScopeLocalizer().localize(sample)

    assert result.function_nodes == []
    assert result.seed_nodes == []
    assert result.function_scope_status == "failed"
