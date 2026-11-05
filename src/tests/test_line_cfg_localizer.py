from types import SimpleNamespace

from src.localizer.FunctionScopeLocalizer import FunctionScopeLocalizer
from src.localizer.LineCFGLocalizer import LineCFGLocalizer


def make_node(node_id, start, end, node_type="Assign"):
    """Build a lightweight CFG node for localization tests."""
    return SimpleNamespace(
        node_id=node_id,
        lineno=start,
        end_lineno=end,
        node_type=node_type,
    )


def make_sample(nodes, seed_lines, function_start=None, function_end=None):
    """Build a sample with one consistent helper signature."""
    sample = SimpleNamespace(
        cfg={"nodes": nodes, "edges": []},
        seed_lines=list(seed_lines),
        line_to_node={},
        seed_nodes=[],
    )

    if function_start is not None:
        sample.function_start = function_start
    if function_end is not None:
        sample.function_end = function_end

    return sample


def test_empty_seed_lines_produce_empty_seed_nodes():
    sample = make_sample(
        nodes=[make_node(0, 1, 3, "FunctionDef")],
        seed_lines=[],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result is sample
    assert result.seed_lines == []
    assert result.seed_nodes == []
    assert result.line_to_node == {}
    assert result.unmatched_seed_lines == []
    assert result.localization_status == "no_changed_lines"
    assert result.exclusion_reason == "no_changed_lines"


def test_function_definition_is_not_selected_when_narrower_node_matches():
    sample = make_sample(
        nodes=[
            make_node(0, 1, 5, "FunctionDef"),
            make_node(1, 2, 2, "Assign"),
            make_node(2, 3, 3, "Return"),
        ],
        seed_lines=[2],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {2: [1]}
    assert result.seed_nodes == [1]
    assert 0 not in result.seed_nodes
    assert result.localization_status == "success"
    assert result.exclusion_reason is None


def test_virtual_nodes_are_not_selected():
    sample = make_sample(
        nodes=[
            make_node(0, -1, -1, "ENTRY"),
            make_node(1, 1, 1, "FunctionDef"),
            make_node(2, 2, 2, "Assign"),
        ],
        seed_lines=[2],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {2: [2]}
    assert result.seed_nodes == [2]


def test_function_scope_removes_seed_outside_function():
    sample = make_sample(
        nodes=[
            make_node(0, 1, 1, "Assign"),
            make_node(1, 2, 2, "Assign"),
            make_node(2, 5, 5, "Assign"),
        ],
        seed_lines=[1, 2, 5],
        function_start=2,
        function_end=4,
    )
    sample.seed_nodes = [0, 1, 2]

    result = FunctionScopeLocalizer().localize(sample)

    valid_seeds = set(result.seed_nodes) & set(result.function_nodes)
    assert result.function_nodes == [1]
    assert valid_seeds == {1}


def test_unmatched_seed_line_maps_to_empty_list():
    sample = make_sample(
        nodes=[
            make_node(0, 1, 1, "Assign"),
            make_node(1, 3, 3, "Return"),
        ],
        seed_lines=[2],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {2: []}
    assert result.seed_nodes == []
    assert result.unmatched_seed_lines == [2]
    assert result.localization_status == "failed"
    assert result.exclusion_reason == "no_valid_seed_nodes"


def test_seed_nodes_are_deduplicated_across_changed_lines():
    sample = make_sample(
        nodes=[
            make_node(0, 1, 5, "FunctionDef"),
            make_node(1, 2, 3, "If"),
            make_node(2, 2, 2, "Assign"),
            make_node(3, 3, 3, "Assign"),
        ],
        seed_lines=[2, 3, 2],
    )

    result = LineCFGLocalizer().localize(sample)

    # The narrowest matching node wins for each changed line.
    assert result.line_to_node == {
        2: [2],
        3: [3],
    }
    assert result.seed_nodes == [2, 3]


def test_virtual_and_invalid_span_nodes_are_excluded():
    sample = make_sample(
        nodes=[
            make_node(0, -1, -1, "ENTRY"),
            make_node(1, 4, 3, "Invalid"),
            make_node(2, 2, 4, "If"),
            make_node(3, 2, 2, "Assign"),
        ],
        seed_lines=[2],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {2: [3]}
    assert result.seed_nodes == [3]


def test_partial_localization_records_unmatched_lines():
    sample = make_sample(
        nodes=[
            make_node(0, 1, 5, "FunctionDef"),
            make_node(1, 2, 2, "Assign"),
        ],
        seed_lines=[2, 99],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {
        2: [1],
        99: [],
    }
    assert result.seed_nodes == [1]
    assert result.unmatched_seed_lines == [99]
    assert result.localization_status == "partial"
    assert result.exclusion_reason is None


def test_function_definition_only_match_is_reported_as_container_only():
    sample = make_sample(
        nodes=[make_node(0, 10, 20, "FunctionDef")],
        seed_lines=[15],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.seed_nodes == []
    assert result.container_only_seed_lines == [15]
    assert result.unmatched_seed_lines == []
    assert result.localization_status == "failed"
    assert result.exclusion_reason == "no_valid_seed_nodes"


def test_no_valid_seeds_marks_sample_as_failed():
    sample = make_sample(
        nodes=[
            make_node(0, -1, -1, "ENTRY"),
            make_node(1, 4, 3, "Invalid"),
        ],
        seed_lines=[2, 3],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.seed_nodes == []
    assert result.unmatched_seed_lines == [2, 3]
    assert result.localization_status == "failed"
    assert result.exclusion_reason == "no_valid_seed_nodes"


def test_multiple_nodes_with_same_narrowest_span_are_kept():
    sample = make_sample(
        nodes=[
            make_node(0, 10, 20, "FunctionDef"),
            make_node(1, 12, 12, "Return"),
            make_node(2, 12, 12, "Call"),
            make_node(3, 11, 15, "If"),
        ],
        seed_lines=[12],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {12: [1, 2]}
    assert result.seed_nodes == [1, 2]
    assert 0 not in result.seed_nodes
    assert 3 not in result.seed_nodes


def test_unmatched_line_is_reported_when_no_node_covers_it():
    sample = make_sample(
        nodes=[make_node(0, 10, 20, "FunctionDef")],
        seed_lines=[25],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result.seed_nodes == []
    assert result.unmatched_seed_lines == [25]
    assert result.container_only_seed_lines == []
    assert result.localization_status == "failed"
    assert result.exclusion_reason == "no_valid_seed_nodes"
