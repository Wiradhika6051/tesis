
from types import SimpleNamespace

from src.localizer.LineCFGLocalizer import LineCFGLocalizer


def test_empty_seed_lines_produce_empty_seed_nodes():
    sample = SimpleNamespace(
        cfg={
            "nodes": [
                SimpleNamespace(
                    node_id=0,
                    lineno=1,
                    end_lineno=3,
                )
            ]
        },
        seed_lines=[],
        line_to_node={},
        seed_nodes=[],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result is sample
    assert result.seed_lines == []
    assert result.seed_nodes == []
    assert result.line_to_node == {}


from types import SimpleNamespace

from src.localizer.LineCFGLocalizer import LineCFGLocalizer
from src.localizer.FunctionScopeLocalizer import FunctionScopeLocalizer


def make_node(node_id, lineno, end_lineno, node_type="Assign"):
    return SimpleNamespace(
        node_id=node_id,
        lineno=lineno,
        end_lineno=end_lineno,
        node_type=node_type,
    )


def make_sample(seed_lines, nodes, function_start, function_end):
    return SimpleNamespace(
        seed_lines=seed_lines,
        cfg={"nodes": nodes, "edges": []},
        function_start=function_start,
        function_end=function_end,
    )


def test_seed_line_maps_to_every_matching_node():
    nodes = [
        make_node(0, 1, 5, "FunctionDef"),
        make_node(1, 2, 2, "Assign"),
        make_node(2, 3, 3, "Return"),
    ]

    sample = make_sample([2], nodes, 1, 5)

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {2: [0, 1]}
    assert result.seed_nodes == [0, 1]


def test_virtual_nodes_are_not_selected():
    nodes = [
        make_node(0, -1, -1, "ENTRY"),
        make_node(1, 1, 1, "FunctionDef"),
        make_node(2, 2, 2, "Assign"),
    ]
    sample = make_sample([2], nodes, 1, 2)

    result = LineCFGLocalizer().localize(sample)

    assert result.seed_nodes == [2]


def test_function_scope_removes_seed_outside_function():
    nodes = [
        make_node(0, 1, 1, "Assign"),
        make_node(1, 2, 2, "Assign"),
        make_node(2, 5, 5, "Assign"),
    ]
    sample = make_sample([1, 2, 5], nodes, 2, 4)
    sample.seed_nodes = [0, 1, 2]

    result = FunctionScopeLocalizer().localize(sample)

    valid_seeds = (
        set(result.seed_nodes) & set(result.function_nodes)
    )

    assert result.function_nodes == [1]
    assert valid_seeds == {1}


def test_seed_with_no_matching_cfg_node_is_empty():
    nodes = [
        make_node(0, 1, 1, "FunctionDef"),
        make_node(1, 3, 3, "Return"),
    ]
    sample = make_sample([2], nodes, 1, 3)

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {2: []}
    assert result.seed_nodes == []



def test_seed_line_maps_to_every_matching_node():
    nodes = [
        make_node(0, 1, 5, "FunctionDef"),
        make_node(1, 2, 2, "Assign"),
        make_node(2, 3, 3, "Return"),
    ]

    sample = make_sample([2], nodes, 1, 5)

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {2: [0, 1]}
    assert result.seed_nodes == [0, 1]



def test_seed_nodes_are_deduplicated_across_changed_lines():
    nodes = [
        make_node(0, 1, 5, "FunctionDef"),
        make_node(1, 2, 3, "If"),
        make_node(2, 2, 2, "Assign"),
        make_node(3, 3, 3, "Assign"),
    ]

    sample = make_sample([2, 3, 2], nodes, 1, 5)

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {
        2: [0, 1, 2],
        3: [0, 1, 3],
    }
    assert result.seed_nodes == [0, 1, 2, 3]


def test_virtual_and_invalid_span_nodes_are_excluded():
    nodes = [
        make_node(0, -1, -1, "ENTRY"),
        make_node(1, 4, 3, "Invalid"),
        make_node(2, 2, 4, "If"),
        make_node(3, 2, 2, "Assign"),
    ]

    sample = make_sample([2], nodes, 1, 5)

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {2: [2, 3]}
    assert result.seed_nodes == [2, 3]

def test_partial_localization_records_unmatched_lines():
    nodes = [
        make_node(0, 1, 5, "FunctionDef"),
        make_node(1, 2, 2, "Assign"),
    ]
    sample = make_sample([2, 99], nodes, 1, 5)

    result = LineCFGLocalizer().localize(sample)

    assert result.line_to_node == {
        2: [0, 1],
        99: [],
    }
    assert result.seed_nodes == [0, 1]
    assert result.unmatched_seed_lines == [99]
    assert result.localization_status == "partial"
    assert result.exclusion_reason is None


def test_no_valid_seeds_marks_sample_as_failed():
    nodes = [
        make_node(0, -1, -1, "ENTRY"),
        make_node(1, 4, 3, "Invalid"),
    ]
    sample = make_sample([2, 3], nodes, 1, 5)

    result = LineCFGLocalizer().localize(sample)

    assert result.seed_nodes == []
    assert result.unmatched_seed_lines == [2, 3]
    assert result.localization_status == "failed"
    assert result.exclusion_reason == "no_valid_seed_nodes"


def test_no_changed_lines_are_recorded_separately():
    nodes = [make_node(0, 1, 5, "FunctionDef")]
    sample = make_sample([], nodes, 1, 5)

    result = LineCFGLocalizer().localize(sample)

    assert result.seed_nodes == []
    assert result.unmatched_seed_lines == []
    assert result.localization_status == "no_changed_lines"
    assert result.exclusion_reason == "no_changed_lines"