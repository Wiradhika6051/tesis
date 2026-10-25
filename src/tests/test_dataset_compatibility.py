
from types import SimpleNamespace

from src.cfg.cfg_builder import CFGBuilder
from src.localizer.LineCFGLocalizer import LineCFGLocalizer


def test_line_localizer_handles_empty_seed_lines():
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
    assert result.seed_nodes == []
    assert result.line_to_node == {}


def test_cfg_builder_builds_whole_file():
    source = """
def target_function(value):
    return value

def unrelated_function():
    return 42
"""

    sample = SimpleNamespace(source=source)
    cfg = CFGBuilder().build(sample)

    assert cfg is not None
    assert "nodes" in cfg
    assert "edges" in cfg

    function_nodes = [
        node.text
        for node in cfg["nodes"]
        if node.node_type in ("FunctionDef", "AsyncFunctionDef")
    ]

    assert "def target_function" in function_nodes
    assert "def unrelated_function" in function_nodes


def test_cfg_builder_returns_none_for_syntax_error():
    sample = SimpleNamespace(
        source="def broken_function(:\n    pass\n"
    )

    assert CFGBuilder().build(sample) is None
