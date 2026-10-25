
from types import SimpleNamespace

from src.cfg.cfg_builder import CFGBuilder


def test_cfg_builder_accepts_sample_and_builds_whole_file():
    source = """
def vulnerable_function(user_input):
    query = user_input
    return query

def unrelated_function():
    return 42
"""

    sample = SimpleNamespace(source=source)
    builder = CFGBuilder()

    cfg = builder.build(sample)

    assert cfg is not None
    assert "nodes" in cfg
    assert "edges" in cfg

    function_names = {
        node.text
        for node in cfg["nodes"]
        if node.node_type in ("FunctionDef", "AsyncFunctionDef")
    }

    assert "def vulnerable_function" in function_names
    assert "def unrelated_function" in function_names


def test_cfg_builder_returns_none_for_invalid_python():
    sample = SimpleNamespace(
        source="def broken_function(:\n    pass\n"
    )

    cfg = CFGBuilder().build(sample)

    assert cfg is None
