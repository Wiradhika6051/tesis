
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
