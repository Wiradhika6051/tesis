
import ast
import json
import sys
from pathlib import Path

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
]

SYNTHETIC_NODE_TYPES = [
    "ENTRY",
    "EXIT",
    "FUNCTION_EXIT",
    "MERGE",
    "LOOP_NORMAL_EXIT",
    "LOOP_EXIT",
]


def get_ast_statement_types():
    """Discover statement AST classes available in this Python runtime."""
    return sorted(
        name
        for name, obj in vars(ast).items()
        if (
            isinstance(obj, type)
            and issubclass(obj, ast.stmt)
            and obj is not ast.stmt
        )
    )


def create_cfg_vocab():
    """Create a deterministic CFG vocabulary."""
    node_types = (
        SPECIAL_TOKENS
        + get_ast_statement_types()
        + SYNTHETIC_NODE_TYPES
    )

    vocab = {}
    for node_type in node_types:
        if node_type not in vocab:
            vocab[node_type] = len(vocab)

    return vocab


def save_cfg_vocab(vocab, vocab_path):
    """Save the vocabulary and runtime metadata as JSON."""
    data = {
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}"
        ),
        "ast_statement_types": get_ast_statement_types(),
        "synthetic_node_types": SYNTHETIC_NODE_TYPES,
        "vocab": vocab,
    }

    vocab_path = Path(vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cfg_vocab(storage_dir, filename="cfg_vocab.json"):
    """
    Regenerate the CFG vocabulary and save it to the specified directory.

    Args:
        storage_dir: Directory where the JSON vocabulary will be stored.
        filename: Name of the JSON file.

    Returns:
        dict: Mapping from CFG node types to integer IDs.
    """
    storage_dir = Path(storage_dir)
    vocab_path = storage_dir / filename

    vocab = create_cfg_vocab()
    save_cfg_vocab(vocab, vocab_path)

    return vocab
