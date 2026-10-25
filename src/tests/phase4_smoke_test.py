"""Phase 4 end-to-end smoke test for paired-function CFG/GCN experiments.

This module uses the project's existing preparation pipeline, GCNEncoder,
VUDENCVectorizer, build_cfg_vocab, and PhpNetGraph implementations.

Example (run after the notebook has created train_base, val_base, test_base,
make_pipeline, VUDENC_W2V_PATH, and the optional cleanup helpers):

    from phase4_smoke_test_refactored import run_phase4_smoke_test

    result = run_phase4_smoke_test(
        train_base=train_base,
        val_base=val_base,
        test_base=test_base,
        pipeline_factory=make_pipeline,
        w2v_path=VUDENC_W2V_PATH,
        output_dir=f"{DEBUG_DIR}/phase4_smoke_test",
        epochs=2,
        max_samples_per_split=3,
        seed=RANDOM_SEED,
        deduplicate_function_samples=deduplicate_function_samples,
        validate_unique_function_samples=validate_unique_function_samples,
    )

Note: max_samples_per_split counts fixing-commit groups, not individual
function samples. Leave it as None to process the complete supplied splits.
"""
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.encoding.gcn_encoder import GCNEncoder
from src.encoding.vudenc_vectorizer import VUDENCVectorizer
from src.vocabulary import build_cfg_vocab


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Pair validation
# ---------------------------------------------------------------------------

def _pair_id(sample):
    return getattr(sample, "function_pair_id", None)


def validate_sample_pairs(samples, split_name: str) -> dict:
    """Require exactly two samples per function_pair_id, with labels 0 and 1."""
    groups = defaultdict(list)
    missing = []

    for sample in samples:
        pair_id = _pair_id(sample)
        if pair_id is None or str(pair_id).strip() == "":
            missing.append(getattr(sample, "sample_id", None))
        else:
            groups[str(pair_id)].append(sample)

    bad = []
    for pair_id, members in groups.items():
        labels = sorted(getattr(sample, "label", None) for sample in members)
        if len(members) != 2 or labels != [0, 1]:
            bad.append({
                "pair_id": pair_id,
                "count": len(members),
                "labels": labels,
                "sample_ids": [getattr(s, "sample_id", None) for s in members],
            })

    if missing or bad:
        raise ValueError(
            f"{split_name}: invalid vulnerable/fixed pairs. "
            f"Samples missing function_pair_id: {len(missing)}; "
            f"malformed pair groups: {len(bad)}. "
            f"Missing sample IDs (first 10): {missing[:10]}; "
            f"bad pairs (first 10): {bad[:10]}"
        )

    return {"sample_count": len(samples), "pair_count": len(groups)}


def validate_encoded_pairs(graphs, split_name: str) -> dict:
    groups = defaultdict(list)

    for graph in graphs:
        pair_id = getattr(graph, "smoke_pair_id", None)
        if pair_id is None:
            raise ValueError(f"{split_name}: encoded graph has no pair ID")
        groups[str(pair_id)].append(graph)

    bad = []
    for pair_id, members in groups.items():
        labels = sorted(int(graph.y.view(-1)[0].item()) for graph in members)
        if len(members) != 2 or labels != [0, 1]:
            bad.append({"pair_id": pair_id, "count": len(members), "labels": labels})

    if bad:
        raise ValueError(
            f"{split_name}: {len(bad)} malformed encoded pairs; "
            f"examples: {bad[:10]}"
        )

    return {"sample_count": len(graphs), "pair_count": len(groups)}


# ---------------------------------------------------------------------------
# CFG and graph validation / encoding
# ---------------------------------------------------------------------------

def validate_pruned_cfg(sample, split_name: str) -> dict:
    """Validate the exact structure required by GCNEncoder and build_cfg_vocab."""
    cfg = getattr(sample, "pruned_cfg", None)
    sample_id = getattr(sample, "sample_id", None)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: pruned_cfg must be a dict, "
            f"got {type(cfg).__name__}"
        )

    nodes = cfg.get("nodes")
    edges = cfg.get("edges")
    if not isinstance(nodes, (list, tuple)):
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: pruned_cfg['nodes'] "
            "must be a list or tuple"
        )
    if not nodes:
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: pruned_cfg contains no nodes"
        )
    if not isinstance(edges, (list, tuple)):
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: pruned_cfg['edges'] "
            "must be a list or tuple"
        )

    node_ids = set()
    for index, node in enumerate(nodes):
        for attr in ("node_id", "node_type", "text"):
            if not hasattr(node, attr):
                raise ValueError(
                    f"{split_name}: sample_id={sample_id!r}: node {index} "
                    f"is missing required attribute {attr!r}"
                )
        if node.node_id in node_ids:
            raise ValueError(
                f"{split_name}: sample_id={sample_id!r}: duplicate CFG node_id "
                f"{node.node_id!r}"
            )
        node_ids.add(node.node_id)

    invalid_edges = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            invalid_edges.append((edge, "edge must be a 2-item tuple/list"))
            continue
        src, dst = edge
        if src not in node_ids or dst not in node_ids:
            invalid_edges.append((edge, "edge endpoint is absent from nodes"))

    if invalid_edges:
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: invalid CFG edges "
            f"(first 10): {invalid_edges[:10]}"
        )

    return {"nodes": len(nodes), "edges": len(edges)}


def validate_encoded_graph(sample, graph, split_name: str) -> dict:
    """Check the fields PhpNetGraph actually reads before starting training."""
    sample_id = getattr(sample, "sample_id", None)
    required = ("edge_index", "node_types", "node_token_vectors")
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: graph missing {missing}"
        )

    num_nodes = int(graph.node_types.numel())
    if num_nodes <= 0:
        raise ValueError(f"{split_name}: sample_id={sample_id!r}: graph has zero nodes")

    if graph.node_types.ndim != 1:
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: node_types must be 1D, "
            f"got shape {tuple(graph.node_types.shape)}"
        )

    if len(graph.node_token_vectors) != num_nodes:
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: {num_nodes} node types but "
            f"{len(graph.node_token_vectors)} token-vector sequences"
        )

    if graph.edge_index.ndim != 2 or graph.edge_index.size(0) != 2:
        raise ValueError(
            f"{split_name}: sample_id={sample_id!r}: edge_index must have shape "
            f"[2, E], got {tuple(graph.edge_index.shape)}"
        )

    if graph.edge_index.numel() > 0:
        if int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= num_nodes:
            raise ValueError(
                f"{split_name}: sample_id={sample_id!r}: edge_index contains "
                "node indices outside the node_types range"
            )

    for node_index, vectors in enumerate(graph.node_token_vectors):
        if not torch.is_tensor(vectors):
            raise ValueError(
                f"{split_name}: sample_id={sample_id!r}: token vectors for node "
                f"{node_index} are not a torch.Tensor"
            )
        if vectors.ndim != 2 or vectors.size(1) != 200:
            raise ValueError(
                f"{split_name}: sample_id={sample_id!r}: node {node_index} token "
                f"vectors must have shape [tokens, 200], got {tuple(vectors.shape)}"
            )

    # PyG may infer num_nodes from edge_index, which can undercount isolated nodes.
    # Set it from node_types, the authoritative node list in this encoder.
    graph.num_nodes = num_nodes

    # GCNEncoder writes graph.label, but the training loop uses graph.y.
    graph.y = torch.tensor([int(sample.label)], dtype=torch.long)
    graph.smoke_pair_id = str(_pair_id(sample))
    graph.smoke_revision = getattr(sample, "revision", None)
    graph.smoke_sample_id = sample_id

    return {
        "nodes": num_nodes,
        "edges": int(graph.edge_index.size(1)),
        "token_sequences": len(graph.node_token_vectors),
    }


def encode_samples(samples, encoder, split_name: str):
    graphs = []
    failures = []
    graph_stats = []

    for index, sample in enumerate(samples):
        try:
            validate_pruned_cfg(sample, split_name)
            encoded_sample = encoder.encode(sample)
            graph = getattr(encoded_sample, "graph", None)
            if graph is None:
                raise ValueError("GCNEncoder.encode() did not attach sample.graph")

            stats = validate_encoded_graph(sample, graph, split_name)
            graphs.append(graph)
            graph_stats.append({
                "sample_id": getattr(sample, "sample_id", None),
                "pair_id": str(_pair_id(sample)),
                **stats,
            })
        except Exception as exc:
            failures.append({
                "index": index,
                "sample_id": getattr(sample, "sample_id", None),
                "pair_id": _pair_id(sample),
                "error": f"{type(exc).__name__}: {exc}",
            })

    if failures:
        raise RuntimeError(
            f"{split_name}: encoding/validation failed for "
            f"{len(failures)}/{len(samples)} samples. First failures:\n"
            + "\n".join(
                f"  idx={item['index']} sample={item['sample_id']!r} "
                f"pair={item['pair_id']!r}: {item['error']}"
                for item in failures[:10]
            )
        )

    if not graphs:
        raise RuntimeError(f"{split_name}: encoding produced zero graphs")

    validate_encoded_pairs(graphs, split_name)
    return graphs, graph_stats


# ---------------------------------------------------------------------------
# Training / evaluation: one graph at a time
# ---------------------------------------------------------------------------

def _prepare_single_graph(graph, device):
    """Move one graph to device and set the batch vector expected by PhpNetGraph."""
    graph = graph.to(device)
    graph.num_nodes = int(graph.node_types.numel())
    graph.batch = torch.zeros(
        graph.num_nodes,
        dtype=torch.long,
        device=device,
    )
    # Ensure target remains a one-element class-index tensor: shape [1].
    graph.y = graph.y.view(-1).long()
    if graph.y.numel() != 1:
        raise ValueError(f"Expected one label per graph, got y shape {tuple(graph.y.shape)}")
    return graph


def _run_epoch(model, graphs, device, optimizer=None, epoch_name="evaluation"):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for original_graph in graphs:
            graph = _prepare_single_graph(original_graph, device)
            labels = graph.y

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(graph)
            if logits.ndim != 2 or tuple(logits.shape) != (1, 2):
                raise RuntimeError(
                    f"{epoch_name}: PhpNetGraph must return logits with shape [1, 2], "
                    f"got {tuple(logits.shape)}"
                )

            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            loss_sum += float(loss.detach().item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += 1

    if count == 0:
        raise RuntimeError(f"{epoch_name}: no graphs were processed")

    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


def _commit_key(sample):
    """Return a best-effort repository/fixing-change grouping key."""
    repo = getattr(sample, "repo", None)
    commit = (
        getattr(sample, "fix_commit_hash", None)
        or getattr(sample, "parent_commit", None)
        or getattr(sample, "commit_hash", None)
        or getattr(sample, "commit", None)
    )
    if not repo or not commit:
        return None
    return (str(repo), str(commit))


def _limit_base_samples(samples, limit: Optional[int], split_name: str):
    """Select up to `limit` complete commit groups before function extraction."""
    if limit is None:
        return list(samples)
    if limit < 1:
        raise ValueError("max_samples_per_split must be >= 1 or None")

    grouped = defaultdict(list)
    ungrouped = []
    for sample in samples:
        key = _commit_key(sample)
        if key is None:
            ungrouped.append(sample)
        else:
            grouped[key].append(sample)

    if not grouped:
        print(
            f"[{split_name}] No repo/commit keys found; using the entire split "
            "instead of pretending it is commit-limited."
        )
        return list(samples)

    keys = sorted(grouped, key=str)
    chosen = random.sample(keys, min(limit, len(keys)))
    selected = [sample for key in chosen for sample in grouped[key]]

    if ungrouped:
        print(
            f"[{split_name}] Excluding {len(ungrouped)} ungrouped base samples "
            "from the commit-limited smoke-test subset."
        )

    return selected


def _check_no_commit_leakage(base_splits):
    """Fail if the same identifiable fixing change occurs in multiple splits."""
    def keys(samples):
        return {key for sample in samples if (key := _commit_key(sample)) is not None}

    split_names = list(base_splits)
    overlaps = []
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = keys(base_splits[left]) & keys(base_splits[right])
            if overlap:
                overlaps.append({
                    "splits": [left, right],
                    "count": len(overlap),
                    "examples": sorted(overlap)[:10],
                })

    if overlaps:
        raise ValueError(
            "Base split leakage detected: fixing-change groups overlap between "
            "splits. This smoke test expects commit-disjoint splits. Details: "
            f"{overlaps}"
        )


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase4_smoke_test(
    *,
    train_base,
    val_base,
    test_base,
    pipeline_factory: Callable,
    w2v_path,
    output_dir="preparation_audit/phase4_smoke_test",
    epochs=2,
    batch_size=4,
    max_samples_per_split=None,
    seed=42,
    device=None,
    deduplicate_function_samples=None,
    validate_unique_function_samples=None,
):
    """Run preparation, audit, pair/graph validation, and a short train/val run.

    `pipeline_factory` must return a fresh, fully configured pipeline every time.
    The test split is prepared and encoded for compatibility checks but is not
    used for training or validation metrics.
    """
    from src.model.PhpNetGraph import PhpNetGraph

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_splits = {
        "train": copy.deepcopy(list(train_base)),
        "val": copy.deepcopy(list(val_base)),
        "test": copy.deepcopy(list(test_base)),
    }

    if any(not samples for samples in base_splits.values()):
        empty = [name for name, samples in base_splits.items() if not samples]
        raise ValueError(f"Empty base split(s): {empty}")

    _check_no_commit_leakage(base_splits)

    prepared = {}
    audit_snapshots = {}
    split_stats = {}

    for split_name, original_base_samples in base_splits.items():
        base_samples = _limit_base_samples(
            original_base_samples,
            max_samples_per_split,
            split_name,
        )
        if not base_samples:
            raise ValueError(f"Base split {split_name!r} is empty after limiting")

        pipeline = pipeline_factory()
        if pipeline is None or not callable(getattr(pipeline, "prepare", None)):
            raise TypeError(
                "pipeline_factory() must return an object exposing prepare(samples)"
            )

        samples = pipeline.prepare(base_samples)
        if samples is None:
            raise RuntimeError(f"Pipeline returned None for split {split_name}")
        samples = list(samples)

        if deduplicate_function_samples is not None:
            samples = list(deduplicate_function_samples(samples))
        if validate_unique_function_samples is not None:
            validate_unique_function_samples(samples, f"PHASE4 {split_name.upper()}")

        # Snapshot the audit immediately; the next pipeline has independent state.
        audit_snapshots[split_name] = {
            "pipeline_audit": copy.deepcopy(getattr(pipeline, "audit", {})),
            "pairing_audit": copy.deepcopy(
                getattr(getattr(pipeline, "paired_function_extractor", None), "audit", {})
            ),
        }
        _write_json(output_dir / f"{split_name}_audit.json", audit_snapshots[split_name])

        if not samples:
            raise RuntimeError(
                f"Preparation produced zero samples for {split_name}; inspect "
                f"{output_dir / f'{split_name}_audit.json'}"
            )

        split_stats[split_name] = validate_sample_pairs(samples, split_name)
        prepared[split_name] = samples
        print(
            f"[{split_name}] prepared {split_stats[split_name]['sample_count']} samples / "
            f"{split_stats[split_name]['pair_count']} complete pairs"
        )

    # Build node-type vocabulary using training samples only.
    cfg_vocab = build_cfg_vocab(prepared["train"])
    if not cfg_vocab or "<UNK>" not in cfg_vocab:
        raise RuntimeError("CFG vocabulary is empty or does not contain '<UNK>'")

    vectorizer = VUDENCVectorizer(w2v_path)
    encoder = GCNEncoder(vudenc_vectorizer=vectorizer, cfg_vocab=cfg_vocab)

    encoded = {}
    graph_stats = {}
    for split_name, samples in prepared.items():
        encoded[split_name], graph_stats[split_name] = encode_samples(
            samples, encoder, split_name
        )
        positives = sum(int(graph.y.item()) == 1 for graph in encoded[split_name])
        negatives = sum(int(graph.y.item()) == 0 for graph in encoded[split_name])
        print(
            f"[{split_name}] encoded {len(encoded[split_name])} graphs: "
            f"label 1={positives}, label 0={negatives}"
        )

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PhpNetGraph(cfg_vocab_size=len(cfg_vocab), word2vec_dim=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    for epoch in range(1, epochs + 1):
        train_graphs = list(encoded["train"])
        random.shuffle(train_graphs)
        train_metrics = _run_epoch(
            model,
            train_graphs,
            device,
            optimizer=optimizer,
            epoch_name=f"train epoch {epoch}",
        )
        val_metrics = _run_epoch(
            model,
            encoded["val"],
            device,
            optimizer=None,
            epoch_name=f"validation epoch {epoch}",
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.3f} | "
            f"val loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.3f}"
        )

    result = {
        "status": "passed",
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size_requested": batch_size,
        "training_mode": "one graph at a time; batch vector is all zeros per graph",
        "cfg_vocab_size": len(cfg_vocab),
        "splits": split_stats,
        "encoded": {
            name: {
                "sample_count": len(graphs),
                "pair_count": len({graph.smoke_pair_id for graph in graphs}),
                "label_1_count": sum(int(graph.y.item()) == 1 for graph in graphs),
                "label_0_count": sum(int(graph.y.item()) == 0 for graph in graphs),
                "graphs": graph_stats[name],
            }
            for name, graphs in encoded.items()
        },
        "history": history,
        "audit_dir": str(output_dir.resolve()),
        "notes": [
            "The test split was prepared and encoded but not used for training or validation metrics.",
            "Training is graph-by-graph because node_token_vectors is a variable-length Python list.",
            "batch_size is recorded for reference; this smoke-test implementation does not batch graphs.",
        ],
    }
    _write_json(output_dir / "smoke_test_result.json", result)
    print(f"Phase 4 smoke test passed. Reports saved under: {output_dir.resolve()}")
    return result
