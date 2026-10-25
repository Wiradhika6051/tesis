
"""Phase 4 smoke test for the paired-function CFG/GCN pipeline.

Reuses existing base splits from the notebook:
    train_base, val_base, test_base

The pipeline_factory must return a fresh FunctionDatasetPreparationPipeline
with the same components and pruner as the full experiment.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as F

from gcn_encoder import GCNEncoder
from vudenc_vectorizer import VUDENCVectorizer
from vocabulary import build_cfg_vocab


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. PAIR VALIDATION
# ============================================================

def _pair_id(sample):
    return getattr(sample, "function_pair_id", None)

from collections import defaultdict

def validate_sample_pairs(samples, split_name):
    pairs = defaultdict(list)

    for sample in samples:
        if not sample.function_pair_id:
            raise ValueError(
                f"{split_name}: Missing function_pair_id "
                f"for sample_id={sample.sample_id!r}"
            )

        pairs[sample.function_pair_id].append(sample)

    errors = []

    for pair_id, members in pairs.items():
        labels = sorted(sample.label for sample in members)

        if len(members) != 2 or labels != [0, 1]:
            errors.append(
                f"pair_id={pair_id!r}, "
                f"sample_ids={[s.sample_id for s in members]}, "
                f"labels={labels}"
            )

    if errors:
        raise ValueError(
            f"{split_name}: Invalid vulnerable/fixed pairs:\n"
            + "\n".join(errors[:20])
        )

    print(
        f"{split_name}: {len(samples)} samples, "
        f"{len(pairs)} complete pairs — OK"
    )
def validate_encoded_pairs(graphs, split_name):
    """Check that encoded graphs still form complete binary-label pairs."""

    groups = defaultdict(list)

    for graph in graphs:
        pair_id = getattr(graph, "smoke_pair_id", None)

        if pair_id is None:
            raise ValueError(
                f"{split_name}: encoded graph has no pair ID"
            )

        groups[pair_id].append(graph)

    bad = []

    for pair_id, members in groups.items():
        labels = sorted(int(graph.y.item()) for graph in members)

        if len(members) != 2 or labels != [0, 1]:
            bad.append((str(pair_id), len(members), labels))

    if bad:
        raise ValueError(
            f"{split_name}: {len(bad)} malformed encoded pairs. "
            f"Examples: {bad[:5]}"
        )

    return {
        "sample_count": len(graphs),
        "pair_count": len(groups),
    }


# ============================================================
# 3. ENCODING
# ============================================================

def encode_samples(samples, encoder, split_name):
    """Encode samples and report which samples fail."""

    graphs = []
    failures = []

    for index, sample in enumerate(samples):
        try:
            pruned_cfg = getattr(sample, "pruned_cfg", None)

            if not isinstance(pruned_cfg, dict):
                raise ValueError(
                    "pruned_cfg is missing or is not a dictionary"
                )

            if not pruned_cfg.get("nodes", []):
                raise ValueError("pruned_cfg contains no nodes")

            encoded_sample = encoder.encode(sample)
            graph = getattr(encoded_sample, "graph", None)

            if graph is None:
                raise ValueError(
                    "GCNEncoder did not attach sample.graph"
                )

            if graph.num_nodes is None or graph.num_nodes <= 0:
                raise ValueError("Encoded graph has zero nodes")

            if graph.node_types.numel() != graph.num_nodes:
                raise ValueError(
                    "node_types count differs from graph node count"
                )

            if len(graph.node_token_vectors) != graph.num_nodes:
                raise ValueError(
                    "Token-vector count differs from graph node count"
                )

            graph.y = torch.tensor(
                int(sample.label),
                dtype=torch.long,
            )

            graph.smoke_pair_id = _pair_id(sample)
            graph.smoke_revision = getattr(
                sample, "revision", None
            )

            graphs.append(graph)

        except Exception as exc:
            failures.append({
                "index": index,
                "sample_id": str(
                    getattr(sample, "sample_id", None)
                ),
                "pair_id": str(_pair_id(sample)),
                "error": f"{type(exc).__name__}: {exc}",
            })

    if failures:
        details = "\n".join(
            f"  index={failure['index']} "
            f"pair={failure['pair_id']}: {failure['error']}"
            for failure in failures[:10]
        )

        raise RuntimeError(
            f"{split_name}: encoding failed for "
            f"{len(failures)}/{len(samples)} samples.\n"
            f"First failures:\n{details}"
        )

    if not graphs:
        raise RuntimeError(
            f"{split_name}: encoding produced zero graphs"
        )

    validate_encoded_pairs(graphs, split_name)

    return graphs


# ============================================================
# 4. BASE-SPLIT COMMIT GROUPING
# ============================================================

def _commit_key(sample):
    """Best-effort grouping key for file-level samples."""

    repo = getattr(sample, "repo", None)

    commit = (
        getattr(sample, "fix_commit_hash", None)
        or getattr(sample, "parent_commit", None)
        or getattr(sample, "commit_hash", None)
        or getattr(sample, "commit", None)
    )

    if repo is None or commit is None:
        return None

    return str(repo), str(commit)


def _limit_base_samples(samples, limit, split_name):
    """Select whole commit groups rather than arbitrary individual samples."""

    if limit is None:
        return list(samples)

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
            f"[{split_name}] No repo/commit metadata found; "
            "using the entire split."
        )
        return list(samples)

    selected_keys = random.sample(
        sorted(grouped, key=str),
        min(limit, len(grouped)),
    )

    selected = [
        sample
        for key in selected_keys
        for sample in grouped[key]
    ]

    if ungrouped:
        print(
            f"[{split_name}] Excluding {len(ungrouped)} samples "
            "without commit metadata from the limited subset."
        )

    return selected


def _check_no_commit_leakage(base_splits):
    """Fail if identifiable commit groups occur in multiple base splits."""

    def commits(samples):
        return {
            key
            for sample in samples
            if (key := _commit_key(sample)) is not None
        }

    names = list(base_splits)

    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = commits(base_splits[left]) & commits(
                base_splits[right]
            )

            if overlap:
                raise ValueError(
                    f"Base split leakage: {left}/{right} share "
                    f"{len(overlap)} commit groups. "
                    f"Examples: {list(overlap)[:3]}"
                )


# ============================================================
# 5. GRAPH-BY-GRAPH TRAINING
# ============================================================

def _train_one_epoch(model, graphs, optimizer, device):
    """Train one graph at a time to avoid batching variable-length lists."""

    model.train()

    loss_sum = 0.0
    correct = 0
    count = 0

    for graph in graphs:
        graph = graph.to(device)

        # Every node belongs to graph zero.
        graph.batch = torch.zeros(
            graph.num_nodes,
            dtype=torch.long,
            device=device,
        )

        labels = graph.y.view(1).long()

        optimizer.zero_grad()

        logits = model(graph)

        if logits.ndim != 2 or tuple(logits.shape) != (1, 2):
            raise RuntimeError(
                f"Expected logits shape [1, 2], "
                f"received {tuple(logits.shape)}"
            )

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        correct += int(
            (logits.argmax(dim=1) == labels).sum().item()
        )
        count += 1

    if count == 0:
        raise RuntimeError("Training produced no graph examples")

    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "samples": count,
    }


@torch.no_grad()
def _evaluate(model, graphs, device):
    model.eval()

    loss_sum = 0.0
    correct = 0
    count = 0

    for graph in graphs:
        graph = graph.to(device)

        graph.batch = torch.zeros(
            graph.num_nodes,
            dtype=torch.long,
            device=device,
        )

        labels = graph.y.view(1).long()
        logits = model(graph)

        if logits.ndim != 2 or tuple(logits.shape) != (1, 2):
            raise RuntimeError(
                f"Expected logits shape [1, 2], "
                f"received {tuple(logits.shape)}"
            )

        loss_sum += float(
            F.cross_entropy(logits, labels).item()
        )

        correct += int(
            (logits.argmax(dim=1) == labels).sum().item()
        )

        count += 1

    if count == 0:
        raise RuntimeError("Validation produced no graph examples")

    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "samples": count,
    }


# ============================================================
# 6. MAIN PHASE 4 SMOKE TEST
# ============================================================

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
    """Prepare, audit, encode, and briefly train using existing base splits."""

    from PhpNetGraph import PhpNetGraph

    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Work on copies so the notebook's original samples are not mutated.
    base_splits = {
        "train": copy.deepcopy(list(train_base)),
        "val": copy.deepcopy(list(val_base)),
        "test": copy.deepcopy(list(test_base)),
    }

    _check_no_commit_leakage(base_splits)

    prepared = {}
    audit_snapshots = {}
    split_stats = {}

    # --------------------------------------------------------
    # 6.1 PREPARATION AND AUDIT
    # --------------------------------------------------------

    for split_name, base_samples in base_splits.items():
        base_samples = _limit_base_samples(
            base_samples,
            max_samples_per_split,
            split_name,
        )

        if not base_samples:
            raise ValueError(
                f"Base split {split_name} is empty"
            )

        # Each split receives its own pipeline and audit state.
        pipeline = pipeline_factory()

        samples = pipeline.prepare(base_samples)

        # Match the notebook's existing post-preparation cleanup.
        if deduplicate_function_samples is not None:
            samples = deduplicate_function_samples(samples)

        if validate_unique_function_samples is not None:
            validate_unique_function_samples(
                samples,
                f"PHASE4 {split_name.upper()}",
            )

        audit_snapshots[split_name] = {
            "pipeline_audit": copy.deepcopy(
                getattr(pipeline, "audit", {})
            ),
            "pairing_audit": copy.deepcopy(
                getattr(
                    getattr(
                        pipeline,
                        "paired_function_extractor",
                        None,
                    ),
                    "audit",
                    {},
                )
            ),
        }

        audit_path = output_dir / f"{split_name}_audit.json"

        with audit_path.open("w", encoding="utf-8") as file:
            json.dump(
                audit_snapshots[split_name],
                file,
                indent=2,
                default=str,
            )

        if not samples:
            raise RuntimeError(
                f"Preparation produced zero samples for "
                f"{split_name}. Inspect {audit_path}"
            )

        split_stats[split_name] = validate_sample_pairs(
            samples,
            split_name,
        )

        prepared[split_name] = list(samples)

        print(
            f"[{split_name}] prepared "
            f"{split_stats[split_name]['sample_count']} samples / "
            f"{split_stats[split_name]['pair_count']} pairs"
        )

    # --------------------------------------------------------
    # 6.2 VOCABULARY AND ENCODING
    # --------------------------------------------------------

    # Learn vocabulary from training data only.
    cfg_vocab = build_cfg_vocab(prepared["train"])

    if not cfg_vocab:
        raise RuntimeError("CFG vocabulary is empty")

    vectorizer = VUDENCVectorizer(w2v_path)

    encoder = GCNEncoder(
        vudenc_vectorizer=vectorizer,
        cfg_vocab=cfg_vocab,
    )

    encoded = {}

    for split_name, samples in prepared.items():
        encoded[split_name] = encode_samples(
            samples,
            encoder,
            split_name,
        )

        vulnerable_count = sum(
            int(graph.y.item()) == 1
            for graph in encoded[split_name]
        )

        fixed_count = sum(
            int(graph.y.item()) == 0
            for graph in encoded[split_name]
        )

        print(
            f"[{split_name}] encoded "
            f"{len(encoded[split_name])} graphs "
            f"({vulnerable_count} vulnerable, "
            f"{fixed_count} fixed)"
        )

    # --------------------------------------------------------
    # 6.3 MODEL AND SHORT TRAINING RUN
    # --------------------------------------------------------

    device = torch.device(
        device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = PhpNetGraph(
        cfg_vocab_size=len(cfg_vocab),
        word2vec_dim=200,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    history = []

    for epoch in range(1, epochs + 1):
        train_graphs = list(encoded["train"])
        random.shuffle(train_graphs)

        train_metrics = _train_one_epoch(
            model,
            train_graphs,
            optimizer,
            device,
        )

        val_metrics = _evaluate(
            model,
            encoded["val"],
            device,
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.3f} | "
            f"val loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.3f}"
        )

    # --------------------------------------------------------
    # 6.4 SAVE SUMMARY
    # --------------------------------------------------------

    result = {
        "status": "passed",
        "device": str(device),
        "batch_size_requested": batch_size,
        "training_mode": (
            "one graph at a time; variable-length node token lists"
        ),
        "cfg_vocab_size": len(cfg_vocab),
        "splits": split_stats,
        "encoded": {
            name: {
                "sample_count": len(graphs),
                "pair_count": len({
                    graph.smoke_pair_id for graph in graphs
                }),
                "positive_count": sum(
                    int(graph.y.item()) == 1
                    for graph in graphs
                ),
                "negative_count": sum(
                    int(graph.y.item()) == 0
                    for graph in graphs
                ),
            }
            for name, graphs in encoded.items()
        },
        "history": history,
        "audit_dir": str(output_dir.resolve()),
    }

    result_path = output_dir / "smoke_test_result.json"

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, default=str)

    print(f"Smoke test passed. Reports saved to: {output_dir.resolve()}")

    return result
