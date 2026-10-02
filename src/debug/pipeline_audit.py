
import json
import os
from pathlib import Path

import numpy as np


class PipelineAuditExporter:

    def __init__(self, output_dir, include_vectors=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # False: export metadata and shapes only.
        # True: include actual Word2Vec vectors in JSONL.
        self.include_vectors = include_vectors

    # ============================================================
    # Public API
    # ============================================================

    def export_raw_samples(self, samples):
        self._export_jsonl(
            "01_raw_samples.jsonl",
            [self._serialize_raw_sample(s) for s in samples]
        )

    def export_diff_localized(self, samples):
        self._export_jsonl(
            "02_diff_localized.jsonl",
            [
                self._serialize_diff_sample(s)
                for s in samples
                if getattr(s, "seed_lines", None)
            ]
        )

    def export_function_samples(self, samples):
        self._export_jsonl(
            "03_function_samples.jsonl",
            [self._serialize_function_sample(s) for s in samples]
        )

    def export_cfg(self, samples):
        self._export_jsonl(
            "04_cfg.jsonl",
            [
                self._serialize_cfg_sample(s)
                for s in samples
                if getattr(s, "cfg", None)
            ]
        )

    def export_localized(self, samples):
        self._export_jsonl(
            "05_seed_nodes.jsonl",
            [
                self._serialize_localized_sample(s)
                for s in samples
                if getattr(s, "cfg", None)
            ]
        )

    def export_function_scope(self, samples):
        self._export_jsonl(
            "06_function_scope.jsonl",
            [self._serialize_function_scope(s) for s in samples]
        )

    def export_pruned(self, samples, strategy_name):
        filename = "07_pruned_cfg_{}.jsonl".format(
            self._safe_name(strategy_name)
        )

        self._export_jsonl(
            filename,
            [
                self._serialize_pruned_sample(s)
                for s in samples
                if getattr(s, "pruned_cfg", None)
            ]
        )

    def export_tokens(self, samples):
        self._export_jsonl(
            "08_tokens.jsonl",
            [
                self._serialize_tokens(s)
                for s in samples
                if hasattr(s, "tokens")
            ]
        )

    def export_word2vec_vectors(self, samples, architecture="bilstm"):
        """
        Export vectorization metadata.

        BiLSTM:
            sample.encoded = [sequence_length, embedding_dim]

        GCN:
            sample.graph.node_token_vectors = list of
            [tokens_in_node, embedding_dim] tensors.
        """
        records = []

        for sample in samples:
            if architecture.lower() == "gcn":
                record = self._serialize_gcn_vectors(sample)
            else:
                record = self._serialize_bilstm_vectors(sample)

            if record is not None:
                records.append(record)

        self._export_jsonl("09_word2vec_vectors.jsonl", records)

    def export_encoded(self, samples, architecture="bilstm"):
        self._export_jsonl(
            "10_encoded_samples.jsonl",
            [
                self._serialize_encoded(s, architecture)
                for s in samples
            ]
        )

    # ============================================================
    # Serialization: Existing pipeline stages
    # ============================================================

    def _serialize_raw_sample(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "repo": getattr(sample, "repo", ""),
            "parent_commit": getattr(sample, "parent_commit", ""),
            "commit": getattr(sample, "commit", ""),
            "file_path": getattr(sample, "file_path", ""),
            "label": getattr(sample, "label", None),
            "source": getattr(sample, "source", ""),
            "diff": getattr(sample, "diff", ""),
        }

    def _serialize_diff_sample(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "repo": getattr(sample, "repo", ""),
            "parent_commit": getattr(sample, "parent_commit", ""),
            "commit": getattr(sample, "commit", ""),
            "file_path": getattr(sample, "file_path", ""),
            "label": getattr(sample, "label", None),
            "seed_lines": list(getattr(sample, "seed_lines", [])),
        }

    def _serialize_function_sample(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "repo": getattr(sample, "repo", ""),
            "parent_commit": getattr(sample, "parent_commit", ""),
            "commit": getattr(sample, "commit", ""),
            "file_path": getattr(sample, "file_path", ""),
            "label": getattr(sample, "label", None),
            "function_name": getattr(sample, "function_name", None),
            "function_start": getattr(sample, "function_start", None),
            "function_end": getattr(sample, "function_end", None),
            "seed_lines": list(getattr(sample, "seed_lines", [])),
        }

    def _serialize_cfg_sample(self, sample):
        cfg = sample.cfg

        return {
            "sample_id": self._sample_id(sample),
            "function_name": getattr(sample, "function_name", None),
            "function_start": getattr(sample, "function_start", None),
            "function_end": getattr(sample, "function_end", None),
            "nodes": self._serialize_nodes(cfg),
            "edges": self._serialize_edges(cfg),
        }

    def _serialize_localized_sample(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "seed_lines": list(getattr(sample, "seed_lines", [])),
            "seed_nodes": list(getattr(sample, "seed_nodes", [])),
            "line_to_node": getattr(sample, "line_to_node", {}),
        }

    def _serialize_function_scope(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "function_name": getattr(sample, "function_name", None),
            "function_start": getattr(sample, "function_start", None),
            "function_end": getattr(sample, "function_end", None),
            "function_nodes": list(getattr(sample, "function_nodes", [])),
            "seed_nodes": list(getattr(sample, "seed_nodes", [])),
        }

    def _serialize_pruned_sample(self, sample):
        cfg = sample.pruned_cfg

        return {
            "sample_id": self._sample_id(sample),
            "seed_lines": list(getattr(sample, "seed_lines", [])),
            "seed_nodes": list(getattr(sample, "seed_nodes", [])),
            "function_nodes": list(getattr(sample, "function_nodes", [])),
            "nodes": self._serialize_nodes(cfg),
            "edges": self._serialize_edges(cfg),
            "original_to_pruned": {
                str(k): v
                for k, v in cfg.get("original_to_pruned", {}).items()
            },
        }

    # ============================================================
    # Tokenization
    # ============================================================

    def _serialize_tokens(self, sample):
        tokens = list(getattr(sample, "tokens", []))

        return {
            "sample_id": self._sample_id(sample),
            "tokens": tokens,
            "token_count": len(tokens),
        }

    # ============================================================
    # Word2Vec vectorization
    # ============================================================

    def _serialize_bilstm_vectors(self, sample):
        encoded = getattr(sample, "encoded", None)

        if encoded is None:
            return None

        array = self._to_numpy(encoded)

        if array is None:
            return None

        record = {
            "sample_id": self._sample_id(sample),
            "architecture": "bilstm",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nonzero_vectors": int(np.any(array != 0, axis=-1).sum()),
            "zero_vectors": int(np.all(array == 0, axis=-1).sum()),
        }

        if self.include_vectors:
            record["vectors"] = array.tolist()

        return record

    def _serialize_gcn_vectors(self, sample):
        graph = getattr(sample, "graph", None)

        if graph is None:
            return None

        node_vectors = getattr(graph, "node_token_vectors", None)

        if node_vectors is None:
            return None

        node_records = []

        for index, tensor in enumerate(node_vectors):
            array = self._to_numpy(tensor)

            if array is None:
                continue

            node_record = {
                "node_index": index,
                "shape": list(array.shape),
                "token_count": int(array.shape[0]) if array.ndim > 0 else 0,
            }

            if self.include_vectors:
                node_record["vectors"] = array.tolist()

            node_records.append(node_record)

        edge_index = self._to_numpy(
            getattr(graph, "edge_index", None)
        )

        node_types = self._to_numpy(
            getattr(graph, "node_types", None)
        )

        return {
            "sample_id": self._sample_id(sample),
            "architecture": "gcn",
            "node_count": len(node_vectors),
            "edge_count": (
                int(edge_index.shape[1])
                if edge_index is not None and edge_index.ndim == 2
                else 0
            ),
            "node_types": (
                node_types.tolist()
                if node_types is not None
                else []
            ),
            "nodes": node_records,
            "edges": (
                edge_index.T.tolist()
                if edge_index is not None and edge_index.ndim == 2
                else []
            ),
        }

    # ============================================================
    # Encoded sample metadata
    # ============================================================

    def _serialize_encoded(self, sample, architecture):
        result = {
            "sample_id": self._sample_id(sample),
            "architecture": architecture.lower(),
            "label": getattr(sample, "label", None),
        }

        if architecture.lower() == "gcn":
            graph = getattr(sample, "graph", None)

            if graph is not None:
                node_vectors = getattr(
                    graph,
                    "node_token_vectors",
                    []
                )

                result["node_count"] = len(node_vectors)
                result["node_type_shape"] = list(
                    graph.node_types.shape
                )
                result["edge_index_shape"] = list(
                    graph.edge_index.shape
                )

                result["node_token_shapes"] = [
                    list(tensor.shape)
                    for tensor in node_vectors
                ]

        else:
            encoded = getattr(sample, "encoded", None)

            if encoded is not None:
                array = self._to_numpy(encoded)

                if array is not None:
                    result["shape"] = list(array.shape)
                    result["dtype"] = str(array.dtype)

        return result

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()

        return np.asarray(value)

    @staticmethod
    def _serialize_nodes(cfg):
        result = []

        for node in cfg.get("nodes", []):
            if isinstance(node, dict):
                result.append(node)
            else:
                result.append({
                    "node_id": node.node_id,
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "node_type": node.node_type,
                    "text": node.text,
                })

        return result

    @staticmethod
    def _serialize_edges(cfg):
        return [list(edge) for edge in cfg.get("edges", [])]

    @staticmethod
    def _sample_id(sample):
        return "{}:{}:{}:{}:{}:{}".format(
            getattr(sample, "repo", ""),
            getattr(
                sample,
                "parent_commit",
                getattr(sample, "commit", "")
            ),
            getattr(sample, "file_path", ""),
            getattr(sample, "function_start", None),
            getattr(sample, "function_end", None),
            getattr(sample, "label", None),
        )

    @staticmethod
    def _safe_name(name):
        return (
            str(name).lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

    def _export_jsonl(self, filename, records):
        path = self.output_dir / filename

        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

        print(
            "[PIPELINE AUDIT] {} records -> {}".format(
                len(records),
                path
            )
        )

        return str(path)