from collections import deque

from src.debug.pruning_validator import PruningValidator


class BackwardSliceValidator(PruningValidator):

    def validate(self, record):

        all_nodes = self._node_ids(
            record["cfg"]
        )

        function_nodes = set(
            record.get("function_nodes", [])
        )

        seed_nodes = set(
            record.get("seed_nodes", [])
        )

        valid_seeds = seed_nodes & function_nodes

        expected_nodes = self._compute_slice(
            record["cfg"]["edges"],
            valid_seeds,
            function_nodes
        )

        original_to_pruned = record["pruned_cfg"].get(
            "original_to_pruned",
            {}
        )

        retained_nodes = {
            int(original_id)
            for original_id in original_to_pruned.keys()
        }

        valid = (
            expected_nodes == retained_nodes
        )

        if not valid_seeds:
            reason = "No valid seed nodes inside function scope"
        elif valid:
            reason = "OK"
        else:
            reason = "Actual pruning differs from expected backward slice"

        return self.build_result(
            record=record,
            valid=valid,
            reason=reason,
            expected_nodes=expected_nodes,
            retained_nodes=retained_nodes,
            missing_seeds=valid_seeds - retained_nodes
        )

    @staticmethod
    def _compute_slice(edges, seeds, function_nodes):

        reverse_graph = {}

        for src, dst in edges:
            reverse_graph.setdefault(dst, []).append(src)

        visited = set(seeds)
        queue = deque(seeds)

        while queue:

            node = queue.popleft()

            for parent in reverse_graph.get(node, []):

                if parent not in function_nodes:
                    continue

                if parent in visited:
                    continue

                visited.add(parent)
                queue.append(parent)

        return visited

    @staticmethod
    def _node_ids(cfg):
        return {
            node["node_id"]
            for node in cfg.get("nodes", [])
        }