from collections import deque

from src.debug.pruning_validator import PruningValidator


class NeighborhoodValidator(PruningValidator):

    def __init__(self, hops):
        self.hops = hops

    def validate(self, record):

        function_nodes = set(
            record.get("function_nodes", [])
        )

        seed_nodes = set(
            record.get("seed_nodes", [])
        )

        valid_seeds = seed_nodes & function_nodes

        expected_nodes = self._compute_neighborhood(
            record["cfg"]["edges"],
            valid_seeds,
            function_nodes
        )

        retained_nodes = self._node_ids(
            record["pruned_cfg"]
        )

        valid = (
            expected_nodes == retained_nodes
        )

        if not valid_seeds:
            reason = "No valid seed nodes inside function scope"
        elif valid:
            reason = "OK"
        else:
            reason = (
                "Actual pruning differs from expected "
                "{}-hop neighborhood".format(self.hops)
            )

        return self.build_result(
            record=record,
            valid=valid,
            reason=reason,
            expected_nodes=expected_nodes,
            retained_nodes=retained_nodes,
            missing_seeds=seed_nodes - retained_nodes
        )

    def _compute_neighborhood(
        self,
        edges,
        seeds,
        function_nodes
    ):

        graph = {}

        for src, dst in edges:

            if src not in function_nodes:
                continue

            if dst not in function_nodes:
                continue

            graph.setdefault(src, []).append(dst)
            graph.setdefault(dst, []).append(src)

        visited = set(seeds)
        queue = deque(
            (seed, 0)
            for seed in seeds
        )

        while queue:

            node, distance = queue.popleft()

            if distance >= self.hops:
                continue

            for neighbor in graph.get(node, []):

                if neighbor in visited:
                    continue

                visited.add(neighbor)

                queue.append(
                    (neighbor, distance + 1)
                )

        return visited

    @staticmethod
    def _node_ids(cfg):
        return {
            node["node_id"]
            for node in cfg.get("nodes", [])
        }