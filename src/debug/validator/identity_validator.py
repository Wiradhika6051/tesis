from src.debug.pruning_validator import PruningValidator


class IdentityValidator(PruningValidator):

    def validate(self, record):

        original_nodes = self._node_ids(
            record["cfg"]
        )

        retained_nodes = self._node_ids(
            record["pruned_cfg"]
        )

        valid = (
            original_nodes == retained_nodes
        )

        reason = (
            "OK"
            if valid
            else "Pruned node set differs from original CFG"
        )

        return self.build_result(
            record=record,
            valid=valid,
            reason=reason,
            expected_nodes=original_nodes,
            retained_nodes=retained_nodes
        )

    @staticmethod
    def _node_ids(cfg):
        return {
            node["node_id"]
            for node in cfg.get("nodes", [])
        }