class IdentityValidator(PruningValidator):

    def validate(self, record):

        expected_nodes = set(
            record.get("function_nodes", [])
        )

        original_to_pruned = record[
            "pruned_cfg"
        ].get(
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

        if valid:
            reason = "OK"
        else:
            reason = (
                "Actual pruning differs "
                "from function scope"
            )

        return self.build_result(
            record=record,
            valid=valid,
            reason=reason,
            expected_nodes=expected_nodes,
            retained_nodes=retained_nodes,
            missing_seeds=(
                set(record.get("seed_nodes", []))
                - retained_nodes
            )
        )