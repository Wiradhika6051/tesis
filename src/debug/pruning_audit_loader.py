import json


class PruningAuditLoader:

    @staticmethod
    def load(path):
        records = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                records.append(json.loads(line))

        print(
            "[PRUNING AUDIT] Loaded {} records <- {}".format(
                len(records),
                path
            )
        )

        return records