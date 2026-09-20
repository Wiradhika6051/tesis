import csv
import json
import os


class PruningValidationRunner:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, records, validator, name):

        results = []

        for record in records:

            result = validator.validate(record)

            results.append(result)

        output_path = os.path.join(
            self.output_dir,
            "{}.csv".format(
                self._safe_name(name)
            )
        )

        self._write_csv(
            output_path,
            results
        )

        self._print_summary(
            name,
            results
        )

        return results

    def _write_csv(self, path, results):

        fieldnames = [
            "sample_id",
            "repo",
            "file_path",
            "commit",
            "label",
            "valid",
            "reason",
            "retention_ratio",
            "missing_seeds",
            "missing_nodes",
            "unexpected_nodes",
            "expected_nodes",
            "retained_nodes"
        ]

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for result in results:

                row = dict(result)

                for field in [
                    "missing_seeds",
                    "missing_nodes",
                    "unexpected_nodes",
                    "expected_nodes",
                    "retained_nodes"
                ]:
                    row[field] = json.dumps(
                        row[field]
                    )

                writer.writerow(row)

    @staticmethod
    def _print_summary(name, results):

        total = len(results)

        valid = sum(
            1
            for result in results
            if result["valid"]
        )

        invalid = total - valid

        percentage = (
            100.0 * valid / total
            if total
            else 0.0
        )

        print()
        print(
            "[PRUNING VALIDATION] {}".format(name)
        )
        print(
            "  Total:   {}".format(total)
        )
        print(
            "  Valid:   {}".format(valid)
        )
        print(
            "  Invalid: {}".format(invalid)
        )
        print(
            "  Validity: {:.2f}%".format(
                percentage
            )
        )

    @staticmethod
    def _safe_name(name):
        return (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )