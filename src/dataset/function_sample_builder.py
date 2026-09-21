import ast
import copy


class FunctionSampleBuilder:

    def build(self, sample):
        """
        Split a file-level sample into function-level samples.

        All changed lines belonging to the same function are combined
        into a single sample.

        Returns:
            list of function-level samples
        """

        if not sample.seed_lines:
            return []

        try:
            tree = ast.parse(sample.source)
        except SyntaxError:
            return []

        functions = self._find_functions(tree)

        function_samples = []

        for function in functions:

            changed_lines = [
                line
                for line in sample.seed_lines
                if function["start"] <= line <= function["end"]
            ]

            if not changed_lines:
                continue

            function_sample = copy.deepcopy(sample)

            function_sample.seed_lines = sorted(
                set(changed_lines)
            )

            function_sample.function_name = function["name"]
            function_sample.function_start = function["start"]
            function_sample.function_end = function["end"]

            function_sample.parent_sample_id = (
                self._sample_id(sample)
            )

            function_sample.sample_id = (
                self._function_sample_id(
                    sample,
                    function
                )
            )

            function_samples.append(function_sample)

        return function_samples

    @staticmethod
    def _find_functions(tree):

        functions = []

        for node in ast.walk(tree):

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                )
            ):
                continue

            start = node.lineno
            end = getattr(
                node,
                "end_lineno",
                start
            )

            functions.append({
                "name": node.name,
                "start": start,
                "end": end,
            })

        return functions

    @staticmethod
    def _sample_id(sample):

        return "{}:{}:{}".format(
            getattr(sample, "repo", ""),
            getattr(sample, "commit", ""),
            getattr(sample, "file_path", ""),
        )

    @staticmethod
    def _function_sample_id(sample, function):

        return "{}:{}:{}:{}:{}".format(
            getattr(sample, "repo", ""),
            getattr(sample, "commit", ""),
            getattr(sample, "file_path", ""),
            function["name"],
            function["start"],
        )