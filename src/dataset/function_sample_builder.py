import ast
import copy

from src.cfg.cfg_builder import CFGBuilder


class FunctionSampleBuilder:

    def build(self, sample):
        try:
            tree = ast.parse(sample.source)
        except SyntaxError:
            return []

        functions = self._find_functions(tree, sample.source)
        changed_seed_lines = set(sample.seed_lines or [])
        function_samples = []
        # print("SOURCE:\n", sample.source)
        print("SAMPLE SEED LINES:", sample.seed_lines)
        print("FUNCTIONS:")
        for function in functions:
            print(
                function["qualified_name"],
                function["start"],
                function["end"],
                function["body_lines"],
            )
        for function in functions:
            changed_lines = sorted(
                changed_seed_lines & function["body_lines"]
            )

            function_sample = copy.deepcopy(sample)

            # May be empty: retained so pairing can find the counterpart.
            function_sample.seed_lines = changed_lines

            function_sample.function_name = function["name"]
            function_sample.qualified_name = function["qualified_name"]
            function_sample.function_start = function["start"]
            function_sample.function_end = function["end"]
            function_sample.function_source = function["source"]
            function_sample.function_is_async = function["is_async"]
            function_sample.function_body_lines = sorted(function["body_lines"])

            function_sample.parent_sample_id = self._sample_id(sample)
            function_sample.sample_id = self._function_sample_id(
                sample, function
            )

            function_samples.append(function_sample)

        return function_samples

    @staticmethod
    def _find_functions(tree, source):
        functions = []

        def visit(node, scope):
            for child in ast.iter_child_nodes(node):

                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = child.lineno
                    
                    end = max(
                        (
                            getattr(descendant, "lineno", start)
                            for descendant in ast.walk(child)
                            if hasattr(descendant, "lineno")
                        ),
                        default=start,
                    )


                    nested_spans = []

                    for descendant in ast.walk(child):
                        if descendant is child:
                            continue

                        if isinstance(
                            descendant,
                            (
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                                ast.ClassDef,
                            ),
                        ):
                            nested_end = max(
                                (
                                    getattr(node, "lineno", descendant.lineno)
                                    for node in ast.walk(descendant)
                                    if hasattr(node, "lineno")
                                ),
                                default=descendant.lineno,
                            )

                            nested_spans.append((
                                descendant.lineno,
                                nested_end,
                            ))


                    body_lines = set(range(start, end + 1))

                    for nested_start, nested_end in nested_spans:
                        body_lines.difference_update(
                            range(nested_start, nested_end + 1)
                        )
                    builder = CFGBuilder()
                    builder.source = source
                    functions.append({
                        "name": child.name,
                        "qualified_name": ".".join(scope + [child.name]),
                        "start": start,
                        "end": end,
                        "body_lines": body_lines,
                        # "source": ast.get_source_segment_(source, child) or "",
                        "source": builder.get_source_segment_compat(child) or "",
                        "is_async": isinstance(child, ast.AsyncFunctionDef),
                    })

                    visit(child, scope + [child.name])

                elif isinstance(child, ast.ClassDef):
                    visit(child, scope + [child.name])

                else:
                    visit(child, scope)

        visit(tree, [])
        return functions

    @staticmethod
    def _sample_id(sample):
        return "{}:{}:{}".format(
            sample.repo,
            sample.commit_hash,
            sample.file_path,
        )

    @staticmethod
    def _function_sample_id(sample, function):
        return "{}:{}:{}:{}:{}".format(
            sample.repo,
            sample.commit_hash,
            sample.file_path,
            function["qualified_name"],
            function["start"],
        )