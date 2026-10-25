import ast
import hashlib

from collections import defaultdict
from difflib import SequenceMatcher


class PairedFunctionExtractor:

    def __init__(
        self,
        similarity_threshold=0.85,
        minimum_similarity_margin=0.10,
    ):
        self.similarity_threshold = similarity_threshold
        self.minimum_similarity_margin = minimum_similarity_margin
        self.audit = {"counts": {}}

    def _count(self, reason, amount=1):
        counts = self.audit["counts"]
        counts[reason] = counts.get(reason, 0) + amount

    def extract(self, function_samples):
        self.audit = {"counts": {}}

        groups = defaultdict(list)

        for sample in function_samples:
            key = (
                sample.repo,
                sample.fix_commit_hash,
                sample.file_path,
            )
            groups[key].append(sample)

        paired_samples = []

        for group_key, candidates in groups.items():
            vulnerable = [
                s for s in candidates
                if s.revision == "vulnerable" and s.label == 1
            ]
            fixed = [
                s for s in candidates
                if s.revision == "fixed" and s.label == 0
            ]

            if not vulnerable or not fixed:
                self._count("missing_revision_candidates")
                continue

            matched, ambiguous_old, ambiguous_new = self._match(
                vulnerable, fixed
            )

            self._count("ambiguous_vulnerable", len(ambiguous_old))
            self._count("ambiguous_fixed", len(ambiguous_new))

            used_old = set()
            used_new = set()

            for old, new in matched:
                used_old.add(id(old))
                used_new.add(id(new))

                # Require a changed region on at least one side.
                if not old.seed_lines and not new.seed_lines:
                    self._count("unchanged_pairs_excluded")
                    continue

                pair_id = self._pair_id(group_key, old, new)

                old.function_pair_id = pair_id
                old.pair_role = "vulnerable"

                new.function_pair_id = pair_id
                new.pair_role = "fixed"

                paired_samples.extend([old, new])
                self._count("accepted_pairs")

            self._count(
                "unmatched_vulnerable",
                sum(id(s) not in used_old for s in vulnerable),
            )
            self._count(
                "unmatched_fixed",
                sum(id(s) not in used_new for s in fixed),
            )

        return paired_samples

    def _match(self, vulnerable, fixed):
        matched = []
        used_old = set()
        used_new = set()

        old_by_name = defaultdict(list)
        new_by_name = defaultdict(list)

        for sample in vulnerable:
            old_by_name[sample.qualified_name].append(sample)

        for sample in fixed:
            new_by_name[sample.qualified_name].append(sample)

        # First: unique qualified-name matches.
        for name, old_candidates in old_by_name.items():
            new_candidates = new_by_name.get(name, [])

            if len(old_candidates) == 1 and len(new_candidates) == 1:
                old = old_candidates[0]
                new = new_candidates[0]

                matched.append((old, new))
                used_old.add(id(old))
                used_new.add(id(new))

        remaining_old = [
            s for s in vulnerable if id(s) not in used_old
        ]
        remaining_new = [
            s for s in fixed if id(s) not in used_new
        ]

        # Flag duplicate qualified names as ambiguous rather than
        # arbitrarily choosing one of several definitions.
        ambiguous_old = [
            s for s in remaining_old
            if len(old_by_name[s.qualified_name]) > 1
        ]
        ambiguous_new = [
            s for s in remaining_new
            if len(new_by_name[s.qualified_name]) > 1
        ]

        blocked_old = {id(s) for s in ambiguous_old}
        blocked_new = {id(s) for s in ambiguous_new}

        remaining_old = [
            s for s in remaining_old if id(s) not in blocked_old
        ]
        remaining_new = [
            s for s in remaining_new if id(s) not in blocked_new
        ]

        # Second: conservative AST similarity matching for renames.
        scores = {}

        for old in remaining_old:
            for new in remaining_new:
                score = self._similarity(
                    old.function_source or "",
                    new.function_source or "",
                )
                scores[(id(old), id(new))] = score

        best_new = {}
        for old in remaining_old:
            ranked = sorted(
                (
                    (scores[(id(old), id(new))], new)
                    for new in remaining_new
                ),
                key=lambda item: item[0],
                reverse=True,
            )

            if self._is_confident(ranked):
                best_new[id(old)] = ranked[0][1]

        best_old = {}
        for new in remaining_new:
            ranked = sorted(
                (
                    (scores[(id(old), id(new))], old)
                    for old in remaining_old
                ),
                key=lambda item: item[0],
                reverse=True,
            )

            if self._is_confident(ranked):
                best_old[id(new)] = ranked[0][1]

        for old in remaining_old:
            new = best_new.get(id(old))

            if new is None:
                continue

            if best_old.get(id(new)) is new and id(new) not in used_new:
                matched.append((old, new))
                used_old.add(id(old))
                used_new.add(id(new))

        return matched, ambiguous_old, ambiguous_new

    def _is_confident(self, ranked):
        if not ranked:
            return False

        best_score = ranked[0][0]

        if best_score < self.similarity_threshold:
            return False

        second_score = ranked[1][0] if len(ranked) > 1 else 0.0

        return (
            best_score - second_score
            >= self.minimum_similarity_margin
        )

    @staticmethod
    def _normalized_ast(source):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        # Ignore the root function name so renamed functions can match.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.name = "<FUNCTION>"
                break

        return ast.dump(tree, include_attributes=False)

    def _similarity(self, old_source, new_source):
        old_ast = self._normalized_ast(old_source)
        new_ast = self._normalized_ast(new_source)

        if old_ast is None or new_ast is None:
            return 0.0

        return SequenceMatcher(
            None, old_ast, new_ast
        ).ratio()

    @staticmethod
    def _pair_id(group_key, old, new):
        raw = "|".join([
            *group_key,
            old.qualified_name or "",
            new.qualified_name or "",
            str(old.function_start),
            str(new.function_start),
        ])

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]