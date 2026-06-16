import re
from tesis.dataset.pruning_strategies.base_strategy import (
    DatasetStrategy
)

from tesis.dataset.pruning_strategies.source_strategy import (
    reconstruct_fixed_source
)



class TaintStrategy(
    DatasetStrategy
):

    def extract(
        self,
        file
    ):

        source = file.get(
            "sourceWithComments",
            file.get(
                "source",
                ""
            )
        )

        if not source:
            return []

        fixed_source = reconstruct_fixed_source(
            source,
            file.get(
                "changes",
                []
            )
        )

        result = []

        for change in file.get(
            "changes",
            []
        ):

            badparts = change.get(
                "badparts",
                []
            )

            goodparts = change.get(
                "goodparts",
                []
            )

            pairs = min(
                len(badparts),
                len(goodparts)
            )

            for i in range(pairs):

                bad = badparts[i]
                good = goodparts[i]

                # positive
                vuln_slice = extract_taint_slice(
                    source,
                    bad
                )

                if vuln_slice:

                    result.append({
                        "code": vuln_slice,
                        "label": 1
                    })

                # negative
                fixed_slice = extract_taint_slice(
                    fixed_source,
                    good
                )

                if fixed_slice:

                    result.append({
                        "code": fixed_slice,
                        "label": 0
                    })

        return result

IDENTIFIER_RE = re.compile(
    r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
)

PYTHON_KEYWORDS = {
    "if",
    "else",
    "for",
    "while",
    "return",
    "def",
    "class",
    "import",
    "from",
    "try",
    "except",
    "with",
    "and",
    "or",
    "not",
    "in",
    "is",
    "True",
    "False",
    "None"
}

def extract_identifiers(text):

    ids = set(
        IDENTIFIER_RE.findall(text)
    )

    return {
        x
        for x in ids
        if x not in PYTHON_KEYWORDS
    }

def get_assignment_target(line):

    if "=" not in line:
        return None

    left = line.split("=")[0].strip()

    if not left.isidentifier():
        return None

    return left

def extract_taint_slice(
    source,
    vulnerable_snippet
):

    lines = source.splitlines()

    selected = []

    tainted = extract_identifiers(
        vulnerable_snippet
    )

    for line in vulnerable_snippet.splitlines():
        selected.append(line)

    changed = True

    while changed:

        changed = False

        for line in lines:

            target = get_assignment_target(
                line
            )

            if (
                target
                and target in tainted
            ):

                selected.append(line)

                before = len(tainted)

                tainted.update(
                    extract_identifiers(line)
                )

                if len(tainted) > before:
                    changed = True

    return "\n".join(
        dict.fromkeys(selected)
    )