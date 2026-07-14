import re
CFG_STATS = {
    "target_found": 0,
    "target_missing": 0,
    "empty_after_prune": 0
}

def normalize_code(text):

    return re.sub(
        r"\s+",
        "",
        text
    )

import re


def extract_changed_lines(
    diff,
    label
):
    """
    Extract changed line numbers from a unified diff.

    label == 1 -> vulnerable (old file)
    label == 0 -> fixed (new file)
    """

    changed = []

    old_line = None
    new_line = None

    hunk_pattern = re.compile(
        r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@"
    )

    for line in diff.splitlines():

        match = hunk_pattern.match(line)

        if match:

            old_line = int(match.group(1))
            new_line = int(match.group(2))
            continue

        #
        # Not inside a hunk yet
        #
        if old_line is None:
            continue

        if line.startswith("-") and not line.startswith("---"):

            if label == 1:
                changed.append(old_line)

            old_line += 1
            continue

        if line.startswith("+") and not line.startswith("+++"):

            if label == 0:
                changed.append(new_line)

            new_line += 1
            continue

        #
        # Context
        #
        old_line += 1
        new_line += 1
    return changed

def find_snippet_line(
    source,
    snippet
):

    source_lines = source.splitlines()

    normalized_source = re.sub(
        r"\s+",
        "",
        source
    )

    normalized_snippet = re.sub(
        r"\s+",
        "",
        snippet
    )

    pos = normalized_source.find(
        normalized_snippet
    )

    if pos == -1:
        return None

    current_pos = 0

    for idx, line in enumerate(source_lines):

        current_pos += len(
            re.sub(
                r"\s+",
                "",
                line
            )
        )

        if current_pos >= pos:

            return idx + 1

    return None

def find_target_nodes(
    cfg,
    diff,
    label
):

    target_lines = extract_changed_lines(
        diff,
        label
    )

    if not target_lines:
        return []

    target_nodes = []

    #
    # Exact line match
    #
    for node in cfg["nodes"]:

        if node.lineno in target_lines:

            target_nodes.append(
                node.node_id
            )

    #
    # Fallback:
    # closest node to each changed line
    #
    if not target_nodes:

        for target_line in target_lines:

            best_dist = float("inf")
            best = None

            for node in cfg["nodes"]:

                if node.lineno < 0:
                    continue

                dist = abs(
                    node.lineno -
                    target_line
                )

                if dist < best_dist:

                    best_dist = dist
                    best = node.node_id

            if best is not None:

                target_nodes.append(best)

    return list(
        set(target_nodes)
    )

def old_find_target_nodes(
    cfg,
    source,
    snippet
):


    line_no = find_snippet_line(
        source,
        snippet
    )

    if line_no is None:
        return []

    best_dist = float("inf")
    best_nodes = []

    for node in cfg["nodes"]:

        if node.lineno < 0:
            continue

        dist = abs(
            node.lineno - line_no
        )

        if dist < best_dist:

            best_dist = dist

            best_nodes = [
                node.node_id
            ]

        elif dist == best_dist:

            best_nodes.append(
                node.node_id
            )
    if not best_nodes:

        CFG_STATS[
            "target_missing"
        ] += 1

    else:

        CFG_STATS[
            "target_found"
        ] += 1
    return best_nodes

def old_2_find_target_nodes(
    cfg,
    source,
    snippet
):

    line_no = find_snippet_line(
        source,
        snippet
    )

    if line_no is None:
        print("LINE NOT FOUND")
        print(snippet[:100])
        return []

    targets = [
        node.node_id
        for node in cfg["nodes"]
        if node.lineno == line_no
    ]

    if len(targets) == 0:
        print(
            f"NO NODE AT LINE {line_no}"
        )

    return targets

def old_find_target_nodes(
    cfg,
    snippet
):

    target = normalize_code(
        snippet
    )

    result = []

    for node in cfg["nodes"]:

        if target in normalize_code(
            node.text
        ):

            result.append(
                node.node_id
            )

    return result

def prune_cfg(
    cfg,
    keep_nodes
):

    nodes = [

        n

        for n in cfg["nodes"]

        if n.node_id in keep_nodes
    ]

    edges = [

        e

        for e in cfg["edges"]

        if (
            e[0] in keep_nodes
            and
            e[1] in keep_nodes
        )
    ]

    keep_lines = {
        node.lineno
        for node in nodes
    }
    if len(nodes) == 0:

        CFG_STATS[
            "empty_after_prune"
        ] += 1

    old_to_new = {}

    for new_id, node in enumerate(nodes):

        old_to_new[
            node.node_id
        ] = new_id

    new_edges = []

    for src, dst in edges:

        new_edges.append(
            (
                old_to_new[src],
                old_to_new[dst]
            )
        )
    for new_id, node in enumerate(nodes):

        node.node_id = new_id
    return {
        **cfg,
        "nodes": nodes,
        "edges": new_edges,
        "kept_lines": keep_lines
    }


def prune_source_by_lines(
    source,
    keep_lines
):
    if not keep_lines:
        return source
    source_lines = source.splitlines()

    result = []

    for idx, line in enumerate(
        source_lines,
        start=1
    ):

        if idx in keep_lines:

            result.append(
                line
            )

    return "\n".join(
        result
    )