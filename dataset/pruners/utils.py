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

    return {
        **cfg,
        "nodes": nodes,
        "edges": edges,
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