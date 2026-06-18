import re


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

    return [

        node.node_id

        for node in cfg["nodes"]

        if node.lineno == line_no
    ]

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

    return {
        **cfg,
        "nodes": nodes,
        "edges": edges,
        "kept_lines": keep_lines
    }