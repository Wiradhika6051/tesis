import re


def normalize_code(text):

    return re.sub(
        r"\s+",
        "",
        text
    )


def find_target_nodes(
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

    return {
        **cfg,
        "nodes": nodes,
        "edges": edges
    }