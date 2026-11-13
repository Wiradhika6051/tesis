#!/usr/bin/env python3
"""Create Graphviz difference graphs: Identity CFG vs each pruned CFG.

Red = removed from pruned graph; green = added; grey = retained.
Node labels render as bold N<id>|<source snippet>, with type and line range below.

Requirements:
  pip install pandas
  Install Graphviz and ensure the `dot` executable is on PATH.

Run from the directory containing the CSV and JSONL files:
  python generate_pruning_difference_graphs.py \
    --selected-csv step3_selected_cases.csv \
    --graph-dir . \
    --out-dir pruning_difference_graphs

Input filenames expected in --graph-dir:
  08_pruned_cfg_identity.jsonl
  08_pruned_cfg_neighborhood_1hop.jsonl ... 08_pruned_cfg_neighborhood_6hop.jsonl
  08_pruned_cfg_backward_slicing.jsonl
  08_pruned_cfg_forward_slicing.jsonl

The script also checks for an automatic '(2)' filename suffix.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
import pandas as pd

FILES = {
    "Identity": "08_pruned_cfg_identity.jsonl",
    "Neighborhood 1-hop": "08_pruned_cfg_neighborhood_1hop.jsonl",
    "Neighborhood 2-hop": "08_pruned_cfg_neighborhood_2hop.jsonl",
    "Neighborhood 3-hop": "08_pruned_cfg_neighborhood_3hop.jsonl",
    "Neighborhood 4-hop": "08_pruned_cfg_neighborhood_4hop.jsonl",
    "Neighborhood 5-hop": "08_pruned_cfg_neighborhood_5hop.jsonl",
    "Neighborhood 6-hop": "08_pruned_cfg_neighborhood_6hop.jsonl",
    "Backward Slice": "08_pruned_cfg_backward_slicing.jsonl",
    "Forward Slice": "08_pruned_cfg_forward_slicing.jsonl",
}

def sample_id(row):
    uid = json.loads(row["sample_uid"])
    return f"{uid[0]}:{row['commit_hash']}:{row['file_path']}:{int(row['function_start'])}:{int(row['function_end'])}:{int(row['true_label'])}"

def node_map(rec):
    return {n["node_id"]: n for n in rec.get("nodes", [])}

def edge_set(rec):
    return {tuple(e) for e in rec.get("edges", [])}

def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def clip(value, limit=90):
    value = clean(value)
    return value if len(value) <= limit else value[:limit - 1] + "…"

def dq(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

def id_sort(value):
    try:
        return (0, int(value))
    except (ValueError, TypeError):
        return (1, str(value))

def html_escape(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

def node_label(node_id, node):
    code = html_escape(clip(node.get("text", ""), 90))
    node_type = html_escape(clean(node.get("node_type", "?")))
    start, end = node.get("lineno", "?"), node.get("end_lineno", "?")
    # Graphviz renders the ID in bold; the pipe replaces the old newline separator.
    return f'<<B>N{html_escape(node_id)}</B>|{code}<BR/><FONT POINT-SIZE="8">{node_type} · L{start}-{end}</FONT>>'

def create_dot(identity, pruned, title, dot_path):
    base_nodes, pruned_nodes = node_map(identity), node_map(pruned)
    base_edges, pruned_edges = edge_set(identity), edge_set(pruned)

    removed_nodes = set(base_nodes) - set(pruned_nodes)
    added_nodes = set(pruned_nodes) - set(base_nodes)
    removed_edges = base_edges - pruned_edges
    added_edges = pruned_edges - base_edges

    # Draw the union so removed items remain visible in red.
    union_nodes = dict(base_nodes)
    union_nodes.update({nid: pruned_nodes[nid] for nid in added_nodes})
    union_edges = base_edges | pruned_edges

    lines = [
        "digraph CFG {",
        'graph [rankdir=TB, bgcolor="white", pad="0.25", nodesep="0.30", ranksep="0.45", splines=true, labelloc="t", label=' + dq(title) + ', fontsize=18, fontname="Arial"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, margin="0.12,0.08"];',
        'edge [arrowsize=0.7, color="#94a3b8", penwidth=1.2];'
    ]

    for nid in sorted(union_nodes, key=id_sort):
        if nid in removed_nodes:
            attrs = 'fillcolor="#fee2e2", color="#dc2626", penwidth=2.0, fontcolor="#7f1d1d"'
        elif nid in added_nodes:
            attrs = 'fillcolor="#dcfce7", color="#16a34a", penwidth=2.0, fontcolor="#14532d"'
        else:
            attrs = 'fillcolor="#f8fafc", color="#64748b", fontcolor="#111827"'
        lines.append(f'{dq("n" + str(nid))} [label={node_label(nid, union_nodes[nid])}, {attrs}];')

    for src, dst in sorted(union_edges, key=lambda e: (id_sort(e[0]), id_sort(e[1]))):
        if (src, dst) in removed_edges:
            attrs = 'color="#dc2626", style="dashed", penwidth=2.4'
        elif (src, dst) in added_edges:
            attrs = 'color="#16a34a", style="dashed", penwidth=2.4'
        else:
            attrs = 'color="#94a3b8", style="solid", penwidth=1.1'
        lines.append(f'{dq("n" + str(src))} -> {dq("n" + str(dst))} [{attrs}];')

    lines.append("}")
    with open(dot_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return {
        "identity_nodes": len(base_nodes), "pruned_nodes": len(pruned_nodes),
        "identity_edges": len(base_edges), "pruned_edges": len(pruned_edges),
        "nodes_removed": len(removed_nodes), "edges_removed": len(removed_edges),
        "nodes_added": len(added_nodes), "edges_added": len(added_edges),
        "status": "No reported structural change" if not (
            removed_nodes or added_nodes or removed_edges or added_edges
        ) else "Structural change reported",
        "removed_node_ids": ", ".join(map(str, sorted(removed_nodes, key=id_sort))) or "None",
        "added_node_ids": ", ".join(map(str, sorted(added_nodes, key=id_sort))) or "None",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default="step3_selected_cases.csv")
    parser.add_argument("--graph-dir", default=".", help="Directory containing the JSONL graph files")
    parser.add_argument("--out-dir", default="pruning_difference_graphs")
    parser.add_argument("--zip", dest="zip_path", default="pruning_difference_graphs.zip")
    args = parser.parse_args()

    if not shutil.which("dot"):
        raise SystemExit("Graphviz 'dot' executable not found. Install Graphviz and add it to PATH.")

    paths = {}
    for name, filename in FILES.items():
        candidate = os.path.join(args.graph_dir, filename)
        if not os.path.exists(candidate):
            stem, ext = os.path.splitext(filename)
            suffixed = os.path.join(args.graph_dir, f"{stem}(2){ext}")
            if os.path.exists(suffixed):
                candidate = suffixed
        paths[name] = candidate
    missing = [p for p in [args.selected_csv, *paths.values()] if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing input file(s):\n" + "\n".join(missing))

    os.makedirs(args.out_dir, exist_ok=True)
    selected = pd.read_csv(args.selected_csv)
    records = {}
    for strategy, path in paths.items():
        records[strategy] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    records[strategy][rec["sample_id"]] = rec

    aliases = {f"Neighborhood ({i}-hop)": f"Neighborhood {i}-hop" for i in range(1, 7)}
    index_rows = []

    for index, row in selected.iterrows():
        sid = sample_id(row)
        case_no = index + 1
        if sid not in records["Identity"]:
            print(f"Skipping case {case_no}: no matching Identity graph")
            continue

        identity = records["Identity"][sid]
        selected_strategy = aliases.get(str(row.get("strategy", "")), str(row.get("strategy", "")))

        for strategy in FILES:
            if strategy == "Identity" or sid not in records[strategy]:
                continue

            stem = re.sub(r"[^A-Za-z0-9_-]+", "_",
                          f"case{case_no}_{row.get('transition', '')}_{strategy}")
            dot_path = os.path.join(args.out_dir, stem + ".dot")
            png_path = os.path.join(args.out_dir, stem + ".png")
            stats = create_dot(identity, records[strategy][sid],
                               f"Case {case_no} | {strategy} vs Identity", dot_path)
            subprocess.run(["dot", "-Tpng", "-Gdpi=140", dot_path, "-o", png_path], check=True)

            index_rows.append({
                "case": case_no, "transition": row.get("transition"),
                "true_label": row.get("true_label"), "file_path": row.get("file_path"),
                "function_start": row.get("function_start"), "function_end": row.get("function_end"),
                "selected_strategy": selected_strategy, "strategy_compared": strategy,
                **stats, "png_file": png_path, "dot_file": dot_path,
            })

    index_path = os.path.join(args.out_dir, "pruning_difference_graph_index.csv")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(index_path, arcname=os.path.basename(index_path))
        for filename in os.listdir(args.out_dir):
            if filename.endswith((".png", ".dot")):
                archive.write(os.path.join(args.out_dir, filename), arcname=filename)

    print(f"Generated {len(index_rows)} difference graphs.")
    print(f"Image and DOT files: {args.out_dir}")
    print(f"Index CSV: {index_path}")
    print(f"ZIP archive: {args.zip_path}")

if __name__ == "__main__":
    main()
