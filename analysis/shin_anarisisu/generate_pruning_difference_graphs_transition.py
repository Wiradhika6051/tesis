#!/usr/bin/env python3
"""Generate CFG difference graphs for selected prediction-transition cases.

Based on the supplied generate_pruning_difference_graphs.py.
Expected Graphviz `dot` executable on PATH and pandas installed.

Example:
python generate_pruning_difference_graphs_transition.py \
  --selected-csv transition_graph_cases.csv \
  --graph-dir . \
  --out-dir pruning_transition_graphs \
  --zip pruning_transition_graphs.zip

The resolver accepts common names including *_test.jsonl and automatic suffixes such as (1).
Each selected case is compared against every available pruning strategy, so use a shortlist.
"""
import argparse, json, os, re, shutil, subprocess, zipfile, glob
import pandas as pd

FILES = {
    "Identity": ["08_pruned_cfg_identity_test.jsonl", "08_pruned_cfg_identity.jsonl"],
    "Neighborhood 1-hop": ["08_pruned_cfg_neighborhood_1hop_test.jsonl", "08_pruned_cfg_neighborhood_1hop.jsonl"],
    "Neighborhood 2-hop": ["08_pruned_cfg_neighborhood_2hop_test.jsonl", "08_pruned_cfg_neighborhood_2hop.jsonl"],
    "Neighborhood 3-hop": ["08_pruned_cfg_neighborhood_3hop_test.jsonl", "08_pruned_cfg_neighborhood_3hop.jsonl"],
    "Neighborhood 4-hop": ["08_pruned_cfg_neighborhood_4hop_test.jsonl", "08_pruned_cfg_neighborhood_4hop.jsonl"],
    "Neighborhood 5-hop": ["08_pruned_cfg_neighborhood_5hop_test.jsonl", "08_pruned_cfg_neighborhood_5hop.jsonl"],
    "Neighborhood 6-hop": ["08_pruned_cfg_neighborhood_6hop_test.jsonl", "08_pruned_cfg_neighborhood_6hop.jsonl"],
    "Backward Slice": ["08_pruned_cfg_backward_slicing_test.jsonl", "08_pruned_cfg_backward_slicing.jsonl"],
    "Forward Slice": ["08_pruned_cfg_forward_slicing_test.jsonl", "08_pruned_cfg_forward_slicing.jsonl"],
}

def sample_id(row):
    uid = json.loads(row["sample_uid"])
    label = row.get("true_label", row.get("label"))
    return f"{uid[0]}:{row['commit_hash']}:{row['file_path']}:{int(row['function_start'])}:{int(row['function_end'])}:{int(label)}"

def node_map(rec):
    return {str(n["node_id"]): n for n in rec.get("nodes", [])}

def edge_set(rec):
    return {tuple(map(str, e)) for e in rec.get("edges", [])}

def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def clip(value, limit=90):
    value = clean(value)
    return value if len(value) <= limit else value[:limit - 1] + "…"

def dq(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

def id_sort(value):
    try: return (0, int(value))
    except (ValueError, TypeError): return (1, str(value))

def html_escape(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

def node_label(node_id, node):
    code = html_escape(clip(node.get("text", ""), 90))
    node_type = html_escape(clean(node.get("node_type", "?")))
    start, end = node.get("lineno", "?"), node.get("end_lineno", "?")
    return f'<<B>N{html_escape(node_id)}</B>|{code}<BR/><FONT POINT-SIZE="8">{node_type} · L{start}-{end}</FONT>>'

def create_dot(identity, pruned, title, dot_path):
    base_nodes, pruned_nodes = node_map(identity), node_map(pruned)
    base_edges, pruned_edges = edge_set(identity), edge_set(pruned)
    removed_nodes = set(base_nodes) - set(pruned_nodes)
    added_nodes = set(pruned_nodes) - set(base_nodes)
    removed_edges = base_edges - pruned_edges
    added_edges = pruned_edges - base_edges
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
    with open(dot_path, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
    return {
        "identity_nodes": len(base_nodes), "pruned_nodes": len(pruned_nodes),
        "identity_edges": len(base_edges), "pruned_edges": len(pruned_edges),
        "nodes_removed": len(removed_nodes), "edges_removed": len(removed_edges),
        "nodes_added": len(added_nodes), "edges_added": len(added_edges),
        "status": "No reported structural change" if not (removed_nodes or added_nodes or removed_edges or added_edges) else "Structural change reported",
        "removed_node_ids": ", ".join(map(str, sorted(removed_nodes, key=id_sort))) or "None",
        "added_node_ids": ", ".join(map(str, sorted(added_nodes, key=id_sort))) or "None",
    }

def resolve_file(graph_dir, candidates):
    for name in candidates:
        direct = os.path.join(graph_dir, name)
        if os.path.exists(direct): return direct
        stem, ext = os.path.splitext(name)
        matches = sorted(glob.glob(os.path.join(graph_dir, stem + "*" + ext)))
        # Prefer a file with exact expected base, then test split, then lowest suffix.
        if matches:
            matches.sort(key=lambda p: (("(1)" in p), ("_test" not in p), len(os.path.basename(p))))
            return matches[0]
    return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default="transition_graph_cases.csv")
    parser.add_argument("--graph-dir", default=".")
    parser.add_argument("--out-dir", default="pruning_transition_graphs")
    parser.add_argument("--zip", dest="zip_path", default="pruning_transition_graphs.zip")
    args = parser.parse_args()
    if not shutil.which("dot"):
        raise SystemExit("Graphviz 'dot' executable not found. Install Graphviz and add it to PATH.")
    selected = pd.read_csv(args.selected_csv)
    paths = {name: resolve_file(args.graph_dir, candidates) for name, candidates in FILES.items()}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise SystemExit("Missing graph JSONL for: " + ", ".join(missing))
    os.makedirs(args.out_dir, exist_ok=True)
    records = {}
    for strategy, path in paths.items():
        records[strategy] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records[strategy][rec["sample_id"]] = rec
    index_rows = []
    for idx, row in selected.iterrows():
        sid = sample_id(row)
        case_no = idx + 1
        if sid not in records["Identity"]:
            print(f"Skipping case {case_no}: no matching Identity graph: {sid}")
            continue
        identity = records["Identity"][sid]
        selected_strategy = str(row.get("strategy", ""))
        transition = str(row.get("transition", ""))
        category = str(row.get("category", ""))
        for strategy in FILES:
            if strategy == "Identity" or sid not in records[strategy]:
                continue
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", f"case{case_no}_{transition}_{strategy}")
            dot_path = os.path.join(args.out_dir, safe + ".dot")
            png_path = os.path.join(args.out_dir, safe + ".png")
            stats = create_dot(identity, records[strategy][sid],
                               f"Case {case_no} | {category} | {transition} | {strategy} vs Identity", dot_path)
            subprocess.run(["dot", "-Tpng", "-Gdpi=140", dot_path, "-o", png_path], check=True)
            index_rows.append({
                "case": case_no, "transition": transition, "category": category,
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
    print(f"Graph folder: {args.out_dir}")
    print(f"Index CSV: {index_path}")
    print(f"ZIP archive: {args.zip_path}")

if __name__ == "__main__":
    main()
