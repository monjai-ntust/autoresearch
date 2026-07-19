"""
Evidence-path coverage diagnostic for CODE-ACCORD Graph RAG.

This is a Phase B diagnostic inspired by cross-document RE path-ranking work:
before changing training or prompts, measure whether the extracted KG contains
document-local paths that can reach each gold answer tail from the gold head.
Low recall here means the bottleneck is extraction/KG coverage, not generation.
"""
import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kg", required=True, help="Constructed KG JSON")
    p.add_argument("--gold-jsonl", required=True, help="Inference JSONL with gold triples")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--similarity-threshold", type=float, default=0.72)
    p.add_argument("--output", default="results/evidence_path_diagnostic.json")
    return p.parse_args()


def norm(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text):
    return set(norm(text).split())


def sim(a, b):
    a_norm = norm(a)
    b_norm = norm(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm in b_norm or b_norm in a_norm:
        return 1.0
    a_tok = token_set(a_norm)
    b_tok = token_set(b_norm)
    jaccard = len(a_tok & b_tok) / max(len(a_tok | b_tok), 1)
    return max(jaccard, SequenceMatcher(None, a_norm, b_norm).ratio())


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def edge_text(edge):
    sources = " ".join(edge.get("source_sentences", []))
    return " ".join([edge["head"], edge["relation"], edge["tail"], sources])


def build_adjacency(edges):
    adj = defaultdict(list)
    for edge in edges:
        adj[norm(edge["head"])].append(edge)
        adj[norm(edge["tail"])].append(edge)
    return adj


def candidate_start_nodes(nodes, head, threshold):
    scored = [(node, sim(head, node)) for node in nodes]
    return [node for node, score in scored if score >= threshold]


def path_candidates(adj, starts, max_hops=2):
    paths = []
    frontier = [(start, [], {start}) for start in starts]
    for _ in range(max_hops):
        next_frontier = []
        for node, path, visited in frontier:
            for edge in adj.get(node, []):
                edge_key = (edge["head"], edge["relation"], edge["tail"])
                new_path = path + [edge]
                paths.append(new_path)
                for nxt in (norm(edge["head"]), norm(edge["tail"])):
                    if nxt not in visited:
                        next_frontier.append((nxt, new_path, visited | {nxt}))
        frontier = next_frontier
    return paths


def score_path(path, gold):
    text = " ".join(edge_text(edge) for edge in path)
    tail_score = sim(gold["tail_text"], text)
    rel_score = max((sim(gold["relation"], edge["relation"]) for edge in path), default=0.0)
    source_score = max(
        (sim(gold["tail_text"], source)
         for edge in path for source in edge.get("source_sentences", [])),
        default=0.0,
    )
    return 0.55 * tail_score + 0.30 * rel_score + 0.15 * source_score


def main():
    args = parse_args()
    kg = json.load(open(args.kg))
    records = load_jsonl(args.gold_jsonl)
    edges = kg.get("edges", [])
    nodes = [node.get("id", "") for node in kg.get("nodes", [])]
    adj = build_adjacency(edges)

    total = 0
    head_hits = 0
    tail_hits = 0
    relation_hits = 0
    source_hits = 0
    examples = []
    by_relation = defaultdict(lambda: {
        "n": 0,
        "head_hits": 0,
        "tail_hits": 0,
        "relation_hits": 0,
        "source_hits": 0,
    })

    for rec in records:
        for gold in rec.get("gold_triples", []):
            total += 1
            starts = candidate_start_nodes(nodes, gold["head_text"], args.similarity_threshold)
            if starts:
                head_hits += 1
            paths = path_candidates(adj, starts, max_hops=2) if starts else []
            ranked = sorted(paths, key=lambda path: score_path(path, gold), reverse=True)[:args.top_k]

            tail_hit = any(
                sim(gold["tail_text"], edge["head"]) >= args.similarity_threshold
                or sim(gold["tail_text"], edge["tail"]) >= args.similarity_threshold
                for path in ranked for edge in path
            )
            rel_hit = any(
                edge["relation"] == gold["relation"]
                for path in ranked for edge in path
            )
            src_hit = any(
                sim(gold["tail_text"], source) >= args.similarity_threshold
                for path in ranked for edge in path
                for source in edge.get("source_sentences", [])
            )

            tail_hits += int(tail_hit)
            relation_hits += int(rel_hit)
            source_hits += int(src_hit)
            rel_stats = by_relation[gold["relation"]]
            rel_stats["n"] += 1
            rel_stats["head_hits"] += int(bool(starts))
            rel_stats["tail_hits"] += int(tail_hit)
            rel_stats["relation_hits"] += int(rel_hit)
            rel_stats["source_hits"] += int(src_hit)

            if len(examples) < 12:
                examples.append({
                    "gold": gold,
                    "head_matched": bool(starts),
                    "tail_hit_topk": tail_hit,
                    "relation_hit_topk": rel_hit,
                    "source_tail_hit_topk": src_hit,
                    "top_path": [
                        {
                            "head": edge["head"],
                            "relation": edge["relation"],
                            "tail": edge["tail"],
                        }
                        for edge in (ranked[0] if ranked else [])
                    ],
                })

    summary = {
        "n_gold_triples": total,
        "top_k": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "head_node_recall": head_hits / max(total, 1),
        "tail_reachable_topk": tail_hits / max(total, 1),
        "relation_present_topk": relation_hits / max(total, 1),
        "source_tail_present_topk": source_hits / max(total, 1),
        "by_relation": {
            rel: {
                "n": stats["n"],
                "head_node_recall": stats["head_hits"] / max(stats["n"], 1),
                "tail_reachable_topk": stats["tail_hits"] / max(stats["n"], 1),
                "relation_present_topk": stats["relation_hits"] / max(stats["n"], 1),
                "source_tail_present_topk": stats["source_hits"] / max(stats["n"], 1),
            }
            for rel, stats in sorted(by_relation.items())
        },
        "examples": examples,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("=== Evidence Path Diagnostic ===")
    print(f"  Gold triples: {total}")
    print(f"  Head node recall:       {summary['head_node_recall']:.3f}")
    print(f"  Tail reachable top-{args.top_k}: {summary['tail_reachable_topk']:.3f}")
    print(f"  Relation present top-{args.top_k}: {summary['relation_present_topk']:.3f}")
    print(f"  Source tail top-{args.top_k}:     {summary['source_tail_present_topk']:.3f}")
    print("  By relation:")
    for rel, stats in summary["by_relation"].items():
        print(
            f"    {rel:14s} n={stats['n']:3d} "
            f"head={stats['head_node_recall']:.3f} "
            f"tail={stats['tail_reachable_topk']:.3f} "
            f"rel={stats['relation_present_topk']:.3f}"
        )
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
