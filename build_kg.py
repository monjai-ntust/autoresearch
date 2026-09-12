"""
KG Construction Pipeline: Entity Resolution + Triple Dedup + Schema Validation.

Takes verified triples from verify_triples_llm.py and constructs a clean KG:
1. Entity Resolution: merge mentions referring to the same entity
2. Triple Deduplication: remove redundant (h, r, t) after entity resolution
3. Schema Validation: filter triples violating ontology constraints
4. Pronoun Filtering: remove triples with pronoun entities (it, they, etc.)

Output: structured KG as JSON (nodes + edges) + statistics.

Usage:
    uv run python build_kg.py \
        --input results/kg_verified_scierc_test.jsonl \
        --output results/kg_constructed.json \
        --filter-mode verified  # use conf>=0.5 + LLM=yes
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# Pronouns and stopword entities to filter
PRONOUN_ENTITIES = {
    "it", "they", "them", "this", "that", "these", "those",
    "we", "our", "us", "he", "she", "its", "their",
    "the method", "the approach", "the system", "the model",
    "the algorithm", "the technique", "the framework",
    "the proposed method", "the proposed approach",
}

# SciERC relation schema constraints (head_type → allowed relations → tail_type)
# Loose constraints — mainly for filtering obviously wrong type combos
SCIERC_RELATIONS = {
    "USED-FOR", "FEATURE-OF", "HYPONYM-OF", "EVALUATE-FOR",
    "PART-OF", "COMPARE", "CONJUNCTION",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Verified JSONL from verify_triples_llm.py")
    p.add_argument("--output", required=True, help="Output KG JSON")
    p.add_argument("--filter-mode", default="verified",
                   choices=["all", "confidence", "llm", "verified"],
                   help="all: keep all predicted triples. "
                        "confidence: conf>=0.5. llm: LLM=yes. "
                        "verified: conf>=0.5 + LLM=yes (default).")
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--similarity-threshold", type=float, default=0.8,
                   help="String similarity threshold for entity resolution.")
    return p.parse_args()


def normalize_entity(text):
    """Normalize entity text for comparison."""
    text = text.lower().strip()
    # Remove parenthetical abbreviations: "machine translation -lrb- mt -rrb-" → "machine translation"
    text = re.sub(r'\s*-lrb-.*?-rrb-\s*', '', text)
    text = re.sub(r'\s*\(.*?\)\s*', '', text)
    # Remove leading articles
    text = re.sub(r'^(the|a|an)\s+', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def string_similarity(a, b):
    """Word-set Jaccard similarity."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_substring_match(a, b):
    """
    Check if one is a substring of the other, with guards against
    over-merging short generic words.

    Rules:
    - Both must have >= 2 words (no single-word substring matches)
    - The shorter must be >= 60% of the longer's word count
    - This prevents "method" from matching "random-projection based methods"
    """
    words_a = a.split()
    words_b = b.split()
    if len(words_a) < 2 or len(words_b) < 2:
        return False
    shorter = min(len(words_a), len(words_b))
    longer = max(len(words_a), len(words_b))
    if shorter / longer < 0.6:
        return False
    return a in b or b in a


def pluralize_match(a, b):
    """Check if a and b differ only by trailing 's/es', and both have >= 2 words."""
    # Only match multi-word phrases to avoid "model" ↔ "models" generic merging
    if len(a.split()) < 2 and len(b.split()) < 2:
        return False
    if a + 's' == b or b + 's' == a:
        return True
    if a + 'es' == b or b + 'es' == a:
        return True
    return False


def build_entity_clusters(entities, sim_threshold=0.8, use_embeddings=False,
                          embedding_model=None, embedding_threshold=0.85):
    """
    Cluster entity mentions by string similarity + optional embedding similarity.
    Returns: dict mapping each mention → canonical form (most frequent).

    Two-pass approach:
      Pass 1: exact normalized form match (always)
      Pass 2: string Jaccard + substring + pluralize (guarded)
      Pass 3 (optional): embedding cosine similarity for remaining singletons
    """
    normalized = {e: normalize_entity(e) for e in entities}

    # Pass 1: Group by normalized form (exact match after normalization)
    norm_groups = defaultdict(list)
    for original, norm in normalized.items():
        norm_groups[norm].append(original)

    # Pass 2: Merge groups with high string similarity or substring match
    group_list = list(norm_groups.items())
    merged = [False] * len(group_list)
    clusters = []

    for i in range(len(group_list)):
        if merged[i]:
            continue
        cluster = list(group_list[i][1])
        norm_i = group_list[i][0]

        for j in range(i + 1, len(group_list)):
            if merged[j]:
                continue
            norm_j = group_list[j][0]

            if (string_similarity(norm_i, norm_j) >= sim_threshold
                    or is_substring_match(norm_i, norm_j)
                    or pluralize_match(norm_i, norm_j)):
                cluster.extend(group_list[j][1])
                merged[j] = True

        clusters.append(cluster)

    # Pass 3: Embedding-based merging for remaining singletons
    if use_embeddings and embedding_model is not None:
        singleton_indices = [i for i, c in enumerate(clusters) if len(c) == 1]
        if len(singleton_indices) >= 2:
            singleton_texts = [normalize_entity(clusters[i][0]) for i in singleton_indices]

            # Encode all singletons
            import torch
            with torch.no_grad():
                encodings = embedding_model(
                    singleton_texts, padding=True, truncation=True,
                    max_length=32, return_tensors="pt",
                )
                # Use [CLS] embedding
                embeddings = encodings.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                sim_matrix = torch.mm(embeddings, embeddings.t())

            # Greedy merge high-similarity singletons
            emb_merged = [False] * len(singleton_indices)
            for i in range(len(singleton_indices)):
                if emb_merged[i]:
                    continue
                for j in range(i + 1, len(singleton_indices)):
                    if emb_merged[j]:
                        continue
                    if sim_matrix[i, j].item() >= embedding_threshold:
                        # Merge j into i's cluster
                        ci = singleton_indices[i]
                        cj = singleton_indices[j]
                        clusters[ci].extend(clusters[cj])
                        clusters[cj] = []  # mark empty
                        emb_merged[j] = True

            # Remove empty clusters
            clusters = [c for c in clusters if c]

    # Pick canonical: longest mention in the largest cluster (more informative)
    mention_to_canonical = {}
    for cluster in clusters:
        canonical = max(cluster, key=lambda x: (len(x.split()), len(x)))
        for mention in cluster:
            mention_to_canonical[mention] = canonical

    return mention_to_canonical, clusters


def filter_triple(triple, filter_mode, conf_threshold):
    """Check if triple passes the filter."""
    if filter_mode == "all":
        return True
    elif filter_mode == "confidence":
        return triple["triple_conf"] >= conf_threshold
    elif filter_mode == "llm":
        return triple.get("llm_verdict") in ("yes", "keep")
    elif filter_mode == "verified":
        return (triple["triple_conf"] >= conf_threshold
                and triple.get("llm_verdict") in ("yes", "keep"))
    return True


def is_pronoun_entity(text):
    """Check if entity is a pronoun or generic reference."""
    normalized = text.lower().strip()
    return normalized in PRONOUN_ENTITIES or len(normalized) <= 2


def main():
    args = parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f]

    print(f"=== KG Construction Pipeline ===")
    print(f"  Input: {args.input} ({len(records)} docs)")
    print(f"  Filter: {args.filter_mode} (conf>={args.conf_threshold})")

    # Step 1: Collect all triples that pass filter
    raw_triples = []
    all_entity_mentions = set()

    for rec in records:
        for t in rec["predicted_triples"]:
            if not filter_triple(t, args.filter_mode, args.conf_threshold):
                continue
            h = t["head_text"].lower().strip()
            tl = t["tail_text"].lower().strip()

            # Pronoun filtering
            if is_pronoun_entity(h) or is_pronoun_entity(tl):
                continue

            all_entity_mentions.add(h)
            all_entity_mentions.add(tl)
            raw_triples.append({
                "head": h,
                "tail": tl,
                "relation": t["relation"],
                "triple_conf": t["triple_conf"],
                "source_doc": rec["doc_id"],
                "sentence": rec["sentence"],
            })

    print(f"  After filtering: {len(raw_triples)} triples, {len(all_entity_mentions)} entity mentions")

    # Step 2: Entity Resolution
    mention_to_canonical, clusters = build_entity_clusters(
        all_entity_mentions, sim_threshold=args.similarity_threshold,
    )
    n_merged = len(all_entity_mentions) - len(clusters)
    canonical_entities = set(mention_to_canonical.values())
    print(f"  Entity resolution: {len(all_entity_mentions)} mentions → {len(canonical_entities)} entities ({n_merged} merged)")

    # Step 3: Apply resolution + deduplicate triples
    resolved_triples = set()
    triple_details = {}  # (h, r, t) → best confidence + sources

    for t in raw_triples:
        h = mention_to_canonical.get(t["head"], t["head"])
        tl = mention_to_canonical.get(t["tail"], t["tail"])
        rel = t["relation"]

        if h == tl:  # self-loop after resolution
            continue

        key = (h, rel, tl)
        if key not in triple_details or t["triple_conf"] > triple_details[key]["best_conf"]:
            triple_details[key] = {
                "best_conf": t["triple_conf"],
                "sources": [],
            }
        triple_details[key]["sources"].append(t["source_doc"])
        resolved_triples.add(key)

    print(f"  After dedup: {len(resolved_triples)} unique triples")

    # Step 4: Build KG structure
    nodes = {}
    edges = []

    for (h, rel, tl), details in triple_details.items():
        if h not in nodes:
            nodes[h] = {"id": h, "frequency": 0, "relations": []}
        if tl not in nodes:
            nodes[tl] = {"id": tl, "frequency": 0, "relations": []}
        nodes[h]["frequency"] += 1
        nodes[tl]["frequency"] += 1

        edge = {
            "head": h,
            "relation": rel,
            "tail": tl,
            "confidence": details["best_conf"],
            "n_sources": len(set(details["sources"])),
        }
        edges.append(edge)
        nodes[h]["relations"].append(rel)
        nodes[tl]["relations"].append(rel)

    # Sort edges by confidence
    edges.sort(key=lambda x: -x["confidence"])

    kg = {
        "metadata": {
            "source": args.input,
            "filter_mode": args.filter_mode,
            "conf_threshold": args.conf_threshold,
            "n_documents": len(records),
            "n_raw_triples": len(raw_triples),
            "n_entity_mentions": len(all_entity_mentions),
            "n_canonical_entities": len(canonical_entities),
            "n_merged_entities": n_merged,
            "n_unique_triples": len(resolved_triples),
        },
        "nodes": sorted(nodes.values(), key=lambda x: -x["frequency"]),
        "edges": edges,
        "entity_clusters": [
            {"canonical": max(c, key=lambda x: (len(x.split()), len(x))),
             "mentions": c, "size": len(c)}
            for c in clusters if len(c) > 1
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(kg, f, indent=2)

    # Print summary
    print(f"\n=== KG Summary ===")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Entity clusters (merged): {len([c for c in clusters if len(c) > 1])}")
    print(f"  Output: {out_path}")

    # Relation distribution
    rel_counts = defaultdict(int)
    for e in edges:
        rel_counts[e["relation"]] += 1
    print(f"\n  Relation distribution:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel:20s}: {cnt}")

    # Top entities
    print(f"\n  Top 10 entities by degree:")
    for node in kg["nodes"][:10]:
        print(f"    {node['frequency']:3d}  {node['id']}")

    # Entity resolution examples
    print(f"\n  Entity resolution examples (clusters > 1):")
    for cluster in sorted(kg["entity_clusters"], key=lambda x: -x["size"])[:10]:
        print(f"    {cluster['canonical']} ← {cluster['mentions']}")


if __name__ == "__main__":
    main()
