"""Intrinsic graph metrics with explicit unavailable disposition."""

from __future__ import annotations

from graph_rag_eval.capabilities import Capability, CapabilityUnavailable
from graph_rag_eval.graphs.matching import GraphMatch, match_graphs
from graph_rag_eval.graphs.snapshots import GraphSnapshot


def evaluate_intrinsic(
    predicted: GraphSnapshot,
    gold: GraphSnapshot | None,
    *,
    capabilities: tuple[str, ...],
    relaxed: bool = False,
) -> GraphMatch | CapabilityUnavailable:
    unavailable = CapabilityUnavailable.for_required(
        "intrinsic_graph_metrics",
        (
            Capability.GOLD_ENTITIES,
            Capability.GOLD_RELATIONS,
            Capability.GOLD_TRIPLES,
        ),
        capabilities,
    )
    if unavailable is not None:
        return unavailable
    if gold is None:
        return CapabilityUnavailable(
            operation="intrinsic_graph_metrics",
            missing=("gold_graph_snapshot",),
            reason="capabilities were declared but no gold graph was materialized",
        )
    return match_graphs(predicted, gold, relaxed=relaxed)
