"""Frozen identifiers and vocabularies from the approved Phase B protocol."""

PROTOCOL_ID = "B04-PATH-A-1.3"
WORKFLOW_ID = "PATH-A-WORKFLOW-1.3"
MATCHER_ID = "CODE-STRICT-1"
MATRIX_REVISION = "PATH-A-MATRIX-1.0"

ENTITY_TYPES = ("Object", "Property", "Quality", "Value")
RELATION_TYPES = (
    "selection",
    "necessity",
    "part-of",
    "not-part-of",
    "equal",
    "greater",
    "greater-equal",
    "less",
    "less-equal",
)
TRAINING_SEEDS = tuple(range(42, 50))
CONDITION_IDS = (
    "VER-RAW",
    "VER-CONFIDENCE",
    "VER-SIMPLE",
    "VER-CORRECTIVE",
)

RUN_DIRECTORIES = (
    "manifests",
    "inputs",
    "data-prepared",
    "checkpoints",
    "predictions/dev",
    "predictions/test",
    "verifier/simple",
    "verifier/corrective",
    "outcomes",
    "metrics",
    "publication",
    "audit/journal",
    "logs",
)
