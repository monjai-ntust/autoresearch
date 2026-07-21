"""Frozen identifiers and vocabularies from the approved Phase B protocol."""

PROTOCOL_ID = "B04-PATH-A-1.3"
WORKFLOW_ID = "PATH-A-WORKFLOW-1.3"
MATCHER_ID = "CODE-STRICT-1"
MATRIX_REVISION = "PATH-A-MATRIX-1.0"
SECTION5_EVIDENCE_REVISION = "SECTION5-EVIDENCE-2.0"

SECTION5_CLAIM_IDS = (
    "S5-1-ENCODER-RECIPE",
    "S5-1-AUGMENTATION",
    "S5-1-CROSS-DATASET",
    "S5-2-GRAPH-RAG",
    "S5-3-VERIFIER",
    "S5-4-INDUSTRY-RESOURCE-UI",
    "S5-5-ZH-HANT",
)

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

CODE_ACCORD_ARCHIVE = {
    "dataset_id": "CODE-ACCORD-v1.0.0",
    "zenodo_record": "10210022",
    "name": "CODE-ACCORD-v1.0.0.zip",
    "url": (
        "https://zenodo.org/api/records/10210022/files/"
        "Accord-Project/CODE-ACCORD-v1.0.0.zip/content"
    ),
    "bytes": 101265616,
    "md5": "57e2efa465f41e2f582db62810fb50f5",
    "license": "CC-BY-4.0",
}

CODE_ACCORD_ANNOTATION_FILES = {
    "annotated_data/entities/all.csv": {
        "bytes": 479772,
        "sha256": "0c041af90ca22b35af46fc22065935ffa94fc6893f7bf31366d12d65c89d796a",
    },
    "annotated_data/entities/train.csv": {
        "bytes": 389041,
        "sha256": "c13ad02ab72f0f3a3ddca588c02bcd5d7db1622ae81351d967431623963e4fcd",
    },
    "annotated_data/entities/test.csv": {
        "bytes": 90767,
        "sha256": "e84246648e27fb24fbf3409130d8a9cb79793f0c14ec4390129605436ceeeac3",
    },
    "annotated_data/relations/all.csv": {
        "bytes": 2049753,
        "sha256": "a321659f3fc8028a48d408f73d885e7a5f4bc765a0e3d42e7878aaf7f8803bef",
    },
    "annotated_data/relations/train.csv": {
        "bytes": 1651655,
        "sha256": "31621624b129a412f2cf7bb0dff72938a8c97ace043629aaafd42864a5d1ee1c",
    },
    "annotated_data/relations/test.csv": {
        "bytes": 398156,
        "sha256": "13cee7f40202ac08b0d5a8720a87d6d80108992d887050f99569c493563b29fa",
    },
}

CODE_ACCORD_COUNTS = {
    "sentences": 862,
    "entity_train_sentences": 689,
    "entity_test_sentences": 173,
    "entity_spans": 4297,
    "relation_rows": 4329,
    "positive_relation_rows": 3329,
    "none_relation_rows": 1000,
    "relation_covered_sentences": 857,
    "entity_only_sentences": 5,
}

CODE_ACCORD_RELATION_SPLIT_AUDIT = {
    "train_rows": 3463,
    "test_rows": 866,
    "train_unique_sentence_ids": 824,
    "test_unique_sentence_ids": 507,
    "train_test_sentence_id_overlap": 473,
    "entity_test_ids_in_relation_train": 162,
    "entity_test_ids_in_relation_test": 96,
    "entity_train_ids_in_relation_test": 411,
}

CODE_ACCORD_REPAIRED_UUID = "8feaf7f2-dfd8-43ae-ab8d-791ee1ffd86c"
TREE_HASH_REVISION = "PHASE-B-TREE-1"

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
