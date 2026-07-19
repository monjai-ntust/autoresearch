# Historical secondary implementations

This directory preserves noncanonical scripts that produced or diagnosed
secondary evidence reported during the project. They are retained for source
lineage and historical reproduction; none is a paper-facing Phase B command.

Run a retained entry point from the repository root as a module, for example:

```bash
uv run --frozen python -B -m provenance.inference_kg --help
```

Canonical primary-data generation uses `phase_b.py`. Historical scripts retain
their original defaults and data assumptions, may require inputs that are not
bundled, and must write new diagnostic outputs beneath a separately identified
`output/<run-id>/` tree when they are deliberately exercised. See
`docs/historical-transition-map.md` for each family’s limitations and canonical
disposition.
