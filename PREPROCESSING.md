
## Preprocessing documentation

The preprocessing and traversal-extraction workflow is described in:

PREPROCESSING.md

This file explains how the raw AGV telemetry is transformed into the final edge-level dataset used by run_asoc.py.

The workflow includes:

1. raw telemetry inspection,
2. temporal normalization and cleaning,
3. graph alignment,
4. directed traversal segmentation,
5. traversal feature generation,
6. final edge-level dataset construction,
7. fixed Session 1 / Session 2 split.
