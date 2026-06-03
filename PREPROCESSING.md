# Preprocessing and Traversal-Extraction Workflow

This document explains how the raw AGV telemetry is transformed into the final edge-level benchmark dataset used by `run_asoc.py`.

## 1. Raw telemetry input

The raw telemetry file contains AGV operational records collected from the CNT laboratory AGV platform. The telemetry includes timestamped movement and operational signals such as speed-related values, power/current-related values, scanner or safety-related indicators, and graph-related movement information.

The telemetry source used for preparing the benchmark dataset is:

- `naveen12features_title.csv`

## 2. Directed graph information

The AGV working environment is represented as a directed graph. Nodes correspond to operational locations in the shop-floor layout, and directed edges correspond to feasible AGV movements between these locations.

The graph-distance file used by the benchmark is:

- `Edge_Distances3_.csv`

## 3. Temporal normalization and cleaning

The telemetry is first converted into a stable time-indexed representation. During this stage, irregular records, duplicate timestamps, missing values, and implausible movement values are handled before edge-level sample construction.

This step is required because traversal time and traversal energy must be computed from temporally consistent AGV movement records.

## 4. Graph alignment

After temporal normalization and cleaning, the AGV trajectory is aligned with the directed shop-floor graph. Each valid movement record is associated with graph-node or edge-related information so that continuous AGV telemetry can be converted into graph-based traversal samples.

## 5. Directed traversal segmentation

A directed traversal sample is created when the AGV moves from a source node `u` to a destination node `v`. Each valid transition is treated as one directed edge-level sample.

For each traversal sample, the pipeline stores:

- `session`
- `u`
- `v`
- traversal-time target
- traversal-energy target
- telemetry-derived traversal descriptors

## 6. Feature generation

For every directed traversal, the following features are generated and stored in the final benchmark dataset:

- `edge_distance`
- `mean_speed`
- `slowdown_idx`
- `std_speed`
- `turn_intensity`
- `mean_power_W`
- `std_power_W`
- `peak_power_W`
- `mean_current`
- `std_current`
- `wheel_diff_mean`
- `stop_ratio`
- `scanner_ratio`
- `obs_t_start`

The supervised prediction targets are:

- `target_time`
- `target_energy`

## 7. Final edge-level benchmark dataset

The final processed dataset used by the benchmark script is:

- `outputs/edge_samples_combined.csv`

This file is the direct input to `run_asoc.py`.

## 8. Cross-session split

The benchmark follows a fixed cross-session protocol:

- Session 1 is used for training and validation.
- Session 2 is used as the held-out test session.
- Within Session 1, 80% of the samples are used for training and 20% are used for validation.

This protocol evaluates estimator behavior under sparse support and cross-session generalization.

## 9. Running the benchmark after preprocessing

After the final edge-level dataset and graph-distance file are available, the benchmark can be reproduced by running:

```bash
pip install -r requirements.txt
python run_asoc.py
