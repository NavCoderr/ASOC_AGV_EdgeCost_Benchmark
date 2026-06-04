# Preprocessing and Traversal-Extraction Workflow

This document explains how the raw AGV telemetry is transformed into the final edge-level benchmark dataset used by `run_asoc.py`.

## 1. Raw telemetry input

The raw telemetry file contains AGV operational records collected from the laboratory AGV platform. The telemetry includes timestamped movement and operational signals such as position, speed, power consumption, cumulative energy consumption, current, heading, wheel-speed values, scanner status, and safety-related indicators.

The telemetry source used for preparing the benchmark dataset is:

- `naveen12features_title.csv`

## 2. Directed graph information

The AGV working environment is represented as a directed graph. Nodes correspond to operational locations in the shop-floor layout, and directed edges correspond to feasible AGV movements between these locations.

The graph and node files used by the preprocessing workflow are:

- `Node_F3.csv`
- `Edge_Distances3_.csv`

## 3. Preprocessing script

The preprocessing and traversal-extraction workflow is implemented in:

- `preprocess.py`

This script converts the raw telemetry into a cleaned 1 Hz representation, aligns the trajectory with the directed graph, extracts directed traversal samples, and creates the final edge-level benchmark dataset.

To run the preprocessing workflow, use:

```bash
python preprocess.py
```

## 4. Temporal normalization and cleaning

The telemetry is first converted into a stable 1 Hz time-indexed representation. During this stage, timestamps are parsed, duplicate records are removed, session information is assigned, energy increments are computed from the cumulative energy meter, and missing time points are filled using interpolation.

This step is required because traversal time and traversal energy must be computed from temporally consistent AGV movement records.

The preprocessing script also generates intermediate check files:

- `outputs/check_raw_sample.csv`
- `outputs/check_1hz_session1.csv`
- `outputs/check_1hz_session2.csv`
- `outputs/check_1hz_combined_move.csv`

These files are used only for inspection and verification.

## 5. Graph alignment

After temporal normalization and cleaning, the AGV trajectory is aligned with the directed shop-floor graph. Each valid AGV position is snapped to the nearest graph node within the configured snapping radius. Short missing snapping gaps are forward-filled to preserve movement continuity.

The graph-alignment stage generates:

- `outputs/check_snapped_s1.csv`
- `outputs/check_snapped_s2.csv`

These files are used to verify that the trajectory was correctly aligned with graph nodes.

## 6. Directed traversal segmentation

A directed traversal sample is created when the AGV moves from a source node `u_node_id` to a destination node `v_node_id`. Each valid transition must correspond to a feasible directed edge in the graph.

For each traversal sample, the pipeline stores:

- `session`
- `u_node_id`
- `v_node_id`
- traversal-time target
- traversal-energy target
- telemetry-derived traversal descriptors

Invalid, too short, too long, or insufficiently moving segments are filtered during traversal extraction.

## 7. Feature generation

For every valid directed traversal, the following columns are generated and stored in the final benchmark dataset:

- `session`
- `u_node_id`
- `v_node_id`
- `edge_distance`
- `time_s`
- `energy_J`
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

The supervised prediction targets used by the benchmark are:

- `time_s`
- `energy_J`

## 8. Final edge-level benchmark dataset

The final processed dataset used by the benchmark script is:

- `outputs/edge_samples_combined.csv`

This file is the direct input to `run_asoc.py`.

The validated dataset contains:

- 339 directed traversal samples
- 251 samples from Session 1
- 88 samples from Session 2
- 88 observed unique directed edges

The preprocessing script also saves the session-level traversal files:

- `outputs/edge_samples_s1.csv`
- `outputs/edge_samples_s2.csv`

## 9. Cross-session split

The benchmark follows a fixed cross-session protocol:

- Session 1 is used for training and validation.
- Session 2 is used as the held-out test session.
- Within Session 1, 80% of the samples are used for training and 20% are used for validation.

This protocol evaluates estimator behavior under sparse support and cross-session generalization.

## 10. Running the benchmark after preprocessing

After the final edge-level dataset and graph-distance file are available, the benchmark can be reproduced by running:

```bash
pip install -r requirements.txt
python run_asoc.py
```

The generated benchmark result tables and prediction files are saved in the `asoc/` directory.
