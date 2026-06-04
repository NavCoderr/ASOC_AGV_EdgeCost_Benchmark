# AGV Edge-Cost Benchmark

Benchmark code and processed data for AGV edge-cost prediction using graph and non-graph estimators.

## Project title

Cross-session benchmarking of graph and non-graph estimators for AGV edge-cost prediction under sparse real telemetry

## Description

This repository contains the Python code, processed benchmark dataset, preprocessing workflow, and generated result tables used for an estimator-level machine-learning study on AGV edge-cost prediction. The study compares graph and non-graph estimators for predicting traversal time and traversal energy from sparse real AGV telemetry under a fixed cross-session evaluation protocol.

The focus of this repository is the estimator layer. It is not a complete AGV route-planning, dispatching, OPC UA, traffic-management, or deployment framework.

## Main scripts

The main benchmark script is:

- `run_asoc.py`

The preprocessing and traversal-extraction script is:

- `preprocess.py`

## Models included

The benchmark includes the following graph and non-graph estimators:

- GGNN
- GCN
- GAT
- GraphSAGE
- Ridge Regression
- SVR-RBF
- KernelRidge-RBF
- MLP

## Required input files

The benchmark expects the following files:

- `outputs/edge_samples_combined.csv`
- `Edge_Distances3_.csv`

The file `outputs/edge_samples_combined.csv` is the final edge-level benchmark dataset used for training, validation, and testing.

For reproducing the preprocessing workflow from raw telemetry, the following files are used:

- `naveen12features_title.csv`
- `Node_F3.csv`
- `Edge_Distances3_.csv`
- `preprocess.py`

## Expected columns in `edge_samples_combined.csv`

The final edge-level benchmark dataset contains the following columns:

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

The supervised prediction targets are:

- `time_s`
- `energy_J`

## Dataset summary

The validated processed dataset contains:

- 339 directed traversal samples
- 251 samples from Session 1
- 88 samples from Session 2
- 88 observed unique directed edges

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to run preprocessing

To reproduce the edge-level benchmark dataset from the raw telemetry and graph files, run:

```bash
python preprocess.py
```

This script generates the final processed dataset:

- `outputs/edge_samples_combined.csv`

It also generates intermediate inspection files, such as:

- `outputs/check_raw_sample.csv`
- `outputs/check_1hz_session1.csv`
- `outputs/check_1hz_session2.csv`
- `outputs/check_snapped_s1.csv`
- `outputs/check_snapped_s2.csv`
- `outputs/edge_samples_s1.csv`
- `outputs/edge_samples_s2.csv`

## How to run the benchmark

After the processed edge-level dataset is available, run the main benchmark script:

```bash
python run_asoc.py
```

Generated results are saved in the `asoc/` directory.

## Reproducibility

The benchmark uses fixed random seeds to support reproducible evaluation.

Global seed:

```text
42
```

Graph-model ensemble seeds:

```text
11, 21, 31, 41, 51
```

Ablation seeds:

```text
11, 21, 31, 41, 51
```

All models use the same cross-session evaluation protocol:

- Session 1 is used for training and validation.
- Session 2 is used as the held-out test session.
- Within Session 1, 80% of the samples are used for training and 20% are used for validation.
- Input features and continuous targets are standardized using training-set statistics only.
- Predictions are inverse-transformed before metric computation.

## Main result files

The important generated result files include:

- `asoc/table_main_cross_session.csv`
- `asoc/table_graph_family_cross_session.csv`
- `asoc/table_non_graph_cross_session.csv`
- `asoc/table_seen_unseen.csv`
- `asoc/table_behavior_error_turn_stop_slowdown.csv`
- `asoc/table_support_buckets.csv`
- `asoc/table_feature_graph_ablation.csv`
- `asoc/table_seedwise_graph_model_mean_std.csv`
- `asoc/table_wilcoxon_significance_tests.csv`
- `asoc/table_uncertainty_corr_all_models.csv`
- `asoc/table_selective_prediction_all_models.csv`
- `asoc/table_model_implementation.csv`
- `asoc/detailed_predictions.csv`
- `asoc/run_config.json`
- `asoc/sanity_report.json`

These files support the manuscript results, including:

- main cross-session model comparison,
- graph-family comparison,
- non-graph baseline comparison,
- seen/unseen directed-edge error analysis,
- behavior-aware error analysis for turning-heavy, stop-heavy, and slowdown-heavy traversals,
- support-bucket analysis,
- feature and graph-context ablation,
- seed-wise stability analysis,
- Wilcoxon signed-rank significance tests,
- uncertainty-error correlation analysis,
- selective prediction analysis,
- detailed per-sample predictions and errors.

## Preprocessing documentation

The preprocessing and traversal-extraction workflow is described in:

- `PREPROCESSING.md`

This file explains how the raw AGV telemetry is transformed into the final edge-level dataset used by `run_asoc.py`.

The workflow includes:

1. raw telemetry inspection,
2. temporal normalization and cleaning,
3. graph loading and graph alignment,
4. directed traversal segmentation,
5. traversal feature generation,
6. final edge-level dataset construction,
7. fixed Session 1 / Session 2 split.

## Research scope

This repository supports an estimator-level benchmark paper. The benchmark studies model behavior under sparse support, cross-session generalization, feature and graph-context ablation, statistical robustness, and uncertainty-error correlation.

The results should be interpreted within the investigated sparse cross-session AGV telemetry setting, not as a universal ranking of graph neural network architectures.
