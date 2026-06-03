# ASOC AGV Edge-Cost Benchmark

Benchmark code and processed data for AGV edge-cost prediction using graph and non-graph estimators.

## Project title

Cross-session benchmarking of graph and non-graph estimators for AGV edge-cost prediction under sparse real telemetry

## Description

This repository contains the Python code, processed benchmark dataset, and generated result tables used for an estimator-level machine-learning study on AGV edge-cost prediction. The study compares graph and non-graph estimators for predicting traversal time and traversal energy from sparse real AGV telemetry under a fixed cross-session evaluation protocol.

The focus of this repository is the estimator layer. It is not a complete AGV route-planning, dispatching, OPC UA, traffic-management, or deployment framework.

## Main script

run_asoc.py

## Models included

- GGNN
- GCN
- GAT
- GraphSAGE
- Ridge Regression
- SVR-RBF
- KernelRidge-RBF
- MLP

## Required input files

The script expects the following files:

- outputs/edge_samples_combined.csv
- Edge_Distances3_.csv

The file outputs/edge_samples_combined.csv is the final edge-level benchmark dataset used for training, validation, and testing.

## Expected columns in edge_samples_combined.csv

- session
- u
- v
- target_time
- target_energy
- edge_distance
- mean_speed
- slowdown_idx
- std_speed
- turn_intensity
- mean_power_W
- std_power_W
- peak_power_W
- mean_current
- std_current
- wheel_diff_mean
- stop_ratio
- scanner_ratio
- obs_t_start

## Installation

Install the required Python libraries:

pip install -r requirements.txt

## How to run

Run the main benchmark script:

python run_asoc.py

Generated results are saved in the asoc/ directory.

## Reproducibility

The benchmark uses fixed random seeds to support reproducible evaluation.

Global seed:

42

Graph-model ensemble seeds:

11, 21, 31, 41, 51

Ablation seeds:

11, 21, 31, 41, 51

All models use the same cross-session evaluation protocol:

- Session 1 is used for training and validation.
- Session 2 is used as the held-out test session.
- Within Session 1, 80% of the samples are used for training and 20% for validation.
- Input features and continuous targets are standardized using training-set statistics only.
- Predictions are inverse-transformed before metric computation.

## Main result files

The important generated result files are:

- asoc/table_main_cross_session.csv
- asoc/table_seen_unseen.csv
- asoc/table_behavior_error_turn_stop_slowdown.csv
- asoc/table_support_buckets.csv
- asoc/table_feature_graph_ablation.csv
- asoc/table_seedwise_graph_model_mean_std.csv
- asoc/table_wilcoxon_significance_tests.csv
- asoc/table_uncertainty_corr_all_models.csv
- asoc/detailed_predictions.csv
- asoc/run_config.json

These files support the manuscript results, including:

- main cross-session comparison with 95% confidence intervals,
- seen/unseen directed-edge error analysis,
- behavior-aware error analysis for turning-heavy, stop-heavy, and slowdown-heavy traversals,
- support-bucket analysis,
- feature and graph-context ablation,
- seed-wise stability analysis,
- Wilcoxon signed-rank significance tests,
- uncertainty-error correlation analysis,
- detailed per-sample predictions and errors.

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

## Research scope

This repository supports an estimator-level benchmark paper. The benchmark studies model behavior under sparse support, cross-session generalization, feature and graph-context ablation, statistical robustness, and uncertainty-error correlation.

The results should be interpreted within the investigated sparse cross-session AGV telemetry setting, not as a universal ranking of graph neural network architectures.
