# ASOC AGV Edge-Cost Benchmark

Benchmark code for AGV edge-cost prediction using graph and non-graph estimators.

## Project title

Cross-session benchmarking of graph and non-graph estimators for AGV edge-cost prediction under sparse real telemetry

## Description

This repository contains the Python code and processed benchmark files used for an estimator-level machine-learning study on AGV edge-cost prediction. The study compares graph and non-graph estimators for predicting traversal time and traversal energy from sparse real AGV telemetry under a cross-session evaluation protocol.

The focus of this repository is the estimator layer. It is not a complete AGV route-planning, dispatching, or deployment framework.

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

outputs/edge_samples_combined.csv  
Edge_Distances3_.csv

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

## Outputs

The script generates result tables and prediction files, including:

- main cross-session model comparison
- graph-family comparison
- non-graph baseline comparison
- feature and graph-context ablation results
- uncertainty-error correlation results
- detailed prediction files

## Research scope

This repository supports an estimator-level benchmark paper. The benchmark studies model behavior under sparse support, cross-session generalization, feature and graph-context ablation, and uncertainty-error correlation.
