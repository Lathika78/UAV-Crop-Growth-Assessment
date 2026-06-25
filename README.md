# UAV Crop Growth Assessment Using HSV-Based Clustering

## Overview

This project presents a comparative analysis of K-Means, Mean-Shift, and Gaussian Mixture Model (GMM) clustering algorithms for crop growth assessment using UAV imagery. The methodology uses HSV color space to analyze vegetation characteristics and estimate crop growth conditions based on vegetation density.

## Dataset

The repository contains a sample UAV image used for vegetation analysis and clustering experiments.

## Algorithms Implemented

* K-Means Clustering
* Mean-Shift Clustering
* Gaussian Mixture Model (GMM)

## Methodology

1. UAV image acquisition
2. RGB to HSV conversion
3. Pixel-wise color analysis
4. Clustering of vegetation pixels
5. Growth stage classification
6. Vegetation density estimation
7. Performance evaluation

## Performance Metrics

The algorithms were evaluated using:

* Runtime
* Number of Clusters
* Silhouette Score
* Davies-Bouldin Index

## Results Summary

| Algorithm  | Runtime (s) | Clusters | Silhouette Score | Davies-Bouldin Index |
| ---------- | ----------: | -------: | ---------------: | -------------------: |
| K-Means    |        4.84 |        5 |           0.3464 |               0.9269 |
| Mean-Shift |        7.77 |        3 |           0.3697 |               1.0155 |
| GMM        |       25.02 |        5 |           0.1894 |               1.3421 |

The results indicate that K-Means provides the best balance between computational efficiency and clustering quality for HSV-based crop growth assessment.

## Repository Structure

* `dataset/` – Sample UAV image
* `codes/` – Python implementation of clustering algorithms
* `outputs/` – Generated clustering results and performance metrics

## Future Work

* Evaluation on larger UAV datasets
* Integration with vegetation indices such as NDVI
* Development of automated crop monitoring systems

## Authors

INTERNSHIP PROJECT TEAM
