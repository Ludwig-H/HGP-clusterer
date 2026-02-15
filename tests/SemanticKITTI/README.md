# HGP-clusterer : 4D Panoptic Segmentation on SemanticKITTI

This directory contains resources for testing **4D Panoptic Segmentation** (LiDAR Segmentation + Tracking) using the `HGP-clusterer` library on the SemanticKITTI dataset.

## Notebook

The main file is **`HGP_SemanticKITTI_4D_Panoptic.ipynb`**.

### Features

1.  **Universal Setup**:
    *   Works on Google Colab (and local Linux machines).
    *   Automatically installs `geogram` (fast backend) or `cgal` (exact backend).
    *   Installs `HGP-clusterer` and the `semantic-kitti-api`.

2.  **Data Loading**:
    *   Connects to a provided Google Drive folder to download specific sequences (e.g., Sequence 08).
    *   Implements a custom `SemanticKITTILoader` to parse `.bin` (LiDAR), `.label` (Ground Truth), `poses.txt`, and `calib.txt`.
    *   Transforms point clouds into the global World coordinate system.

3.  **4D Point Cloud Construction**:
    *   Accumulates multiple scans over a sliding window (e.g., 10 frames).
    *   Adds a time dimension $t$ (scaled by a factor $\alpha$) to create 4D points $(x, y, z, \alpha t)$.

4.  **Clustering & Tracking**:
    *   Uses **HGP-clusterer** (`HypergraphPercol`) to cluster points in 4D space.
    *   Clustering in 4D inherently solves the **association problem** (Tracking) by grouping spatio-temporally connected points into single instances.
    *   Includes a **Splitting Function** mechanism (Oracle-based in the demo) to refine clusters based on geometric priors or semantic purity.

5.  **Evaluation**:
    *   Approximates the **LSTQ (LiDAR Segmentation and Tracking Quality)** metric, focusing on $S_{assoc}$ (Association Quality).

6.  **Visualization**:
    *   Interactive 3D/4D visualization using `plotly`.

## How to use

1.  Open the notebook in Google Colab.
2.  Select your runtime (GPU is recommended for display, but CPU is fine for Geogram backend).
3.  Run the **Setup** cells to install dependencies.
4.  In the **Data Loading** section, choose `SEQUENCE_TO_TEST` (default: 8) and run the cell. This will download only the necessary data if configured, or you can point it to your local SemanticKITTI dataset.
5.  Run the **Clustering** and **Evaluation** cells.
6.  Explore the results in the **Visualization** section.

## Requirements

*   Python 3.8+
*   `numpy`, `scipy`, `scikit-learn`
*   `plotly` (for visualization)
*   `hgp_clusterer` (installed via the notebook)
*   `cyminiball` (dependency for HGP)
