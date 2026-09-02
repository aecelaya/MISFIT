Medical Imaging Semantic Foundation Toolkit
===

## About

The Medical Imaging Semantic Foundation Toolkit (MISFIT) is a simple, scalable,
and end-to-end framework for pretraining 3D medical imaging foundation models
using masked autoencoders (MAE). MISFIT allows researchers to pretrain,
evaluate, and deploy large-scale encoder models on unlabeled NIfTI datasets, and
to extract rich semantic embeddings for downstream tasks such as classification,
retrieval, and anomaly detection.

MISFIT is developed in collaboration with Rice University and The University of
Texas MD Anderson Cancer Center.

## Overview

MISFIT is organized around two main workflows:

**Pretraining** — Train a SwinUNETR-based masked autoencoder on a large
collection of unlabeled 3D NIfTI volumes. The encoder learns rich semantic
representations of anatomy without any labels.

**Embedding** — Use a pretrained encoder to extract per-volume feature vectors.
These embeddings can be used directly for zero-shot similarity search or
fine-tuned for downstream classification tasks.

## What's New

- March 2026 — Initial alpha release of MISFIT with full MAE pretraining,
  evaluation, inspection, and embedding pipelines.
