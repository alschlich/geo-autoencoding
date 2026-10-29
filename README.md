# Geometry Encodings for AI-enhanced Computer Aided Design

Implementation of the paper **"Geometry Encodings for AI-enhanced Computer Aided Design"** by Alexander Schlicher and Benjamin Schleich (CIRP Design Conference 2026).

This repository contains a **rotation-equivariant point cloud graph autoencoder** for geometries commonly encountered in mechanical CAD. It encodes point clouds sampled from surfaces together with their surface normals and geodesic neighborhood structure into a set of localized, rotation-equivariant feature vectors. The encodings are learned unsupervised via a surface upsampling (autoencoding) task and are intended as input representations for downstream tasks such as CAD sequence generation, canonicalization, symmetry detection, and sub-shape matching.

**Paper**: [*Geometry Encodings for AI-enhanced Computer Aided Design*](https://www.sciencedirect.com/science/article/pii/S2212827126007973), Procedia CIRP 142 (2026) 61–66

## Key Ideas

- **Graph-based input representation**: Points sampled from a surface (with normals) form the nodes; edges encode *geodesic* k-nearest-neighbor relations on the underlying geometry.
- **Rotation equivariance**: Built on [Vector Neurons](https://arxiv.org/abs/2104.12229) and a modified [DGCNN](https://arxiv.org/abs/1801.07829) architecture, so features rotate with the input.
- **Unsupervised training**: The encoder is trained through a surface reconstruction (upsampling) task. The decoder is only needed for training; the intermediate feature spaces are the actual product.
- **Localized encodings**: Features are computed per anchor point rather than compressed into a single fixed-size vector, keeping encodings tied to sampling resolution and enabling sub-shape similarity search in feature space.

## Repository Structure

```
.
├── LICENSE
├── README.md
├── pyproject.toml                  # Project metadata and dependencies (PDM)
├── pdm.lock                        # Locked dependency versions
└── src/
    └── point_cloud_ae/
        ├── main.py                 # Training entry point (hyperparameters, training loop)
        ├── networks.py             # Networks class: train/test steps, inference, checkpointing
        ├── data_loaders.py         # CADGraphDataset: pickled point cloud graphs + pytorch3d meshes
        ├── losses.py               # Surface (point-to-mesh), pairwise spreading, equivariance losses
        ├── geom_utils.py           # Random rotations, graph transforms, batched KNN, sampling
        ├── output_utils.py         # Checkpoint saving, output directory handling
        ├── vis_utils.py            # Matplotlib 3D visualization helpers
        ├── visualizations.py       # Script to generate the result figures from the paper
        └── models/
            ├── encoder.py          # VNGEncoder: geodesic KNN features + VN-DGCNN backbone
            ├── decoder.py          # VNGDecoder: VN-MLP that upsamples per-point features
            ├── vn_layers.py        # Vector Neuron layers (rotation-equivariant building blocks)
            └── nvn_layers.py       # Additional/permutator VN-style layers
```

## Requirements

- Python 3.10
- [PDM](https://pdm-project.org/) for dependency management
- A CUDA-capable GPU is recommended for training

Core dependencies (see `pyproject.toml` for the full list): `torch`, `torch-geometric`, `pytorch3d` (installed from the GitHub `stable` branch), `numpy`, `scipy`, `matplotlib`.

Install with PDM:

```bash
pdm install
```

## Data Preparation

The network is trained on datasets that combine:

- **[DeepCAD](https://github.com/ChrisWu1997/DeepCAD)** — a subset of the [ABC dataset](https://deep-geometry.github.io/abc-dataset/) containing CAD parts created via sketch-and-extrude operations (used for training/testing).
- **[ModelNet40](https://modelnet.cs.princeton.edu/)** — used for validation on unseen, general 3D shapes.

The datasets are not included in this repository. Preprocessing samples points and surface normals from each shape, computes geodesic k-nearest-neighbor edges on the mesh, filters shapes whose meshes do not allow reliable geodesic computation, and stores each sample as a dictionary with:

- `graph`: a `torch_geometric.data.Data` object with node positions, normals, and geodesic edge indices,
- `mesh`: vertex and face arrays, used to build `pytorch3d` meshes for the surface reconstruction loss.

The resulting dataset is pickled to a file (by default `data/dataset.pkl`, configurable in `main.py`) and loaded by `CADGraphDataset` in `src/point_cloud_ae/data_loaders.py`. All shapes are expected to be centered and scaled to a sphere of radius 0.8.

## Training

Hyperparameters are set at the top of `src/point_cloud_ae/main.py`:

| Parameter | Default |
|---|---|
| Input node count | 2048 |
| Encoding size | 1024 |
| Epochs | 250 |
| Batch size | 14 (CUDA) / 3 (CPU) |
| Train/test split | 90% / 10% |
| Optimizer | Adam, learning rate 0.001 |
| Data path | `data/dataset.pkl` |

Run training from the repository root:

```bash
python src/point_cloud_ae/main.py
```

Random rotations are applied on-the-fly during training. The best model (lowest test loss) is saved to `output/best_model.pth`.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{schlicher2026geometry,
  title     = {Geometry Encodings for AI-enhanced Computer Aided Design},
  author    = {Schlicher, Alexander and Schleich, Benjamin},
  journal   = {Procedia CIRP},
  volume    = {142},
  pages     = {61--66},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.procir.2026.05.224},
  note      = {36th CIRP Design Conference (CIRP Design 2026)}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Acknowledgements

Parts of `src/point_cloud_ae/geom_utils.py` are adapted from
[`geom_torch.py`](https://github.com/canonical-capsules/canonical-capsules/blob/main/geom_torch.py)
of [Canonical Capsules](https://github.com/canonical-capsules/canonical-capsules)
(Sun et al., NeurIPS 2021), licensed under the Apache License 2.0.
