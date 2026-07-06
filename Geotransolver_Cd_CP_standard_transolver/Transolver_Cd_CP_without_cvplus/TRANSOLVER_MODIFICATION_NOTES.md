# Ordinary Transolver Backbone Modification Notes

This package uses the PhysicsNeMo `Transolver` backbone only. The local `geotransolver/` package has been removed.

## Main code change

`train.py` now uses:

```python
from physicsnemo.models.transolver import Transolver

self.backbone = Transolver(
    functional_dim=config.FUNCTIONAL_DIM,
    embedding_dim=config.GEOMETRY_DIM,
    out_dim=config.BACKBONE_OUT_DIM,
    structured_shape=None,
    unified_pos=False,
    n_hidden=config.N_HIDDEN,
    n_layers=config.N_LAYERS,
    n_head=config.N_HEAD,
    dropout=config.DROPOUT,
    slice_num=config.SLICE_NUM,
    use_te=False,
    plus=True,
)

output = self.backbone(point_features, embedding=geometry)
```

For this Cd prediction project, the sampled surface coordinates are the only local point features, so the ordinary Transolver receives the same coordinates as both:

- `fx = point_features`, the functional input;
- `embedding = geometry`, the unstructured positional embedding.

Surface point coordinates are used as both functional input and positional embedding. The downstream structured pooling and quantile regression head are unchanged.

## 8-sample overfit smoke test

Use this to verify the Transolver interface and training loop on a fixed 8-sample subset:

```bash
# Linux / Alvis
bash scripts/overfit_single.sh

# Windows PowerShell
.\scripts\overfit_single.ps1

# Or directly
OVERFIT_MODE=1 OVERFIT_SUBSET_SIZE=8 python train.py
```

Key env vars: `OVERFIT_SUBSET_SIZE` (default 8), `LEARNING_RATE` (default 1e-4), `TRAIN_CSV`, `STL_ROOT_DIR`, `NUM_EPOCHS`.
Outputs go to `./checkpoints/overfit_8` and `./logs/overfit_8` by default.
Watch logs for `[OVERFIT]` lines: `|q50-target|` should decrease if the interface is healthy.

For a single fixed sample, use `OVERFIT_SINGLE=1` (optional `OVERFIT_SAMPLE_INDEX`).
Defaults: `LEARNING_RATE=1e-4`, `BATCH_SIZE=8`.

## Important

Checkpoints trained with the removed GeoTransolver backbone are not compatible. Retrain before testing.
