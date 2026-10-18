# mv-PIKN architecture and training modules

This package reorganizes the original `pikn_v11.py` script (whose header identifies it as v12). All 66 top-level function/class definitions retain their original bodies, signatures, defaults, and decorators.

## Files

Keep `pikn_v11.py` beside the `pikn_modules/` directory when uploading to GitHub or running locally. Include `pikn_modules/__init__.py`.

| Module | Responsibility |
| --- | --- |
| `layers.py` | MLP builder, raw and spectral encoders, fusion, decoder |
| `dynamics.py` | Wilson–Cowan cell and Koopman operator |
| `model.py` | Complete `TransitionPIKN` architecture |
| `config.py` | Training settings and output directories |
| `data.py` | Patient splits, shard loading, batches, data loaders |
| `losses.py` | Transition targets and loss functions |
| `metrics.py` | Probabilities, thresholds, calibration and classification metrics |
| `diagnostics.py` | Gradient checks, clipping and reconstruction checks |
| `checkpoints.py` | Save and restore model, optimizer and random states |
| `evaluation.py` | Evaluation loop and prediction extraction |
| `visualization.py` | Training curves, calibration plots and latent projections |
| `training.py` | Training loop and reporting |
| `cli.py` | Command-line arguments and main entry point |
| `utils.py` | Logging, runtime setup and shared helpers |

## Run

Use the same environment as the original script. Core dependencies are PyTorch, NumPy and Matplotlib. The original optional fallbacks for scikit-learn, umap-learn and tqdm are retained.

```bash
python pikn_v11.py --processed-dir tensors_pt --output-dir pipeline_output/v12_regularized
```

Existing imports in `reference.py`, including `_build_spectral_tensor`, continue to be exported by `pikn_v11.py`. New code can import the model directly:

```python
from pikn_modules.model import TransitionPIKN
from pikn_modules.config import TrainConfig
```

The data loader reads `raw_past`, `raw_t`, `raw_tdt`, spectral features, `label`, and `patient_idx` from the shards. With the accompanying preprocessing script, include `--save-raw` and retain the default patient-index output:

```bash
python Preproc_1.py --input-dir raw_h5 --output-dir tensors_pt --save-raw
```

## Verification

- All 66 original definitions compared equal as Python syntax trees, including decorators.
- Every module compiles; cross-module imports have no cycles and global references resolve statically.
- All seven imports used by the existing `reference.py` remain available.
- No numerical forward pass or training run was performed: PyTorch is unavailable in the editing environment. Run a small batch in your existing PyTorch environment before a full training run.

A separate correction was made to the earlier preprocessing split: download the updated `preproc_modules/config.py` from its original Drive location to restore the five dataclass decorators.
