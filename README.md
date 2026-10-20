![License](https://img.shields.io/badge/License-Research-green)

# Data-free BN watermark attack

A PyTorch implementation for the paper "**On the Vulnerabilities and Data-Free Removal of Watermarks in Batch Normalization Layers**" (WIFS). Demonstrates data-free watermark removal attacks on BN-based federated learning watermarking schemes.

Preprint available in arXiv: 

## Citation

If you use this code in your research, please cite the corresponding paper:
```bash
@misc{ 
}
```

## Overview

`attack.py` runs a data-free watermark-removal experiment against a
FedIPR or FedTracker model. For each selected client/model copy, the runner:

1. Evaluates the original model.
2. Re-initializes the corresponding BN parameters.
3. Reconstructs proxy samples from the model and uses them to tune the BN parameters.
4. Evaluates and records main task accuracy and watermark bit accuracy after each stage.

The script contains only the attack: it expects the corresponding upstream
FedIPR and FedTracker repositories and their model checkpoints.

## Requirements

Place the three repositories next to each other:

```text
<parent>/
├── BN-DF-Attack/
├── FedIPR/
└── FedTracker/
```

The runner resolves these repository paths relative to its own location. It
imports code from the upstream repositories at runtime. Keep their dependencies
installed in the Python environment used to run the attack, including:
```
torch
torchvision
numpy
pandas
tqdm
```

The repository is organized as follows:

- `attack.py`: command-line interface.
- `utils.py`: shared reconstruction, normalization, CSV, and pipeline helpers.
- `fedipr.py`: FedIPR-specific loading, BN-tuning, and evaluation.
- `fedtracker.py`: FedTracker-specific loading, BN-tuning, and evaluation.

### Prepare FedIPR

Apply the included `fedipr_modifications.patch` to the FedIPR repository
before running a FedIPR experiment. From the parent directory containing both
repositories:

```bash
git -C FedIPR apply --check ../BN-DF-Attack/fedipr_modifications.patch
git -C FedIPR apply ../BN-DF-Attack/fedipr_modifications.patch
```

The patch:

- Replaces hard-coded repository, data, trigger, and checkpoint locations with
  paths derived from the FedIPR checkout.
- Places generated checkpoints under `FedIPR/save/<model>/<dataset>/`.
- Makes FedIPR argument parsing accept an explicit argument list; the unified
  runner uses this to initialize FedIPR without parsing its own CLI arguments.
- Enables torchvision to download CIFAR-10/CIFAR-100 when they are missing.
- Updates dataset wrappers and client sampling for the current data access
  behavior, and replaces deprecated `np.int`.

The patch's default `data_root` is `FedIPR/data`. FedIPR expects its trigger
assets under that data directory as well. CIFAR datasets can be downloaded by
torchvision; the trigger images and labels used by the FedIPR configuration
must also be present at the paths expected by that checkout.

## Use

Run commands from `BN-DF-Attack/`. Use the Python environment containing the
dependencies of the attack runner and the selected upstream repository.

### FedTracker

Provide the FedTracker argument/configuration file and a model checkpoint. For
multiple model copies, `{cid}` in the model path is replaced by the client ID:

```bash
python attack_unified.py \
  --scheme fedtracker \
  --dataset cifar10 \
  --gpu 0 \
  --fedtracker-args-file ../FedTracker/result/VGG16/args.txt \
  --fedtracker-model-path '../FedTracker/result/VGG16/model_{cid}.pth' \
  --client-ids 0,1,2
```

Note that the fingerprint generation in FedTracker is not deterministic and may change if the attack is not run on the same environment as the model training. In such case, the watermark bit accuracy before the attack will be low, meaning the watermark extraction function will not be useful to determine the efficacy of the attack. The original local fingerprints and extracting matrices can be saved in `.npy` format and passed to the attack through the parser (`--fedtracker-local-fingerprints-file`, `--fedtracker-extracting-matrices-file`).

### FedIPR

Provide the FedIPR training log/checkpoint (`.pkl`). Its saved model name and
dataset are used unless overridden:

```bash
python attack_unified.py \
  --scheme fedipr \
  --dataset cifar10 \
  --gpu 0 \
  --fedipr-log-path ../FedIPR/save/alexnet/cifar10/checkpoint.pkl \
  --client-ids 0,1,2
```

FedIPR's `--client-ids` selects repeated attack runs against the loaded checkpoint; it does not load distinct per-client checkpoints, but the random proxy sample extraction and BN-tuning will provide different results. 

### Other useful configuration
 - `--epochs`: Number of BN-tuning epochs (default 100).
 - `--iterations`: Number of optimization steps for proxy image extraction (default 2000).
 - `--batch-size`: Batch size for the BN-tuning loop (default 16).
 - `--image-size`: Input image size (default 32). 

## Results

For each client ID, the runner appends one row to the results CSV with the
scheme, dataset, model, client ID, initial metrics, and metrics after
re-initialization and BN-tuning. 

## Disclaimer

This is code developed with research purposes only. The code may contain bugs or inefficiencies and is provided as-is for reproducibility purposes. We welcome bug reports and contributions.
