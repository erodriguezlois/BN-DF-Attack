import argparse
import os
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T


DEFAULT_NORMALIZATION_STATS = {
    "fedipr": {
        "cifar10": ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        "cifar100": ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    },
    "fedtracker": {
        "cifar10": ((0.491, 0.482, 0.447), (0.247, 0.244, 0.262)),
        "cifar100": ((0.491, 0.482, 0.447), (0.247, 0.244, 0.262)),
    },
}


def parse_stats(
    value: Optional[str],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if value is None:
        raise ValueError(
            "Normalization statistics must be provided as --norm-stats mean,std,mean,std"
        )
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "Expected six comma-separated values: mean1,mean2,mean3,std1,std2,std3"
        )
    mean = tuple(float(item) for item in parts[:3])
    std = tuple(float(item) for item in parts[3:])
    return mean, std


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def channel_stats(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y_flat = y.transpose(0, 1).reshape(y.shape[1], -1)
    mu = y_flat.mean(dim=(0, 1))
    std = y_flat.std(dim=(0, 1), unbiased=False)
    return mu, std


def reconstruct_batch(
    target_classifier: torch.nn.Module,
    target_classes_list: Sequence[int],
    iterations: int = 2000,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    image_size: int = 32,
) -> torch.Tensor:
    
    target_classifier.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_classifier.to(device)
    
    batch_size = len(target_classes_list)
    target_labels = torch.tensor(target_classes_list, dtype=torch.long, device=device)
    
    # 1. Initialize our bounded image space parameter
    # For CIFAR (32x32), we optimize directly at 32x32 resolution
    w = torch.rand((batch_size, 3, 32, 32), requires_grad=True, device=device)
    
    # Lowered LR to 0.02 to prevent the mid-loop explosion you saw
    optimizer = torch.optim.Adam([w], lr=0.02)
    # Cosine Annealing decreases the LR smoothly to 0 by the end of the run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=0.0001)

    # Simple spatial augmentations to break up pixel-level artifacts/textures
    augmentations = torch.nn.Sequential(
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(p=0.5)
    )

    # Setup Batch Normalization tracking hooks
    bn_layers = [m for m in target_classifier.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    layer_activations = {}
    def hook_fn(name):
        return lambda m, inp, out: layer_activations.__setitem__(name, inp[0])
    hooks = [l.register_forward_hook(hook_fn(f"bn_{i}")) for i, l in enumerate(bn_layers)]

    # Standard normalization stats
    if mean is None:
        mean = (0.5, 0.5, 0.5)
    if std is None:
        std = (0.5, 0.5, 0.5)
    mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    for step in range(iterations):
        optimizer.zero_grad()
        layer_activations.clear()
        
        # 1. Compute the standard deviation across spatial dimensions (per channel, per batch item)
        # Shape: (Batch_Size, 3, 1, 1)
        pixel_std = torch.std(w, dim=[0, 2, 3]).view(1, 3, 1, 1)
        pixel_mean = torch.mean(w, dim=[0, 2, 3]).view(1, 3, 1, 1)

        # Target a coherent standard deviation.
        std_loss = torch.norm(std - pixel_std, 2) 
        mean_loss = torch.norm(mean - pixel_mean, 2)

        # Apply predefined normalization (same as main task)
        images_0_1 = (w - mean) / std

        if step < int(iterations * 0.85):
            # Apply augmentations for the first 85% of the run to build the structure
            inputs_aug = augmentations(images_0_1)
        else:
            # Strict fine-tuning pass for the final 15% to sharpen edges
            inputs_aug = images_0_1
        
        # Evaluate model
        outputs = target_classifier(inputs_aug)
        
        # Compute Distributed Batch Norm Statistics Metric
        bn_loss = 0.0
        for i, layer in enumerate(bn_layers):
            act = layer_activations.get(f"bn_{i}")
            if act is not None:
                c_mean = torch.mean(act, dim=[0, 2, 3])
                c_var = torch.var(act, dim=[0, 2, 3], unbiased=False)
                bn_loss += torch.norm(c_mean - layer.running_mean, 2) + torch.norm(torch.sqrt(c_var) - torch.sqrt(layer.running_var), 2)
                
        ce_loss = F.cross_entropy(outputs, target_labels)
        
        # Total Variation Loss to smoothly bind spatial pixel deltas
        tv_loss = torch.pow(images_0_1[:, :, 1:, :] - images_0_1[:, :, :-1, :], 2).sum() + \
                  torch.pow(images_0_1[:, :, :, 1:] - images_0_1[:, :, :, :-1], 2).sum()
        
        # Balanced loss calculation
        total_loss = total_loss = (1.0 * bn_loss) + (5.0 * ce_loss) + (1e-4 * tv_loss) + (10.0 * (std_loss + mean_loss))
        total_loss.backward()

        # Keep w bounded in [0, 1]
        with torch.no_grad():
            w.clamp_(0.0, 1.0)
        
        optimizer.step()
        scheduler.step() # Step the scheduler to reduce learning rate momentum stability
    for h in hooks: h.remove()
    
    with torch.no_grad():
        final_batch = ((w - mean) / std).detach().cpu()
            
    return final_batch


def prepare_output_csv(csv_file: str) -> pd.DataFrame:
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame()


def resolve_model_path(model_path: str, cid: int) -> str:
    if model_path is None:
        return model_path

    candidate = model_path.format(cid=cid) if "{cid}" in model_path else model_path
    if os.path.isdir(candidate):
        return os.path.join(candidate, f"model_{cid}.pth")
    if os.path.exists(candidate):
        return candidate

    root, extension = os.path.splitext(candidate)
    alternate_candidate = f"{root}_{cid}{extension}"
    if os.path.exists(alternate_candidate):
        return alternate_candidate
    return candidate


def run_shared_attack_pipeline(
    args: argparse.Namespace,
    cfg: argparse.Namespace,
    *,
    scheme: str,
    model_name: str,
    csv_file: str,
    prepare_state,
    reinitialize,
    bntune,
    evaluate,
) -> None:
    all_exp_df = prepare_output_csv(csv_file)

    for cid in args.client_ids:
        experiment = {
            "scheme": scheme,
            "dataset": cfg.dataset,
            "model": model_name,
            "cid": cid,
        }

        print(f"Client ID: {cid}")
        state = prepare_state(cid, args, cfg)
        print("Running initial evaluation...")
        experiment.update(evaluate(state, phase="initial", cid=cid, args=args, cfg=cfg))

        print("Running BN-reinitialization...")
        reinitialize(state, cid, args, cfg)
        experiment.update(
            evaluate(state, phase="after_reinitialization", cid=cid, args=args, cfg=cfg)
        )

        print("Running BN-tuning...")
        bntune(state, cid, args, cfg)
        experiment.update(
            evaluate(state, phase="after_bntune", cid=cid, args=args, cfg=cfg)
        )

        all_exp_df = pd.concat([all_exp_df, pd.DataFrame([experiment])], ignore_index=True)
        all_exp_df.to_csv(csv_file, index=False)
