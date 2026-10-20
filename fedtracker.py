import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


FEDTRACKER_ROOT = Path(__file__).resolve().parent.parent / "FedTracker"


def _load_attack_utils():
    module_name = "_attack_unified_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name("utils.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the shared attack utilities from utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def fedtracker_extracting_fingerprints_idx(
    layers, local_fingerprints, extracting_matrices, idx, epsilon=0.5, hd=False
):
    bit_length = local_fingerprints[0].shape[0]
    weight = layers[0].weight.detach().numpy()
    for index in range(1, len(layers)):
        weight = np.append(weight, layers[index].weight.detach().numpy())
    matrix = extracting_matrices[idx]
    result = np.dot(matrix, weight)
    if hd:
        result[result >= 0] = 1
        result[result < 0] = -1
        return np.sum(result != local_fingerprints[idx]) / bit_length

    result = np.multiply(result, local_fingerprints[idx])
    result[result > epsilon] = epsilon
    return np.sum(result) / bit_length / epsilon


def run_fedtracker_attack(args: argparse.Namespace) -> None:
    shared = _load_attack_utils()
    fedtracker_str = str(FEDTRACKER_ROOT)
    if fedtracker_str in sys.path:
        sys.path.remove(fedtracker_str)
    sys.path.insert(0, fedtracker_str)

    from fed.client import create_clients
    from utils.datasets import get_full_dataset
    from utils.models import get_model
    from utils.test import test_img
    from utils.train import get_optim
    from watermark.fingerprint import (
        extracting_fingerprints,
        generate_extracting_matrices,
        generate_fingerprints,
        get_embed_layers,
        get_embed_layers_length,
    )

    csv_file = args.output_csv or f"attack_results_{args.scheme}.csv"

    with open(args.fedtracker_args_file, "r") as handle:
        cfg = argparse.Namespace(**json.load(handle))
    cfg.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    cfg.gpu = args.gpu
    cfg.dataset = args.dataset or cfg.dataset
    cfg.image_size = args.image_size
    if args.num_clients is not None:
        cfg.num_clients = args.num_clients
    if args.lfp_length is not None:
        cfg.lfp_length = args.lfp_length
    train_dataset, test_dataset = get_full_dataset(
        cfg.dataset, img_size=(cfg.image_size, cfg.image_size)
    )
    model = get_model(cfg)
    weight_size = get_embed_layers_length(model, cfg.embed_layer_names)
    if args.fedtracker_local_fingerprints_file and args.fedtracker_extracting_matrices_file:
        local_fingerprints = np.load(args.fedtracker_local_fingerprints_file, allow_pickle=True)
        extracting_matrices = np.load(args.fedtracker_extracting_matrices_file, allow_pickle=True)
    else:
        local_fingerprints = generate_fingerprints(cfg.num_clients, cfg.lfp_length)
        extracting_matrices = generate_extracting_matrices(
            weight_size, cfg.lfp_length, cfg.num_clients
        )
    clients = create_clients(cfg, train_dataset)

    mean, std = args.norm_stats or shared.DEFAULT_NORMALIZATION_STATS["fedtracker"].get(
        cfg.dataset, ((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
    )

    def prepare_state(cid, args, cfg):
        model_path = shared.resolve_model_path(args.fedtracker_model_path, cid)
        model_state = get_model(cfg)
        model_state.load_state_dict(
            torch.load(model_path, map_location=cfg.device, weights_only=True)
        )
        original_model = copy.deepcopy(model_state)
        return {
            "model": model_state,
            "original_model": original_model,
            "test_dataset": test_dataset,
        }

    def reinitialize(state, cid, args, cfg):
        model = state["model"]
        for bn_layer in model.modules():
            if isinstance(bn_layer, torch.nn.BatchNorm2d):
                bn_layer.to(cfg.device)
                fake_data = (
                    torch.randn(
                        (1000, bn_layer.weight.shape[0], 32, 32), device=cfg.device
                    )
                    * torch.sqrt(bn_layer.running_var).view(
                        1, bn_layer.weight.shape[0], 1, 1
                    )
                    + bn_layer.running_mean.view(1, bn_layer.weight.shape[0], 1, 1)
                )
                fake_data_output_ref = bn_layer(fake_data)
                mu_o, std_o = shared.channel_stats(fake_data_output_ref.detach())
                bn_layer.weight = torch.nn.Parameter(
                    torch.ones_like(bn_layer.weight) * std_o
                )
                bn_layer.bias = torch.nn.Parameter(
                    torch.ones_like(bn_layer.bias) * mu_o
                )

    def bntune(state, cid, args, cfg):
        model = state["model"]
        original_model = state["original_model"]
        model.train()
        for name, param in model.named_parameters():
            if "bn" not in name:
                param.requires_grad = False

        model = model.to(cfg.device)
        optim = get_optim(
            model,
            clients[cid].local_optim,
            clients[cid].local_lr,
            clients[cid].local_momentum,
        )

        for _ in tqdm(range(args.epochs), desc=f"FedTracker client {cid}"):
            if cfg.dataset == "cifar100":
                target_labels = list(np.random.choice(np.arange(100), size=args.batch_size))
            elif cfg.dataset == "cifar10":
                target_labels = list(np.random.choice(np.arange(10), size=args.batch_size))
            else:
                raise ValueError(f"Unsupported dataset: {cfg.dataset}")
            pseudo_input_dataset = shared.reconstruct_batch(
                original_model,
                target_classes_list=target_labels,
                iterations=args.iterations,
                mean=mean,
                std=std,
            )
            pseudo_dataset = torch.utils.data.TensorDataset(
                pseudo_input_dataset, torch.tensor(target_labels)
            )
            pseudo_dataloader = DataLoader(
                pseudo_dataset, batch_size=args.batch_size, shuffle=True
            )
            for images, labels in pseudo_dataloader:
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                model.zero_grad()
                probs = model(images)
                loss = clients[cid].loss(probs, labels)
                loss.backward()
                optim.step()

        state["model"] = model

    def evaluate(state, phase, cid, args, cfg):
        model = state["model"]
        model.to("cpu")
        embed_layers = get_embed_layers(model, cfg.embed_layer_names)
        ber, extract_idx = extracting_fingerprints(
            embed_layers, local_fingerprints, extracting_matrices, hd=True
        )
        print("* Fingerprint recognized as:", extract_idx, "Bit Accuracy:", 1 - ber)
        ber = fedtracker_extracting_fingerprints_idx(
            embed_layers, local_fingerprints, extracting_matrices, cid, hd=True
        )
        print("* Correct Bit Accuracy:", 1 - ber)

        model.to(cfg.device)
        acc, _ = test_img(model, state["test_dataset"], cfg)
        print("* Accuracy:", acc)

        if phase == "initial":
            state["initial_acc"] = acc
            state["initial_bacc"] = 1 - ber
            return {"initial_acc": acc / 100, "initial_bacc": 1 - ber}
        initial_acc = state.get("initial_acc")
        if initial_acc is not None:
            print("* Accuracy loss:", (initial_acc - acc), "%")
        if phase == "after_reinitialization":
            return {
                "acc_after_reinitialization": acc / 100,
                "bacc_after_reinitialization": 1 - ber,
            }
        return {"acc_after_bntune": acc / 100, "bacc_after_bntune": 1 - ber}

    shared.run_shared_attack_pipeline(
        args,
        cfg,
        scheme=args.scheme,
        model_name=Path(args.fedtracker_model_path).parent.name,
        csv_file=csv_file,
        prepare_state=prepare_state,
        reinitialize=reinitialize,
        bntune=bntune,
        evaluate=evaluate,
    )
