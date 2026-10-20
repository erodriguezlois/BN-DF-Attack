import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


FEDIPR_ROOT = Path(__file__).resolve().parent.parent / "FedIPR"


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


def run_fedipr_attack(args: argparse.Namespace) -> None:
    shared = _load_attack_utils()
    fedipr_str = str(FEDIPR_ROOT)
    if fedipr_str in sys.path:
        sys.path.remove(fedipr_str)
    sys.path.insert(0, fedipr_str)

    from main_fedIPR import IPRFederatedLearning
    from models.layers.passportconv2d_private import PassportPrivateBlock
    from utils.args import parser_args

    csv_file = args.output_csv or f"attack_results_{args.scheme}.csv"

    cfg = parser_args([])
    cfg.num_back = args.num_back
    cfg.num_trigger = args.num_trigger
    cfg.num_sign = args.num_sign
    cfg.num_bit = args.num_bit
    cfg.num_users = args.num_users
    cfg.dataset = args.dataset or cfg.dataset
    cfg.model_name = args.model_name or cfg.model_name
    cfg.epochs = args.epochs
    cfg.gpu = str(args.gpu)
    cfg.local_ep = args.local_ep
    cfg.iid = args.iid
    cfg.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    cfg.batch_size = args.batch_size
    log_path = args.fedipr_log_path or args.fedipr_log_path_default
    logs = torch.load(log_path, map_location=cfg.device, weights_only=False)
    fl = IPRFederatedLearning(cfg)

    mean, std = args.norm_stats or shared.DEFAULT_NORMALIZATION_STATS["fedipr"].get(
        cfg.dataset, ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    )

    def prepare_state(cid, args, cfg):
        fl.model.load_state_dict(logs["net_info"]["best_model"])
        fl.keys = logs["net_info"]["keys"][0]
        fl.model.to(cfg.device)
        orig_model = copy.deepcopy(fl.model)
        orig_model.to(cfg.device)
        test_loader = DataLoader(
            fl.test_set,
            batch_size=fl.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return {
            "model": fl.model,
            "original_model": orig_model,
            "test_dataset": fl.test_set,
            "val_loader": test_loader,
            "backend": fl,
        }

    def reinitialize(state, cid, args, cfg):
        fl = state["backend"]
        for wm_layer in fl.model.modules():
            if isinstance(wm_layer, PassportPrivateBlock):
                bn_layer = wm_layer.bn
                fake_data = (
                    torch.randn(
                        (1000, wm_layer.scale.shape[0], 32, 32), device=cfg.device
                    )
                    * torch.sqrt(bn_layer.running_var).view(
                        1, wm_layer.scale.shape[0], 1, 1
                    )
                    + bn_layer.running_mean.view(1, wm_layer.scale.shape[0], 1, 1)
                )
                fake_data_output_ref = (
                    bn_layer(fake_data)
                    * wm_layer.scale.view(1, wm_layer.scale.shape[0], 1, 1)
                    + wm_layer.bias.view(1, wm_layer.scale.shape[0], 1, 1)
                )
                mu_o, std_o = shared.channel_stats(fake_data_output_ref.detach())
                wm_layer.scale = torch.nn.Parameter(
                    torch.ones_like(wm_layer.scale) * std_o
                )
                wm_layer.bias = torch.nn.Parameter(
                    torch.ones_like(wm_layer.bias) * mu_o
                )

    def bntune(state, cid, args, cfg):
        fl = state["backend"]
        fl.model.train()
        for name, param in fl.model.named_parameters():
            if (
                ("scale" not in name and "bias" not in name)
                or ("linear" in name)
                or ("classifier" in name)
            ):
                param.requires_grad = False
            else:
                param.requires_grad = True

        fl.model = fl.model.to(cfg.device)
        fl.optimizer = torch.optim.SGD(
            fl.model.parameters(), fl.lr, momentum=0.9, weight_decay=0.0005
        )

        for _ in tqdm(range(args.epochs), desc=f"FedIPR client {cid}"):
            if cfg.dataset == "cifar100":
                target_labels = list(np.random.choice(np.arange(100), size=args.batch_size))
            elif cfg.dataset == "cifar10":
                target_labels = list(np.random.choice(np.arange(10), size=args.batch_size))
            else:
                raise ValueError(f"Unsupported dataset: {cfg.dataset}")
            pseudo_input_dataset = shared.reconstruct_batch(
                state["original_model"],
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
            for x, y in pseudo_dataloader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                fl.optimizer.zero_grad()
                pred = fl.model(x)
                loss = F.cross_entropy(pred, y)
                loss.backward()
                fl.optimizer.step()

    def evaluate(state, phase, cid, args, cfg):
        fl = state["backend"]
        average_bacc = []
        for idx in range(fl.num_users):
            private_sign_acc = fl.tester.test_signature(fl.keys[idx], fl.scheme)
            print(f"* User {idx} private sign Bit Accuracy: {private_sign_acc:.4f}")
            average_bacc.append(private_sign_acc)
        bacc = float(np.mean(average_bacc))
        print(f"* Average Bit Accuracy: {bacc:.4f}")

        loss_val_mean, acc = fl.trainer.test(state["val_loader"])
        print(f"* Accuracy: {acc * 100:.2f}")

        if phase == "initial":
            state["initial_acc"] = acc
            state["initial_bacc"] = bacc
            return {"initial_acc": acc, "initial_bacc": bacc}
        initial_acc = state.get("initial_acc")
        if initial_acc is not None:
            print("* Accuracy loss:", (initial_acc - acc) * 100, "%")
        if phase == "after_reinitialization":
            return {"acc_after_reinitialization": acc, "bacc_after_reinitialization": bacc}
        return {"acc_after_bntune": acc, "bacc_after_bntune": bacc}

    shared.run_shared_attack_pipeline(
        args,
        cfg,
        scheme=args.scheme,
        model_name=cfg.model_name,
        csv_file=csv_file,
        prepare_state=prepare_state,
        reinitialize=reinitialize,
        bntune=bntune,
        evaluate=evaluate,
    )
