#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path

from fedipr import run_fedipr_attack
from fedtracker import run_fedtracker_attack


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


_utils = _load_attack_utils()
parse_stats = _utils.parse_stats
set_seed = _utils.set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Data-free watermark removal attack of watermarks in batch "
            "normalization layers"
        )
    )
    parser.add_argument(
        "--scheme",
        choices=["fedipr", "fedtracker"],
        required=True,
        help="Watermarking scheme to attack",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name (cifar10 or cifar100)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of BN-tuning epochs")
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Number of optimization steps for proxy image extraction",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the BN-tuning loop")
    parser.add_argument("--image-size", type=int, default=32, help="Input image size")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument(
        "--client-ids",
        default="0",
        help="Comma-separated client IDs to attempt the attack",
    )
    parser.add_argument("--output-csv", default=None, help="Output CSV file for experiment summary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--norm-stats",
        type=parse_stats,
        default=None,
        help=(
            "Override dataset normalization statistics as "
            "mean1,mean2,mean3,std1,std2,std3"
        ),
    )

    parser.add_argument(
        "--fedtracker-args-file",
        default=None,
        help="Path to the FedTracker args.json file",
    )
    parser.add_argument(
        "--fedtracker-model-path",
        default=None,
        help="Path to the FedTracker model checkpoint",
    )
    parser.add_argument(
        "--fedtracker-local-fingerprints-file",
        default=None,
        help="Path to the FedTracker local fingerprints file",
    )
    parser.add_argument(
        "--fedtracker-extracting-matrices-file",
        default=None,
        help="Path to the FedTracker extracting matrices file",
    )
    parser.add_argument("--fedipr-log-path", default=None, help="Path to the FedIPR .pkl checkpoint")
    parser.add_argument(
        "--fedipr-log-path-default",
        default=None,
        help="Fallback FedIPR checkpoint path",
    )
    parser.add_argument("--num-clients", type=int, default=None, help="Override the number of clients")
    parser.add_argument("--lfp-length", type=int, default=None, help="Override the fingerprint length")
    parser.add_argument("--num-back", type=int, default=10)
    parser.add_argument("--num-trigger", type=int, default=10)
    parser.add_argument("--num-sign", type=int, default=10)
    parser.add_argument("--num-bit", type=int, default=40)
    parser.add_argument("--num-users", type=int, default=10)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--local-ep", type=int, default=2)
    parser.add_argument("--iid", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    if args.scheme == "fedtracker":
        if args.fedtracker_args_file is None:
            raise ValueError("--fedtracker-args-file is required for --scheme fedtracker")
        if args.fedtracker_model_path is None:
            raise ValueError("--fedtracker-model-path is required for --scheme fedtracker")
        args.client_ids = [int(item) for item in args.client_ids.split(",") if item.strip()]
        run_fedtracker_attack(args)
    elif args.scheme == "fedipr":
        if args.fedipr_log_path is None and args.fedipr_log_path_default is None:
            raise ValueError(
                "Provide --fedipr-log-path or --fedipr-log-path-default for --scheme fedipr"
            )
        args.client_ids = [int(item) for item in args.client_ids.split(",") if item.strip()]
        run_fedipr_attack(args)
    else:
        raise ValueError(f"Unsupported scheme: {args.scheme}")


if __name__ == "__main__":
    main()
