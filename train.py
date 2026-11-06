import argparse
import math
from pathlib import Path

from utils.download import DATASETS
from utils.training import fit
from utils.workbook import DEFAULT_WORKBOOK, build_config as workbook_config


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def parser():
    p = argparse.ArgumentParser(description="Train DGST-Former on chronological PEMS partitions")
    p.add_argument("--dataset", choices=DATASETS, default="PEMS04")
    p.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    p.add_argument("--print-config", action="store_true", help="Print resolved settings without loading data or training")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-root", default="outputs", help="Parent of a new, uniquely named run")
    p.add_argument("--output-dir", help="Optional exact run directory; it must not already exist")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=positive_int)
    p.add_argument("--batch-size", type=positive_int)
    p.add_argument("--learning-rate", type=nonnegative_float)
    p.add_argument("--weight-decay", type=nonnegative_float, default=0.0)
    p.add_argument("--grad-clip", type=nonnegative_float, default=5.0)
    p.add_argument("--lambda-diss", type=nonnegative_float, default=0.001)
    p.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--patience", type=positive_int)
    p.add_argument("--min-epochs", type=positive_int, default=15)
    p.add_argument("--min-delta", type=nonnegative_float, default=0.01)
    p.add_argument("--mape-threshold", type=nonnegative_float, default=0.0)
    p.add_argument("--input-length", type=positive_int, default=12)
    p.add_argument("--output-length", type=positive_int, default=12)
    p.add_argument("--hidden-dim", type=positive_int)
    p.add_argument("--latent-dim", type=positive_int)
    p.add_argument("--feed-forward-dim", type=positive_int)
    p.add_argument("--num-layers", type=positive_int)
    p.add_argument("--num-heads", type=positive_int)
    p.add_argument("--dropout", type=nonnegative_float)
    p.add_argument("--gamma-min", type=nonnegative_float, default=0.01)
    p.add_argument("--gamma-max", type=nonnegative_float, default=0.10)
    p.add_argument("--contraction-rho", type=nonnegative_float, default=0.99)
    p.add_argument("--jacobian-power-iterations", type=positive_int, default=1)
    p.add_argument("--encoder-precision", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--checkpoint-dynamics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action="store_true")
    return p


def build_config(args):
    authoritative = workbook_config(args.dataset, workbook=args.workbook)
    for field in ('epochs', 'batch_size', 'learning_rate', 'patience'):
        value = getattr(args, field)
        if value is not None and value != authoritative['training'][field]:
            raise ValueError(f'{field} is controlled by {args.workbook}; update the workbook to change it')
        setattr(args, field, authoritative['training'][field])
    if args.dropout is not None and args.dropout != authoritative['model']['dropout']:
        raise ValueError(f'dropout is controlled by {args.workbook}; update the workbook to change it')
    args.dropout = authoritative['model']['dropout']
    if args.early_stopping is not None and args.early_stopping != authoritative['training']['early_stopping']:
        raise ValueError('Early stopping is controlled by the workbook')
    args.early_stopping = authoritative['training']['early_stopping']
    for field in ('hidden_dim', 'latent_dim', 'feed_forward_dim', 'num_layers', 'num_heads'):
        if getattr(args, field) is None:
            setattr(args, field, authoritative['model'][field])
    config = {
        "dataset": args.dataset,
        "data": {
            "processed_root": str(Path(args.data_root) / "processed/main"),
            "input_length": args.input_length, "output_length": args.output_length,
        },
        "model": {
            "name": "DGSTFormer", "activation": "relu", "evolution_mode": "gated",
            "graph_constraint": True, "encoder_graph": True, "gamma_mode": "learned",
            "gamma_conditioning": True, "update_conditioning": True, "delta_scale": 1.0,
            "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
            "feed_forward_dim": args.feed_forward_dim, "num_layers": args.num_layers,
            "num_heads": args.num_heads, "dropout": args.dropout,
            "gamma_min": args.gamma_min, "gamma_max": args.gamma_max,
            "contraction_rho": args.contraction_rho,
            "jacobian_power_iterations": args.jacobian_power_iterations, "jacobian_seed": args.seed,
            "encoder_precision": args.encoder_precision, "checkpoint_dynamics": args.checkpoint_dynamics,
        },
        "training": {
            "device": args.device, "seed": args.seed, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
            "lambda_diss": args.lambda_diss, "early_stopping": args.early_stopping,
            "patience": args.patience, "min_epochs": args.min_epochs, "min_delta": args.min_delta,
            "mape_threshold": args.mape_threshold, "deterministic": args.deterministic,
        },
    }
    config['workbook'] = authoritative['workbook']
    config['training'].update({key: authoritative['training'][key] for key in
                              ('optimizer', 'loss', 'scheduler', 'lr_factor', 'lr_patience', 'min_lr', 'lr_gamma', 'lr_milestones', 'huber_delta')})
    return config


def main():
    p = parser()
    args = p.parse_args()
    try:
        config = build_config(args)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    if args.hidden_dim % args.num_heads:
        p.error("hidden-dim must be divisible by num-heads")
    if args.dropout >= 1:
        p.error("dropout must be less than 1")
    if not 0 <= args.gamma_min <= args.gamma_max < 1:
        p.error("require 0 <= gamma-min <= gamma-max < 1")
    if args.learning_rate <= 0 or args.grad_clip <= 0:
        p.error("learning-rate and grad-clip must be positive")
    if args.seed < 0 or args.seed >= 2**32:
        p.error("seed must be between 0 and 2**32-1")
    if args.print_config:
        import json
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    fit(config, args.output_root, args.output_dir)


if __name__ == "__main__":
    main()
