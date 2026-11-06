import csv
from datetime import datetime, timezone
import json
import math
import time

import numpy as np
import torch

from models.dgst_former import DGSTFormer
from .common import resolve_path, runtime, save_json
from .data import dataset_signature, loaders, processed_dir
from .metrics import MetricAccumulator, masked_mae


class EarlyStopping:
    def __init__(self, patience, min_epochs, min_delta):
        self.patience, self.min_epochs, self.min_delta = patience, min_epochs, min_delta
        self.best = math.inf
        self.bad_epochs = 0

    def update(self, score, epoch):
        if score < self.best - self.min_delta:
            self.best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience


def evaluate(model, loader, adjacency, device, mape_threshold=0.0):
    model.eval()
    meta = loader.dataset.meta
    metrics = MetricAccumulator(loader.dataset.output_length, mape_threshold)
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device), adjacency) * meta["std"] + meta["mean"]
            metrics.update(prediction, y.to(device))
            seen += len(y)
    if seen != len(loader.dataset):
        raise RuntimeError("Evaluation did not traverse the complete partition")
    result = metrics.result()
    if any(value is None or not math.isfinite(value) for value in result["overall"].values()):
        raise FloatingPointError("Evaluation has missing or nonfinite metrics")
    return {**result, "samples": seen}


def build_model(config, folder, device):
    matrix = np.load(folder / "adjacency.npy", allow_pickle=False)
    nodes = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["sensors"]
    if matrix.shape != (nodes, nodes) or not np.isfinite(matrix).all() or not np.array_equal(matrix, matrix.T):
        raise ValueError("Expected a finite symmetric road adjacency matching the sensor count")
    adjacency = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    return DGSTFormer(config, adjacency).to(device), adjacency


def new_run_dir(config, output_root="outputs", output_dir=None):
    if output_dir is not None:
        folder = resolve_path(output_dir)
        folder.parent.mkdir(parents=True, exist_ok=True)
        try:
            folder.mkdir()
        except FileExistsError:
            raise FileExistsError("Output directory already exists; choose another --output-dir or omit it for a new run") from None
        return folder
    base = resolve_path(output_root)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{config['dataset']}_seed{config['training']['seed']}_{stamp}"
    for index in range(100):
        folder = base / (name if index == 0 else f"{name}_{index}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise FileExistsError("Unable to allocate a new run directory")


def save_checkpoint(path, model, config, signature, epoch, score):
    pending = path.with_suffix(".tmp")
    torch.save({
        "state_dict": model.state_dict(), "config": config, "data_signature": signature,
        "epoch": epoch, "validation_MAE": score,
    }, pending)
    pending.replace(path)


def fit(config, output_root="outputs", output_dir=None):
    tc = config["training"]
    device = runtime(tc["device"], tc["seed"], tc["deterministic"])
    data_folder = processed_dir(config)
    signature = dataset_signature(data_folder)
    data = loaders(config, device)
    model, adjacency = build_model(config, data_folder, device)
    optimizer_type = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[tc.get('optimizer', 'Adam')]
    optimizer = optimizer_type(
        model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
        fused=device.type == "cuda",
    )
    if tc.get('scheduler', 'plateau') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=tc.get('lr_factor', .5),
            patience=tc.get('lr_patience', 3), min_lr=min(tc["learning_rate"], tc.get('min_lr', 1e-5)))
    elif tc['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=tc['lr_milestones'], gamma=tc['lr_gamma'])
    elif tc['scheduler'] == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {tc['scheduler']}")
    stopping = EarlyStopping(tc["patience"], tc["min_epochs"], tc["min_delta"])
    folder = new_run_dir(config, output_root, output_dir)
    save_json(folder / "parameters.json", config)
    meta = data["train"].dataset.meta
    best, best_epoch = math.inf, 0
    fields = ["epoch", "train_MAE", "training_loss", "weighted_jacobian_regularization",
              "validation_MAE", "validation_RMSE", "validation_MAPE", "learning_rate", "epoch_seconds"]
    print(f"Run: {folder}\nDevice: {device}; train/val/test windows: "
          + "/".join(str(len(data[s].dataset)) for s in ("train", "val", "test")), flush=True)
    with open(folder / "training.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, tc["epochs"] + 1):
            start = time.perf_counter()
            model.train()
            loss_sum = mae_sum = penalty_sum = 0.0
            valid_count = seen = 0
            for batch, (x, y) in enumerate(data["train"], 1):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                penalty = x.new_zeros(())
                if tc.get('loss', 'mae_jacobian') == 'mae_jacobian' and tc["lambda_diss"] > 0:
                    prediction, info = model(x, adjacency, return_regularization=True)
                    penalty = tc["lambda_diss"] * info["dissipation_loss"]
                else:
                    prediction = model(x, adjacency)
                mae = masked_mae(prediction * meta["std"] + meta["mean"], y)
                if tc.get('loss') == 'huber':
                    mask = torch.isfinite(y) & (y > 0)
                    base = torch.nn.functional.huber_loss((prediction * meta['std'] + meta['mean'])[mask],
                                                         y[mask], delta=tc.get('huber_delta', 2.))
                else:
                    base = mae
                loss = base + penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                count = int((torch.isfinite(y) & (y > 0)).sum())
                loss_sum += float(loss.detach()) * count
                mae_sum += float(mae.detach()) * count
                penalty_sum += float(penalty.detach()) * count
                valid_count += count
                seen += len(y)
                if batch == 1 or batch == len(data["train"]) or batch % 100 == 0:
                    print(f"Epoch {epoch}/{tc['epochs']}: {batch}/{len(data['train'])} train batches", flush=True)
            if seen != len(data["train"].dataset):
                raise RuntimeError("Training did not traverse the complete partition")
            validation = evaluate(model, data["val"], adjacency, device, tc["mape_threshold"])
            score = validation["overall"]["MAE"]
            if score < best:
                best, best_epoch = score, epoch
                save_checkpoint(folder / "best.pt", model, config, signature, epoch, score)
            stop = stopping.update(score, epoch)
            row = {
                "epoch": epoch, "train_MAE": mae_sum / valid_count,
                "training_loss": loss_sum / valid_count,
                "weighted_jacobian_regularization": penalty_sum / valid_count,
                **{"validation_" + key: validation["overall"][key] for key in ("MAE", "RMSE", "MAPE")},
                "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - start,
            }
            writer.writerow(row)
            handle.flush()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            elif scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch}: validation MAE={score:.4f}; best={best:.4f} (epoch {best_epoch})", flush=True)
            if tc["early_stopping"] and stop:
                break
    checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test = evaluate(model, data["test"], adjacency, device, tc["mape_threshold"])
    result = {
        "dataset": config["dataset"], "seed": tc["seed"], "epochs_completed": epoch,
        "best_epoch": best_epoch, "validation_MAE": best, "test": test,
        "stop_reason": "early_stopping" if tc["early_stopping"] and stop else "epochs_complete",
        "parameters": sum(p.numel() for p in model.parameters()),
        "workbook_reference": config.get('workbook'),
    }
    save_json(folder / "result.json", result)
    print(f"Test: {test['overall']}\nCheckpoint: {folder / 'best.pt'}", flush=True)
    return folder, result


def evaluate_checkpoint(checkpoint_path, device="auto", data_root=None, batch_size=None):
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    tc = config["training"]
    if batch_size is not None:
        tc["batch_size"] = batch_size
    runtime_device = runtime(device, tc["seed"], tc["deterministic"])
    folder = processed_dir(config, data_root)
    if dataset_signature(folder) != checkpoint["data_signature"]:
        raise ValueError("Processed data or normalization differs from the checkpoint's training data")
    model, adjacency = build_model(config, folder, runtime_device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    data = loaders(config, runtime_device, data_root, splits=("test",))
    result = evaluate(model, data["test"], adjacency, runtime_device, tc["mape_threshold"])
    save_json(checkpoint_path.with_suffix(".evaluation.json"), result)
    return result
