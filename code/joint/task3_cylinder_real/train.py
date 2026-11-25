"""Train on the isolated 20-to-20 Cylinder real-data split."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from task3_singlefield_v4.common import atomic_json
from task3_singlefield_v4.train import selection_key

from .data import CylinderStore, Windows
from .models import TERMS, CylinderModel, RealPDEBenchBaseline
from .splits import HORIZON, ROLES


def spatial_mean(values, fluid):
    return values[..., fluid, :].mean(dim=(-1, -2))


def subset(rows, size):
    if not size or size >= len(rows):
        return rows
    return [rows[i] for i in np.linspace(0, len(rows) - 1, size).round().astype(int)]


@torch.no_grad()
def evaluate(model, loader, device, scale, details=False):
    model.eval()
    errors = torch.zeros(HORIZON, 2, device=device, dtype=torch.float64)
    normalized = torch.zeros_like(errors)
    reconstruction = zero_latent = 0.
    coefficients, locations = [], []
    success = 0
    for ids, history, target in loader:
        history, target = history.to(device), target.to(device)
        prediction, decoded, valid, _, mean, std = model(history)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {prediction.shape} differs from target {target.shape}")
        by_field = (prediction - target).square()[..., model.fluid, :].mean(-2)
        valid &= torch.isfinite(by_field.flatten(1)).all(1)
        normalized += by_field[valid].sum(0, dtype=torch.float64)
        errors += (by_field[valid] * scale.square()).sum(0, dtype=torch.float64)
        if decoded is not None:
            reconstruction += float(spatial_mean((decoded - history).square(), model.fluid).mean(1).sum())
        if mean is not None:
            zero = model.decode_zeros(history)
            zero_latent += float(spatial_mean((zero - history).square(), model.fluid).mean(1).sum())
            if details:
                coefficients.extend(zip(mean.tolist(), std.tolist(), valid.tolist()))
        success += int(valid.sum())
        if details:
            locations.extend([dict(**loader.dataset.rows[i], successful=ok) for i, ok in zip(ids, valid.tolist())])
    n = len(loader.dataset)
    result = dict(windows=n, successful=success, coverage=success / n, rmse=None)
    if model.kind == "vpc":
        result["reconstruction_rmse"] = math.sqrt(reconstruction / n)
        result["zero_latent_rmse"] = math.sqrt(zero_latent / n)
    if success:
        result.update(rmse=float((normalized.mean() / success).sqrt()),
                      rmse_by_step_and_field=(errors / success).sqrt().tolist(),
                      rmse_by_field=(errors.mean(0) / success).sqrt().tolist())
    if details:
        result["window_status"] = locations
    if coefficients:
        result["pde_terms"] = TERMS
        result["pde_coordinates"] = "grid columns, grid rows, snapshot intervals"
        if model.spatial:
            result["pde_coefficient_axes"] = ["target_channel", "source_channel", "term"]
            result["pde_coordinates"] = "latent grid columns, latent grid rows, snapshot intervals"
        result["pde_windows"] = [dict(sim_id=row["sim_id"], time_id=row["time_id"],
                                      coefficient_mean=mean, coefficient_std=std, successful=ok)
                                 for row, (mean, std, ok) in zip(locations, coefficients)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("vpc", "unet", "fno"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=525)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--vpc-width", type=int, default=32)
    parser.add_argument("--vpc-layers", type=int, default=2)
    parser.add_argument("--vpc-heads", type=int, default=4)
    parser.add_argument("--vpc-encoder", choices=("spatial", "pointwise"), default="spatial")
    parser.add_argument("--vpc-latent-channels", type=int, default=2)
    parser.add_argument("--vpc-latent-stride", type=int, choices=(2, 4), default=4)
    parser.add_argument("--vpc-kl-weight", type=float, default=1e-4)
    parser.add_argument("--realpdebench-config", type=Path)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_windows < 0:
        parser.error("epochs must be positive and max-windows nonnegative")
    if min(args.vpc_width, args.vpc_layers, args.vpc_heads) < 1 or args.vpc_width % args.vpc_heads:
        parser.error("VPC width must be divisible by a positive head count and layers must be positive")
    if args.vpc_encoder == "spatial" and args.vpc_latent_channels < 2:
        parser.error("Spatial VPC requires at least two latent channels")
    if args.vpc_kl_weight < 0 or not math.isfinite(args.vpc_kl_weight):
        parser.error("VPC KL weight must be finite and nonnegative")
    if args.model != "vpc" and args.realpdebench_config is None:
        parser.error("Specify the official Cylinder model config with --realpdebench-config")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    store = CylinderStore(args.shards, args.splits)
    rows = {role: subset(store.splits[role], args.max_windows) for role in ROLES}
    mean, std = store.normalize(rows["train"])
    loaders = {role: DataLoader(Windows(store, rows[role], mean, std), batch_size=1,
                                shuffle=role == "train",
                                generator=torch.Generator().manual_seed(args.seed + 1))
               for role in ROLES if role != "test" or args.test}
    device = torch.device(args.device)
    scale = torch.from_numpy(std).to(device)
    capacity = dict(width=args.vpc_width, layers=args.vpc_layers, heads=args.vpc_heads)
    fluid = torch.from_numpy(store.fluid)
    if args.model == "vpc":
        model = CylinderModel(args.model, fluid, **capacity, spatial=args.vpc_encoder == "spatial",
                          latent_channels=args.vpc_latent_channels,
                          latent_stride=args.vpc_latent_stride).to(device)
        model.capture_training_rollout()
    else:
        import yaml

        config = yaml.safe_load(args.realpdebench_config.read_text())
        if config["model_name"] != args.model or config["dataset_name"] != "cylinder":
            parser.error("The RealPDEBench config must match the requested Cylinder baseline")
        _, history_sample, target_sample = loaders["train"].dataset[0]
        model = RealPDEBenchBaseline(args.model, fluid, (history_sample, target_sample),
                                    config, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", model=args.model, fields=["u", "v"], history=20, horizon=20,
                  batch_size=1, seed=args.seed, learning_rate=3e-4, epoch_limit=args.epochs,
                  vpc_kl_weight=args.vpc_kl_weight if args.model == "vpc" else None,
                  windows={role: len(rows[role]) for role in ROLES},
                  split_sha256=store.audit["prepared_sha256"], mean=mean.tolist(), scale=std.tolist(),
                  parameters=sum(p.numel() * (2 if p.is_complex() else 1)
                                 for p in model.parameters() if p.requires_grad),
                  fluid_points=int(store.fluid.sum()), epochs=[])
    if args.model == "vpc":
        report["transformer"] = capacity
        report["latent"] = dict(encoder=args.vpc_encoder, channels=model.latent_channels,
                                stride=args.vpc_latent_stride if args.vpc_encoder == "spatial" else None,
                                shape=list(model.latent_fluid.shape))
    else:
        import inspect

        report["realpdebench"] = dict(config=str(args.realpdebench_config.resolve()),
                                     implementation=inspect.getfile(type(model.forecaster)),
                                     input_shape=list(history_sample.shape),
                                     output_shape=list(target_sample.shape),
                                     objective="masked_forecast_mse")
    path = args.output / "results.json"
    atomic_json(path, report)
    best, best_epoch = (3, 0., math.inf), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        samples = solved = 0
        forecast_total = recon_total = kl_total = 0.
        gradients = None
        for step, (_, history, target) in enumerate(loaders["train"], start=1):
            history, target = history.to(device), target.to(device)
            forecast, reconstruction, valid, kl, *_ = model(history)
            solved += int(valid.sum())
            if not valid.any():
                continue
            target_valid = target[valid]
            if forecast.shape != target_valid.shape:
                raise ValueError(f"Prediction shape {forecast.shape} differs from target {target_valid.shape}")
            forecast_loss = spatial_mean((forecast - target_valid).square(), model.fluid).mean()
            loss = forecast_loss
            if reconstruction is not None:
                recon_loss = spatial_mean((reconstruction - history[valid]).square(), model.fluid).mean()
                penalty = kl[..., model.latent_fluid].mean()
                loss = loss + recon_loss + args.vpc_kl_weight * penalty
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            if gradients is None:
                names = (["encoder", "decoder", "projection", "transformer", "coefficients"]
                         if args.model == "vpc" else ["forecaster"])
                groups = {name: tuple(p for p in getattr(model, name).parameters() if p.requires_grad)
                          for name in names}
                grads = iter(torch.autograd.grad(forecast_loss, [p for group in groups.values() for p in group],
                                                 retain_graph=True, allow_unused=True))
                audit = {}
                for name, group in groups.items():
                    values = [next(grads) for _ in group]
                    norm = math.sqrt(sum(float(g.detach().abs().double().square().sum())
                                         for g in values if g is not None))
                    audit[name] = norm if math.isfinite(norm) else None
                if all(norm is not None and norm > 0 for norm in audit.values()):
                    gradients = audit
            loss.backward()
            if not torch.isfinite(clip_grad_norm_(model.parameters(), 1.)):
                continue
            optimizer.step()
            samples += 1
            forecast_total += float(forecast_loss.detach())
            if reconstruction is not None:
                recon_total += float(recon_loss.detach())
                kl_total += float(penalty.detach())
            if step % 100 == 0:
                print(f"{args.model} epoch {epoch} step {step}/{len(loaders['train'])} updates={samples}", flush=True)
        val = evaluate(model, loaders["val"], device, scale)
        losses = dict(forecast_mse=forecast_total / samples) if samples else None
        if losses is not None and args.model == "vpc":
            losses.update(reconstruction_mse=recon_total / samples, latent_kl=kl_total / samples)
        report["epochs"].append(dict(epoch=epoch, attempted=len(loaders["train"]), solver_success=solved,
                                     optimizer_samples=samples, loss=losses,
                                     forecast_gradient_norms=gradients, validation=val))
        key = selection_key(val, .95)
        if samples and gradients and val["successful"] and key < best:
            best, best_epoch = key, epoch
            torch.save(model.state_dict(), args.output / "best.pt")
            report["best_epoch"] = epoch
        atomic_json(path, report)
        print(f"{args.model} epoch {epoch} updates={samples} val_RMSE={val['rmse']} coverage={val['coverage']}", flush=True)
        if epoch - best_epoch >= 5:
            break
    if not best_epoch:
        raise RuntimeError("No checkpoint with finite forecast gradients and validation predictions")
    if args.test:
        model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=True))
        report["test"] = evaluate(model, loaders["test"], device, scale, details=True)
    report["status"] = "COMPLETE"
    atomic_json(path, report)


if __name__ == "__main__":
    main()
