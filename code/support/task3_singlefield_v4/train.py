"""Train one forecasting model on one of the eight single-field cases."""

import argparse
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .baselines.adapter import SingleFieldBaseline
from .common import atomic_json
from .diagnostics import export_pdes, forecast_gradient_norms, latent_diagnostics
from .model import CASES, Solver, VPCTransformer, get_frames, kl, load_trajectory, windows
from .operator_library import catalogue, operators


def finite_rows(tensor):
    return torch.isfinite(tensor.flatten(1)).all(1)


def vpc_batch(model, history, target, coefficient_kl):
    z, velocity, coeff, mean, logvar, loc, logvar_c = model.prepare(history)
    last = z[:, -1]
    with torch.no_grad():
        _, usable = model.solver.rollout(last, velocity, coeff, target.shape[1], check=True)
    ids = usable.nonzero().flatten()
    if not ids.numel():
        return None, ids, {}
    velocity = None if velocity is None else velocity[ids]
    future = model.solver.rollout(last[ids], velocity, coeff[ids], target.shape[1])
    forecast = (model.decode(future) - target[ids]).square().flatten(1).mean(1)
    recon = (model.decode(z[ids]) - history[ids]).square().flatten(1).mean(1)
    latent_kl = .5 * (mean[ids].square() + logvar[ids].exp() - 1 - logvar[ids]).flatten(1).mean(1)
    parts = dict(forecast_mse=forecast, reconstruction_mse=recon, latent_kl=latent_kl)
    loss = forecast + recon + 1e-4 * latent_kl
    if coefficient_kl:
        parts["coefficient_kl"] = (.5 * (
            (loc[ids].square() + logvar_c[ids].exp()) / .15**2 - 1
            + 2 * math.log(.15) - logvar_c[ids])).mean(1)
        loss = loss + coefficient_kl * parts["coefficient_kl"]
    return loss, ids, dict(total=loss, **parts)


def update(model, optimizer, history, target, is_vpc, coefficient_kl, audit):
    optimizer.zero_grad(set_to_none=True)
    if is_vpc:
        losses, ids, parts = vpc_batch(model, history, target, coefficient_kl)
        solved = len(ids)
    else:
        forecast, reconstruction = model(history)
        forecast_loss = (forecast - target).square().flatten(1).mean(1)
        recon_loss = (reconstruction - history).square().flatten(1).mean(1)
        losses = forecast_loss + recon_loss
        parts = dict(total=losses, forecast_mse=forecast_loss, reconstruction_mse=recon_loss)
        solved = len(history)
    if losses is None:
        return solved, 0, None, {}
    accepted = len(losses)
    parameters = tuple(model.parameters())
    gradient_audit = None
    if torch.isfinite(losses).all():
        if audit:
            gradient_audit = forecast_gradient_norms(model, parts["forecast_mse"].mean())
        losses.mean().backward()
    norm = nn.utils.clip_grad_norm_(parameters, 1.)
    if not torch.isfinite(losses).all() or not torch.isfinite(norm):
        if not is_vpc:
            raise FloatingPointError("Nonfinite baseline training loss or gradient")
        # Repeat only the problem batch as individual draws so contaminated
        # gradients cannot enter an optimizer update for a valid sample.
        optimizer.zero_grad(set_to_none=True)
        accepted = 0
        recovered = {name: [] for name in parts}
        for i in ids.tolist():
            loss, _, single_parts = vpc_batch(model, history[i:i + 1], target[i:i + 1], coefficient_kl)
            if loss is None or not torch.isfinite(loss).all():
                continue
            grads = torch.autograd.grad(loss.mean(), parameters)
            if not all(torch.isfinite(grad).all() for grad in grads):
                continue
            for param, grad in zip(parameters, grads):
                param.grad = grad if param.grad is None else param.grad + grad
            accepted += 1
            for name, value in single_parts.items():
                recovered[name].append(value.detach())
        if not accepted:
            return solved, 0, None, {}
        for param in parameters:
            param.grad.div_(accepted)
        norm = nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            return solved, 0, None, {}
        parts = {name: torch.cat(values) for name, values in recovered.items()}
    optimizer.step()
    sums = torch.stack([value.detach().sum() for value in parts.values()]).cpu().tolist()
    return solved, accepted, gradient_audit, dict(zip(parts, sums))


@torch.no_grad()
def evaluate(model, data, ids, args, per_window=False):
    model.eval()
    squared = torch.zeros(args.steps, device=data.device, dtype=torch.float64)
    coefficient_batches = []
    success = 0
    window_errors = {}
    for origins in ids.split(args.batch):
        history = get_frames(data, origins, 1 - args.history, 1)
        target = get_frames(data, origins, 1, args.steps + 1)
        if args.model == "vpc":
            z, velocity, coef, *_ = model.prepare(history)
            forecast, valid = model.solver.rollout(z[:, -1], velocity, coef, args.steps, check=True)
            forecast = model.decode(forecast)
        else:
            forecast, _ = model(history)
            valid = torch.ones(len(history), device=history.device, dtype=torch.bool)
        error = (forecast - target).square().mean((2, 3))
        valid = valid & finite_rows(error)
        squared += error[valid].sum(0, dtype=torch.float64)
        success += int(valid.sum())
        if per_window:
            window_errors.update({str(origin): row.tolist() if ok else None
                                  for origin, row, ok in zip(origins.tolist(), error.cpu(), valid.tolist())})
        if args.model == "vpc" and valid.any():
            coefficient_batches.append(coef[valid].cpu())
    result = dict(windows=len(ids), successful=success, failed=len(ids) - success,
                  coverage=success / len(ids), rmse=None, rmse_by_step=None)
    if success:
        result.update(rmse=float((squared.mean() / success).sqrt()),
                      rmse_by_step=(squared / success).sqrt().tolist())
    if per_window:
        result["window_mse_by_step"] = window_errors
    if coefficient_batches:
        result["coefficient_mean_normalized"] = torch.cat(coefficient_batches).mean(0).tolist()
    return result


def selection_key(metrics, coverage_floor):
    if metrics["rmse"] is None:
        return (2, 0., math.inf)
    coverage = metrics["coverage"]
    return (0, 0., metrics["rmse"]) if coverage >= coverage_floor else (1, -coverage, metrics["rmse"])


def choose_windows(ids, cap):
    if not cap or len(ids) <= cap:
        return ids
    return ids[torch.linspace(0, len(ids) - 1, cap, device=ids.device).round().long()]


def learning_rate(args, epoch):
    if args.schedule == "fixed" or epoch <= args.hold_epochs:
        return args.lr
    span = max(1, args.epochs - args.hold_epochs)
    fraction = (epoch - args.hold_epochs) / span
    return args.lr_min + .5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * fraction))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vpc", "unet", "fno"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--library", choices=("small", "large"), default="small")
    for name, default in dict(points=256, history=5, steps=5, batch=4, epochs=60, seed=525,
                              width=32, layers=2, heads=4, modes=12, substeps=2,
                              warmup=0, max_windows=0, patience=10, pde_draws=8, hold_epochs=10, min_epochs=20).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in dict(lr=3e-4, dropout=0., coefficient_std=.003,
                              coefficient_kl=0., coverage_floor=.95, lr_min=3e-5).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--schedule", choices=("fixed", "delayed_cosine"), default="fixed")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args(argv)
    if (args.points < 32 or args.points % 32 or args.steps not in (3, 5, 10, 20)
            or args.history not in (1, args.steps) or min(args.batch, args.epochs, args.substeps,
                                                         args.width, args.heads, args.layers) < 1
            or min(args.patience, args.pde_draws, args.hold_epochs, args.min_epochs) < 0 or args.max_windows < 0 or args.warmup < 0 or not 0 <= args.dropout < 1
            or not 0 < args.coverage_floor <= 1 or not 0 <= args.lr_min <= args.lr or not 0 < args.lr
            or not 0 < args.coefficient_std or not 0 <= args.coefficient_kl):
        parser.error("Invalid training configuration")
    if args.model == "vpc" and args.width % args.heads:
        parser.error("VPC width must be divisible by the attention head count")
    if args.model == "unet" and args.width % 8:
        parser.error("U-Net width must be divisible by eight")
    if args.model == "fno" and not 1 <= args.modes <= (args.points + 6) // 2:
        parser.error("FNO spatial mode count exceeds the padded grid")
    if args.model != "vpc" and args.warmup:
        parser.error("VAE warmup is only available for VPC")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data, x, dt, mode, split = load_trajectory(args.data, args.case, args.points, 525)
    data = data.to(args.device)
    is_vpc = args.model == "vpc"
    result = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  dt=dt, split=split, mode=mode, epochs=[], status="RUNNING")
    data_folder = (args.data / CASES[args.case][0]).parent
    data_files = [args.data / CASES[args.case][0]]
    if args.case == "chaffee":
        data_files += [data_folder / "x.npy", data_folder / "t.npy"]
    result["data_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files}
    if is_vpc:
        provenance = catalogue()
        library = operators(args.library)
        solver = Solver(x, dt, mode, library, args.substeps)
        model = VPCTransformer(solver, args.width, args.layers, args.heads,
                               args.dropout, args.coefficient_std)
        result.update(library=library, library_fields=["power_of_z", "spatial_derivative_order"],
                      coefficient_divisors=solver.coefficient_divisors.tolist(),
                      corpus_sha256=provenance["sha256"])
    else:
        model = SingleFieldBaseline(args.model, args.history, args.steps, args.points,
                                    args.width, args.modes)
    model = model.to(args.device)
    train, val, test = [choose_windows(windows(a, b, args.steps, args.steps, data.device),
                                       args.max_windows) for a, b in zip(split, split[1:])]
    if min(map(len, (train, val, test))) == 0:
        parser.error("History and horizon do not fit the split")
    result["parameters"] = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "results.json"
    atomic_json(path, result)

    if args.warmup:
        pretrain = torch.optim.Adam([*model.encoder.parameters(), *model.decoder.parameters()], lr=1e-3)
        for _ in range(args.warmup):
            ids = torch.randint(split[1], (args.batch,), device=data.device)
            current = data[ids]
            z, mean, logvar = model.encode(current)
            loss = (model.decode(z) - current).square().mean() + 1e-4 * kl(mean, logvar)
            pretrain.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            pretrain.step()

    torch.manual_seed(args.seed + 1)
    permutation = torch.Generator(device=data.device).manual_seed(args.seed + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state = (3, 0., math.inf), None
    result["early_stopped"] = False
    stale = 0
    for epoch in range(1, args.epochs + 1):
        lr = learning_rate(args, epoch)
        optimizer.param_groups[0]["lr"] = lr
        model.train()
        solved = accepted = 0
        gradients = None
        loss_totals = {}
        for origins in train[torch.randperm(len(train), device=data.device, generator=permutation)].split(args.batch):
            history = get_frames(data, origins, 1 - args.history, 1)
            target = get_frames(data, origins, 1, args.steps + 1)
            batch_solved, batch_accepted, audit, parts = update(model, optimizer, history, target,
                                                         is_vpc, args.coefficient_kl, gradients is None)
            if audit and all(value is not None and value > 0 for value in audit.values()):
                gradients = audit
            solved += batch_solved
            accepted += batch_accepted
            for name, value in parts.items():
                loss_totals[name] = loss_totals.get(name, 0.) + value
        metrics = evaluate(model, data, val, args)
        result["epochs"].append(dict(epoch=epoch, windows=len(train), solver_success=solved,
                                     optimizer_samples=accepted, validation=metrics,
                                     forecast_gradient_norms=gradients, lr=lr,
                                     train_loss={name: total / accepted for name, total in loss_totals.items()}))
        key = selection_key(metrics, args.coverage_floor)
        if accepted and metrics["successful"] and key < best:
            best, result["best_epoch"] = key, epoch
            stale = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        atomic_json(path, result)
        print(f"{args.case}/{args.model} epoch {epoch}: updates {accepted}/{len(train)}, "
              f"validation coverage {metrics['coverage']:.1%}, RMSE {metrics['rmse']}", flush=True)
        if args.patience and stale >= args.patience and args.min_epochs <= epoch < args.epochs:
            result["early_stopped"] = True
            break
    if best_state is None:
        result["status"] = "NO_VALID_CHECKPOINT"
        atomic_json(path, result)
        raise RuntimeError("No epoch had both a training update and valid validation forecasts")
    result["coverage_pass"] = best[0] == 0
    model.load_state_dict(best_state)
    torch.save(best_state, args.output / "best.pt")
    result["selected_validation"] = evaluate(model, data, val, args, per_window=True)
    if is_vpc:
        model.eval()
        with torch.no_grad():
            center = float(model.encode(data[:split[1]])[0].mean())
        result["latent_validation"] = latent_diagnostics(model, data, split[1:3], center)
    if args.test:
        result["test"] = evaluate(model, data, test, args, per_window=True)
        if is_vpc:
            result["latent_test"] = latent_diagnostics(model, data, split[2:], center)
            export_pdes(model, data, test, x, args)
    result["status"] = "COMPLETE"
    atomic_json(path, result)


if __name__ == "__main__":
    main()
