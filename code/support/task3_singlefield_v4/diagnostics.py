"""Forecast gradients, latent diagnostics, and per-window PDE inference."""

import numpy as np
import torch

from .common import atomic_json
from .model import get_frames
from .operator_library import operator_name


def forecast_gradient_norms(model, loss):
    names = ["encoder", "decoder"]
    names += ["projection", "transformer", "coefficients"] if hasattr(model, "solver") else ["forecaster"]
    if getattr(model, "initial_velocity", None) is not None:
        names.append("initial_velocity")
    groups = [tuple(param for param in getattr(model, name).parameters() if param.requires_grad)
              for name in names]
    params = [param for group in groups for param in group]
    grads = iter(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True))
    result = {}
    for name, group in zip(names, groups):
        values = [next(grads) for _ in group]
        squares = [g.detach().abs().double().square().sum() for g in values if g is not None]
        norm = float(torch.stack(squares).sum().sqrt()) if squares else 0.
        result[name] = norm if np.isfinite(norm) else None
    return result


@torch.no_grad()
def latent_diagnostics(model, data, span, training_center):
    model.eval()
    obs = data[slice(*span)]
    z, mean, logvar = model.encode(obs)
    error = (model.decode(z) - obs).square().mean()
    without_latent = (model.decode(torch.full_like(z, training_center)) - obs).square().mean()
    shuffle = torch.Generator(device=z.device).manual_seed(552)
    shuffled_z = z.flatten()[torch.randperm(z.numel(), generator=shuffle, device=z.device)].reshape_as(z)
    shuffled = (model.decode(shuffled_z) - obs).square().mean()
    metrics = dict(z_std=float(z.std(unbiased=False)), z_spatial_std=float(z.std(1, unbiased=False).mean()),
                   posterior_mean_std=float(mean.std(unbiased=False)),
                   posterior_kl=float((.5 * (mean.square() + logvar.exp() - 1 - logvar)).mean()),
                   reconstruction_rmse=float(error.sqrt()),
                   constant_latent_rmse=float(without_latent.sqrt()), shuffled_latent_rmse=float(shuffled.sqrt()))
    metrics["collapse_screen_pass"] = (metrics["z_std"] > 1e-3
                                       and metrics["z_spatial_std"] > 1e-3
                                       and float(error) < .99 * min(float(without_latent), float(shuffled)))
    return metrics


@torch.no_grad()
def export_pdes(model, data, ids, x, args):
    model.eval()
    solver = model.solver
    names = [operator_name(term) for term in solver.library]
    time_order = 2 if solver.mode == "second" else 1
    generator = torch.Generator(device=data.device).manual_seed(args.seed + 3)
    rows, states, contributions, draws = [], [], [], []
    for origins in ids.split(args.batch):
        history = get_frames(data, origins, 1 - args.history, 1)
        target = get_frames(data, origins, 1, args.steps + 1)
        z, velocity, _, _, _, mean, logvar = model.prepare(history)
        initial = z[:, -1]
        std = (logvar * .5).exp()
        latent, valid = solver.rollout(initial, velocity, mean, args.steps, check=True)
        forecast = model.decode(latent)
        parts = solver.contributions(latent, mean)
        parts[~valid] = torch.nan
        latent[~valid] = torch.nan
        states.append(latent.cpu().numpy())
        contributions.append(parts.cpu().numpy())
        ablations = []
        for j in range(len(names)):
            coefficients = mean.clone()
            coefficients[:, j] = 0.
            ablated, ok = solver.rollout(initial, velocity, coefficients, args.steps, check=True)
            pred = model.decode(ablated)
            change = (pred - forecast).square().flatten(1).mean(1).sqrt()
            error = (pred - target).square().flatten(1).mean(1).sqrt()
            ablations.append([(float(a), float(b)) if flag else None
                              for a, b, flag in zip(change, error, ok & valid)])
        sampled = []
        for _ in range(args.pde_draws):
            noise = torch.randn(mean.shape, device=mean.device, generator=generator)
            sample, ok = solver.rollout(initial, velocity, mean + std * noise, args.steps, check=True)
            sample[~ok] = torch.nan
            sampled.append((sample.cpu().numpy(), ok.cpu().numpy()))
        if sampled:
            draws.append(np.stack([a for a, _ in sampled], axis=1))
        means = (mean / solver.coefficient_divisors).cpu().numpy()
        stds = (std / solver.coefficient_divisors).cpu().numpy()
        for i, origin in enumerate(origins.tolist()):
            rhs = " ".join(f"{coef:+.8g}*{name.replace(' ', '*')}" for name, coef in zip(names, means[i]))
            rows.append(dict(origin=origin, valid=bool(valid[i]), pde=f"z_{'t' * time_order} = {rhs}",
                             coefficient_mean=means[i].tolist(), coefficient_std=stds[i].tolist(),
                             ablation=[dict(observation_change_rmse=a[i][0], target_rmse=a[i][1])
                                       if a[i] is not None else None for a in ablations],
                             draw_successful=[bool(ok[i]) for _, ok in sampled]))
    atomic_json(args.output / "test_pdes.json", dict(terms=names, time_order=time_order, windows=rows))
    arrays = dict(x=x, origin=ids.cpu().numpy(), latent=np.concatenate(states),
                  contribution=np.concatenate(contributions))
    if draws:
        arrays["sampled_latent"] = np.concatenate(draws)
    np.savez_compressed(args.output / "test_pde_fields.npz", **arrays)
