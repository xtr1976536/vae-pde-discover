"""Optional VAE refinement and frozen-latent dynamics pretraining."""

import math

import torch
from torch.nn.utils import clip_grad_norm_

from task3_singlefield_v4.common import atomic_json


def masked_mse(prediction, target, mask):
    return (prediction - target).square()[..., mask].mean()


def phase_metrics(model, loader, device, stage, optimizer=None, kl_weight=1e-4):
    training = optimizer is not None
    model.train(training)
    model.encoder.train(training and stage == "vae")
    model.decoder.train(training and stage == "vae")
    total, accepted, attempted = 0., 0, 0
    with torch.set_grad_enabled(training):
        for _, history, target in loader:
            attempted += 1
            history = history.to(device)
            if stage == "vae":
                z, kl = model.encode_latent(history, sample=training)
                error = (model.decode(z) - history).square()[..., model.fluid, :].mean()
                loss = error + kl_weight * kl[..., model.latent_fluid].mean()
            else:
                target = target.to(device)
                with torch.no_grad():
                    latent, _ = model.encode_latent(torch.cat((history, target), dim=1))
                    z, future = latent.split((history.shape[1], target.shape[1]), dim=1)
                mean, scale = model.coefficient_distribution(z)
                coefficients = mean + scale * torch.randn_like(mean) if training else mean
                prediction, valid = (model.training_rollout(z[:, -1], coefficients) if training
                                     else model.rollout(z[:, -1], coefficients))
                if not valid.item():
                    continue
                error = masked_mse(prediction, future, model.latent_fluid)
                loss = error
            if not torch.isfinite(loss):
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = optimizer.param_groups[0]["params"]
                if not torch.isfinite(clip_grad_norm_(parameters, 1.)):
                    continue
                optimizer.step()
            accepted += 1
            total += float(error.detach())
            if training and attempted % 1000 == 0:
                print(f"{stage} step {attempted}/{len(loader)} updates={accepted}", flush=True)
    return dict(rmse=math.sqrt(total / accepted) if accepted else None,
                coverage=accepted / attempted, successful=accepted, windows=attempted)


def pretrain(model, loaders, device, args, report, path):
    vae = [*model.encoder.parameters(), *model.decoder.parameters()]
    stages = {"vae": args.vae_refine_epochs, "dynamics": args.dynamics_pretrain_epochs}
    report["pretraining"] = []
    for stage, epochs in stages.items():
        if not epochs:
            continue
        parameters = vae
        if stage == "dynamics":
            model.encoder.requires_grad_(False)
            model.decoder.requires_grad_(False)
            parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-4 if stage == "vae" else 3e-4)
        initial = phase_metrics(model, loaders["val"], device, stage)
        best = (initial["coverage"] < .95, -initial["coverage"] if initial["coverage"] < .95 else 0,
                initial["rmse"] if initial["rmse"] is not None else math.inf)
        chosen = {k: v.detach().clone() for k, v in model.state_dict().items()}
        record = dict(stage=stage, initial_validation=initial, epochs=[], best_epoch=0)
        report["pretraining"].append(record)
        for epoch in range(1, epochs + 1):
            train = phase_metrics(model, loaders["train"], device, stage, optimizer, args.vpc_kl_weight)
            val = phase_metrics(model, loaders["val"], device, stage)
            record["epochs"].append(dict(epoch=epoch, train=train, validation=val))
            if not train["successful"]:
                raise RuntimeError(f"No finite training updates during {stage}")
            if val["rmse"] is not None:
                key = (val["coverage"] < .95, -val["coverage"] if val["coverage"] < .95 else 0, val["rmse"])
                if key < best:
                    best = key
                    chosen = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    record["best_epoch"] = epoch
            atomic_json(path, report)
            print(f"{stage} epoch {epoch} validation={val} selected_epoch={record['best_epoch']}", flush=True)
        model.load_state_dict(chosen)
        model.encoder.requires_grad_(True)
        model.decoder.requires_grad_(True)
