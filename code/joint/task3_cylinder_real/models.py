"""Cylinder forecasting models and masked two-dimensional latent PDE rollout."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from task3_singlefield_v4.common import network

from .splits import HISTORY, HORIZON

TERMS = ("z", "z^2", "z^3", "z_x", "z_y", "z_xx", "z_yy")


class SpatialEncoder(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        if stride == 2:
            first = nn.Conv2d(2, 32, 4, stride=2, padding=1)
            second = nn.Conv2d(32, 2 * channels, 3, padding=1)
        elif stride == 4:
            first = nn.Conv2d(2, 16, 3, stride=2, padding=1)
            second = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        else:
            raise ValueError("Spatial VPC stride must be 2 or 4")
        self.layers = nn.Sequential(
            first, nn.GELU(), second, nn.GELU(),
            *((nn.Conv2d(32, 2 * channels, 3, padding=1),) if stride == 4 else ()),
        )

    def forward(self, history):
        encoded = self.layers(history.flatten(0, 1).movedim(-1, 1))
        return encoded.unflatten(0, history.shape[:2])


class SpatialDecoder(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        stages = [nn.Conv2d(channels, 32, 3, padding=1), nn.GELU()]
        for _ in range(1 if stride == 2 else 2):
            stages.extend((nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                           nn.Conv2d(32, 16, 3, padding=1), nn.GELU()))
            stages.append(nn.Conv2d(16, 32, 3, padding=1))
        stages.extend((nn.Conv2d(32, 2, 3, padding=1),))
        self.layers = nn.Sequential(
            *stages,
        )

    def forward(self, latent):
        decoded = self.layers(latent.flatten(0, 1)).movedim(1, -1)
        return decoded.unflatten(0, latent.shape[:2])


class RealPDEBenchBaseline(nn.Module):
    def __init__(self, name, fluid, sample, config, device):
        super().__init__()
        from realpdebench.model.load_model import load_model

        self.kind = name
        self.register_buffer("fluid", fluid)
        self.forecaster = load_model([sample], device=device, **config)

    def forward(self, history):
        forecast = self.forecaster(history)
        valid = torch.ones(len(history), device=history.device, dtype=torch.bool)
        return forecast, None, valid, None, None, None


class CylinderModel(nn.Module):
    def __init__(self, name, fluid, *, width=32, layers=2, heads=4, spatial=False,
                 latent_channels=2, latent_stride=4):
        super().__init__()
        self.kind = name
        self.register_buffer("fluid", fluid)
        self.spatial = name == "vpc" and spatial
        self.latent_channels = latent_channels if self.spatial else 1
        self.latent_stride = latent_stride
        self.training_rollout = self.single_window_rollout
        if self.spatial:
            self.encoder = SpatialEncoder(self.latent_channels, latent_stride)
            self.decoder = SpatialDecoder(self.latent_channels, latent_stride)
            self.register_buffer("coarse_fluid",
                                 F.avg_pool2d(fluid.float()[None, None], latent_stride)[0, 0] > .5)
        else:
            self.encoder = network((2, 32, 16, 2 if name == "vpc" else 1), nn.Tanh)
            self.decoder = network((1, 16, 32, 2), nn.Tanh)
        if name == "unet":
            from task3_singlefield_v4.baselines.unet import Unet3d

            self.forecaster = Unet3d(dim=16, channels=1, out_channels=1, dim_mults=(1, 2, 4),
                                     in_time=HISTORY, out_time=HORIZON, attn_dim_head=16)
        if name == "fno":
            from task3_singlefield_v4.baselines.fno import FNO3d

            shape = (HISTORY, *fluid.shape, 1)
            self.forecaster = FNO3d(4, 12, 16, 4, 16, shape, (HORIZON, *fluid.shape, 1))
        if name == "vpc":
            self.projection = nn.Linear(self.latent_channels, width)
            coefficient_pool = 4 if self.spatial else 1
            coefficient_shape = (self.latent_fluid.shape[0] // coefficient_pool,
                                 self.latent_fluid.shape[1] // coefficient_pool)
            self.position = nn.Parameter(
                torch.randn(1, 1, coefficient_shape[0] * coefficient_shape[1], width) * .02)
            self.time_position = nn.Parameter(torch.randn(1, HISTORY, 1, width) * .02)
            layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0., batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, layers)
            self.coefficients = nn.Linear(width, 2 * self.latent_channels**2 * len(TERMS))
            nn.init.normal_(self.coefficients.weight, std=.0003)
            nn.init.zeros_(self.coefficients.bias)
            mask = self.latent_fluid
            masks = torch.stack([torch.roll(mask, shift, axis)
                                 for axis in (-1, -2) for shift in (-1, 1)])
            masks[0, :, -1] = masks[1, :, 0] = False
            masks[2, -1, :] = masks[3, 0, :] = False
            self.register_buffer("neighbors", masks)
            self.pool_stride = latent_stride if self.spatial else 16
            self.register_buffer("pool_count",
                                 F.avg_pool2d(mask.float()[None, None], self.pool_stride).clamp_min(.001))
            self.register_buffer("coefficient_pool_count",
                                 F.avg_pool2d(mask.float()[None, None], coefficient_pool)
                                 .clamp_min(.001))
            self.coefficient_pool_stride = coefficient_pool

    @property
    def latent_fluid(self):
        return self.coarse_fluid if self.spatial else self.fluid

    def decode(self, z):
        return self.decoder(z if self.spatial else z.unsqueeze(-1))

    def decode_zeros(self, history):
        shape = (*history.shape[:2], *self.latent_fluid.shape)
        if self.spatial:
            shape = (*shape[:2], self.latent_channels, *shape[2:])
        return self.decode(history.new_zeros(shape))

    def features(self, z):
        right, left, down, up = [torch.where(mask, torch.roll(z, shift, axis), z)
                                 for mask, (axis, shift) in zip(
                                     self.neighbors, ((-1, -1), (-1, 1), (-2, -1), (-2, 1)))]
        return torch.stack((z, z.square(), z.pow(3), (right - left) * .5, (down - up) * .5,
                            right + left - 2 * z, down + up - 2 * z), dim=-1)

    def rollout(self, z, coefficient, check=True, single_training_window=False):
        frames = []
        substeps = []
        valid = torch.ones(z.shape[0], device=z.device, dtype=torch.bool)
        for _ in range(HORIZON):
            for _ in range(2):
                features = self.features(z)
                if self.spatial:
                    tendency = torch.einsum("bihwk,boik->bohw", features, coefficient)
                else:
                    tendency = (features * coefficient[:, None, None]).sum(-1)
                z = torch.where(self.latent_fluid, z + .5 * tendency, 0.)
                if single_training_window:
                    substeps.append(z.detach())
                elif check:
                    valid = valid & (z.abs().flatten(1).amax(1) <= 100.)
                    z = torch.where(valid.view(-1, *([1] * (z.ndim - 1))), z, 0.)
            frames.append(z)
        if single_training_window:
            valid = torch.stack(substeps).abs().flatten(2).amax((0, 2)) <= 100.
        return torch.stack(frames, dim=1), valid

    def single_window_rollout(self, z, coefficients):
        return self.rollout(z, coefficients, single_training_window=True)

    def capture_training_rollout(self):
        if not self.spatial or self.fluid.device.type != "cuda":
            return
        sample = torch.zeros((1, self.latent_channels, *self.latent_fluid.shape),
                             device=self.fluid.device, requires_grad=True)
        coefficients = torch.zeros((1, self.latent_channels, self.latent_channels, len(TERMS)),
                                   device=self.fluid.device, requires_grad=True)
        self.training_rollout = torch.cuda.make_graphed_callables(
            self.single_window_rollout, (sample, coefficients))

    def forward(self, history):
        encoded = self.encoder(history)
        kl, mean, scale = None, None, None
        if self.kind == "vpc":
            loc, logvar = encoded.chunk(2, dim=2) if self.spatial else encoded.unbind(-1)
            logvar = (logvar - 4).clamp(-11, 1)
            kl = .5 * (loc.square() + logvar.exp() - 1 - logvar)
            raw = loc + (logvar * .5).exp() * torch.randn_like(loc) if self.training else loc
            z = raw.tanh() * self.latent_fluid
            channels = z if self.spatial else z.unsqueeze(2)
            pooled = (F.avg_pool2d(channels.flatten(0, 1), self.coefficient_pool_stride)
                      / self.coefficient_pool_count)
            summary = pooled.flatten(2).transpose(1, 2).unflatten(0, z.shape[:2])
            tokens = self.projection(summary) + self.position + self.time_position
            hidden = self.transformer(tokens.flatten(1, 2)).mean(1)
            mean, log_scale = self.coefficients(hidden).chunk(2, -1)
            mean = .01 * mean
            scale = .003 * F.softplus(.1 * log_scale) / math.log(2)
            if self.spatial:
                coefficient_shape = (self.latent_channels, self.latent_channels, len(TERMS))
                mean = mean.unflatten(-1, coefficient_shape)
                scale = scale.unflatten(-1, coefficient_shape)
            coefficients = mean + scale * torch.randn_like(mean) if self.training else mean
            if self.training and history.shape[0] == 1:
                future, valid = self.training_rollout(z[:, -1], coefficients)
                # The training loop skips the entire window after any invalid substep.
                if not valid.item():
                    return self.decode(future[:0]), self.decode(z[:0]), valid, kl[:0], mean, scale
                return self.decode(future), self.decode(z), valid, kl, mean, scale
            if self.training:
                with torch.no_grad():
                    _, valid = self.rollout(z[:, -1], coefficients)
                ids = valid.nonzero().flatten()
                future, _ = self.rollout(z[ids, -1], coefficients[ids], check=False)
                return self.decode(future), self.decode(z[ids]), valid, kl[ids], mean, scale
            future, valid = self.rollout(z[:, -1], coefficients)
        else:
            z = encoded.squeeze(-1).tanh() * self.fluid
            future = self.forecaster(z.unsqueeze(-1)).squeeze(-1)
            valid = torch.ones(len(z), device=z.device, dtype=torch.bool)
        return self.decode(future), self.decode(z), valid, kl, mean, scale
