"""Variational PDE coefficient prediction from indirect observations."""

import math

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
import torch
from torch import nn

from .common import network

CASES = {
    "wave": ("Wave_equation/wave.mat", "u", "t", "second"),
    "convection": ("Convection_diffusion_equation/data.mat", "u", "t", "first"),
    "kg": ("KG_equation/KG_Exp.mat", "usol", "t", "second"),
    "burgers": ("Burgers_equation/burgers_sine.mat", "usol", "t", "periodic"),
    "mixed": ("Eq_6_2_12/data_Eq_6_2_12.csv", None, None, "first"),
    "divide": ("PDE_divide/PDE_divide.npy", None, None, "first"),
    "kdv": ("KdV_equation/KdV-PINN.mat", "uu", "tt", "periodic"),
    "chaffee": ("Chaffee_Infante_equation/CI.npy", None, None, "first"),
}
def load_trajectory(root, case, points, observation_seed):
    path, state, time, mode = CASES[case]
    path = root / path
    if path.suffix == ".mat":
        data = loadmat(path)
        x, t, u = data["x"].ravel(), data[time].ravel(), data[state].real
        # The upstream loader uses time-major storage only for convection.
        if case != "convection":
            u = u.T
    elif case == "mixed":
        u = np.loadtxt(path, delimiter=",")
        x, t = np.arange(501) * .01, np.arange(501) * .02
    elif case == "divide":
        u = np.load(path)
        x, t = np.linspace(1, 2, 100), np.linspace(0, 1, 251)
    else:
        u = np.load(path).T
        x, t = [np.load(path.with_name(f"{axis}.npy")).ravel() for axis in ("x", "t")]
    if u.shape != (len(t), len(x)) or not np.isfinite(u).all():
        raise ValueError(f"{case}: invalid trajectory shape or values")
    if not np.allclose(np.diff(t), t[1] - t[0]):
        raise ValueError(f"{case}: nonuniform timestamps")
    if mode == "periodic":
        period = (x[1] - x[0]) * len(x)
        u = resample(u, points, axis=1)
        x = x[0] + np.arange(points) * period / points
    else:
        grid = np.linspace(x[0], x[-1], points)
        u = np.stack([np.interp(grid, x, frame) for frame in u])
        x = grid
    n = len(t)
    train_end = n - 2 * max(20, n // 4)
    val_end = n - max(20, n // 4)
    truth = torch.tensor(u, dtype=torch.float32)
    a = truth.unsqueeze(-1)
    basis = torch.randn(4, 128, generator=torch.Generator().manual_seed(observation_seed))
    obs = a * basis[0] + a.square() * basis[1] + torch.sin(a * basis[2] + basis[3])
    mean, scale = obs[:train_end].mean((0, 1)), obs[:train_end].std((0, 1), unbiased=False)
    return (obs - mean) / scale.clamp_min(1e-5), x, float(t[1] - t[0]), mode, (0, train_end, val_end, n)


def derivative_matrices(x, periodic):
    n, dx = len(x), x[1] - x[0]
    if periodic:
        k = 2 * np.pi * np.fft.rfftfreq(n, dx)
        transform = np.fft.rfft(np.eye(n), axis=0)
        matrices = []
        for order in (1, 2, 3):
            multiplier = (1j * k) ** order
            if n % 2 == 0 and order % 2:
                multiplier[-1] = 0
            matrices.append(np.fft.irfft(multiplier[:, None] * transform, n=n, axis=0))
        return np.stack(matrices)
    matrices = np.zeros((3, n, n))
    for i in range(n):
        start = max(0, min(i - 2, n - 5))
        offsets = np.arange(start, start + 5) - i
        vandermonde = np.stack([offsets ** j for j in range(5)])
        for order in (1, 2, 3):
            rhs = np.eye(5)[order] * math.factorial(order) / dx ** order
            matrices[order - 1, i, start:start + 5] = np.linalg.solve(vandermonde, rhs)
    return matrices


class Solver(nn.Module):
    def __init__(self, x, dt, mode, library, substeps=2):
        super().__init__()
        derivatives = derivative_matrices(x, mode == "periodic")
        scales = np.maximum(1, np.linalg.norm(derivatives, ord=np.inf, axis=(1, 2)))
        self.register_buffer("derivatives", torch.tensor(derivatives / scales[:, None, None], dtype=torch.float32))
        self.register_buffer("eye", torch.eye(len(x)))
        self.dt = dt
        self.substeps = substeps
        self.mode = mode
        self.library = tuple(library)
        self.linear_ids = [j for j, (power, order) in enumerate(library)
                           if (power == 1 and order == 0) or power == 0]
        self.nonlinear_ids = [j for j in range(len(library)) if j not in self.linear_ids]
        mats = [self.eye if library[j][1] == 0 else self.derivatives[library[j][1] - 1]
                for j in self.linear_ids]
        self.register_buffer("linear", torch.stack(mats))
        self.register_buffer("unit_factors", torch.tensor([1. if order == 0 else scales[order - 1]
                                                           for _, order in library], dtype=torch.float32))
        self.nonlinear_terms = tuple(library[j] for j in self.nonlinear_ids)
        self.nonlinear_orders = max(order for _, order in self.nonlinear_terms)

    @property
    def coefficient_divisors(self):
        return self.unit_factors * self.dt ** (2 if self.mode == "second" else 1)

    def features(self, z, terms, orders):
        powers = (torch.ones_like(z), z, z.square(), z.pow(3))
        spatial = torch.einsum("...i,kji->...kj", z, self.derivatives[:orders]) if orders else None
        return torch.stack([powers[power] * spatial[..., order - 1, :] if order
                            else powers[power] for power, order in terms], dim=-1)

    def contributions(self, z, coefficients):
        features = self.features(z, self.library, 3)
        return features * (coefficients / self.dt ** (2 if self.mode == "second" else 1))[:, None, None, :]

    def rollout(self, z, velocity, coefficients, steps, check=False):
        # Time is measured in snapshot intervals during integration.
        c = coefficients[:, :len(self.library)]
        linear = torch.einsum("bk,kij->bij", c[:, self.linear_ids], self.linear)
        mass = self.eye.expand(len(z), -1, -1)
        h = 1 / self.substeps
        if self.mode == "second":
            implicit = mass - h * h * linear
        else:
            implicit = mass - h * linear
        if check:
            finite = torch.isfinite(implicit).all((1, 2))
            implicit = torch.where(finite[:, None, None], implicit, self.eye)
            lu, pivots, info = torch.linalg.lu_factor_ex(implicit)
            valid = finite & (info == 0) & torch.isfinite(lu).all((1, 2))
            # Keep failed systems out of the numerical solve.
            lu = torch.where(valid[:, None, None], lu, self.eye)
            identity_pivots = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=pivots.dtype)
            pivots = torch.where(valid[:, None], pivots, identity_pivots)
            c = torch.where(valid[:, None], c, 0.)
            linear = torch.where(valid[:, None, None], linear, 0.)
            mass = torch.where(valid[:, None, None], mass, self.eye)
        else:
            lu, pivots = torch.linalg.lu_factor_ex(implicit)[:2]
        frames = []
        for _ in range(steps):
            for _ in range(self.substeps):
                columns = self.features(z, self.nonlinear_terms, self.nonlinear_orders)
                nl = (columns * c[:, None, self.nonlinear_ids]).sum(-1)
                if self.mode == "second":
                    right = velocity + h * ((linear @ z.unsqueeze(-1)).squeeze(-1) + nl)
                    velocity = torch.linalg.lu_solve(lu, pivots, right.unsqueeze(-1)).squeeze(-1)
                    z = z + h * velocity
                else:
                    right = (mass @ z.unsqueeze(-1)).squeeze(-1) + h * nl
                    z = torch.linalg.lu_solve(lu, pivots, right.unsqueeze(-1)).squeeze(-1)
                if check:
                    # States outside this range cannot be evaluated reliably in float32.
                    valid = valid & (z.abs().amax(1) <= 1e6)
                    if velocity is not None:
                        valid = valid & (velocity.abs().amax(1) <= 1e6)
                        velocity = torch.where(valid[:, None], velocity, 0.)
                    z = torch.where(valid[:, None], z, 0.)
            frames.append(z)
        frames = torch.stack(frames, dim=1)
        return (frames, valid) if check else frames


def sample(mean, logvar, stochastic):
    return mean + (0.5 * logvar).exp() * torch.randn_like(mean) if stochastic else mean


def kl(mean, logvar, prior_std=1.0):
    return .5 * ((mean.square() + logvar.exp()) / prior_std**2 - 1
                 + 2 * math.log(prior_std) - logvar).mean()


class VPCTransformer(nn.Module):
    def __init__(self, solver, width=32, layers=2, heads=4, dropout=0., coefficient_std=.003):
        super().__init__()
        self.solver = solver
        self.encoder = network((128, 64, 32, 2), nn.Tanh)
        self.decoder = network((1, 32, 64, 128), nn.Tanh)
        self.projection = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, 32, width) * .02)
        self.time_position = nn.Parameter(torch.randn(1, 20, 1, width) * .02)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        for block in self.transformer.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        count = len(solver.library)
        self.coefficients = nn.Linear(width, 2 * count)
        nn.init.normal_(self.coefficients.weight, std=.0003)
        nn.init.zeros_(self.coefficients.bias)
        self.coefficient_std = coefficient_std
        self.initial_velocity = network((2, 16, 1), nn.Tanh) if solver.mode == "second" else None

    def encode(self, x):
        mean, logvar = self.encoder(x).unbind(-1)
        logvar = (logvar - 4).clamp(-11, 1)
        return sample(mean, logvar, self.training).tanh(), mean, logvar

    def decode(self, z):
        return self.decoder(z.unsqueeze(-1))

    def prepare(self, history):
        z, mean, logvar = self.encode(history)
        b, h, n = z.shape
        summary = z.reshape(b, h, 32, n // 32).mean(-1)
        tokens = self.projection(summary.unsqueeze(-1)) + self.position + self.time_position[:, :h]
        hidden = self.transformer(tokens.flatten(1, 2)).mean(1)
        loc, logvar_c = self.coefficients(hidden).chunk(2, -1)
        loc = .01 * loc  # Scale optimizer steps in coefficient coordinates.
        scale = self.coefficient_std * nn.functional.softplus(.1 * logvar_c) / math.log(2)
        logvar_c = 2 * scale.clamp_min(torch.finfo(scale.dtype).tiny).log()
        coeff = loc + scale * torch.randn_like(loc) if self.training else loc
        initial = z[:, -1]
        velocity = None
        if self.initial_velocity is not None:
            center = mean.tanh()
            slope = center[:, -1] - center[:, -2] if h > 1 else torch.zeros_like(initial)
            velocity = slope + .025 * self.initial_velocity(torch.stack((center[:, -1], slope), -1)).squeeze(-1)
        return z, velocity, coeff, mean, logvar, loc, logvar_c


def windows(start, end, history, horizon, device):
    return torch.arange(start + history - 1, end - horizon, device=device)


def get_frames(data, origins, start, stop):
    offsets = torch.arange(start, stop, device=data.device)
    return data[origins[:, None] + offsets]


