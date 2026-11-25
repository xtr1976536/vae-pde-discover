"""Adapt the server's RealPDEBench models to a single latent field."""

import torch
from torch import nn

from ..common import network


class SingleFieldBaseline(nn.Module):
    def __init__(self, name, history, horizon, points, width, modes):
        super().__init__()
        self.encoder = network((128, 64, 32, 1), nn.Tanh)
        self.decoder = network((1, 32, 64, 128), nn.Tanh)
        if name == "unet":
            from .unet import Unet3d

            self.transverse_points = 4  # The two spatial downsampling stages require four cells.
            self.forecaster = Unet3d(dim=width, channels=1, out_channels=1,
                                     dim_mults=(1, 2, 4), in_time=history, out_time=horizon)
        else:
            from .fno import FNO3d

            self.transverse_points = 1
            self.forecaster = FNO3d(
                modes1=min(4, (history + 6) // 2), modes2=modes, modes3=4,
                n_layers=4, width=width, shape_in=(history, points, 1, 1),
                shape_out=(horizon, points, 1, 1),
            )

    def forward(self, history):
        field = self.encoder(history)
        volume = field.unsqueeze(-2).expand(-1, -1, -1, self.transverse_points, -1)
        forecast = self.forecaster(volume).mean(-2)
        return self.decoder(forecast), self.decoder(field)
