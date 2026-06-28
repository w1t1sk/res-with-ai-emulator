from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f

from upd_emulator.constants_v2 import (
    HELPER_CHANNELS,
    INPUT_CHANNELS,
    LATENT_SHAPE,
    PHYSICAL_CHANNELS,
    SWIN_SHIFT_SIZE,
    SWIN_WINDOW_SIZE,
)
from upd_emulator.models.swin_v3 import SwinTransformerV3Stack


def _validate_window_geometry() -> None:
    height, width = LATENT_SHAPE
    if height % SWIN_WINDOW_SIZE != 0 or width % SWIN_WINDOW_SIZE != 0:
        raise ValueError(
            f"LATENT_SHAPE={LATENT_SHAPE} must be divisible by "
            f"SWIN_WINDOW_SIZE={SWIN_WINDOW_SIZE}."
        )
    if not 0 <= SWIN_SHIFT_SIZE < SWIN_WINDOW_SIZE:
        raise ValueError(
            f"SWIN_SHIFT_SIZE={SWIN_SHIFT_SIZE} must be in [0, {SWIN_WINDOW_SIZE})."
        )


class CircConv2d(nn.Module):
    """conv2d with circular longitude padding and zero latitude padding"""

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        # circular pad on longitude (last dim), constant (zero) pad on latitude (second to last dim)
        x = f.pad(x, (1, 1, 0, 0), mode="circular")
        x = f.pad(x, (0, 0, 1, 1), mode="constant", value=0)
        return self.conv(x)


class FuXiForecastModelV3(nn.Module):
    """High-fidelity forecast model with a full-resolution latent grid."""

    def __init__(
        self,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        _validate_window_geometry()
        self.encoder = CircConv2d(INPUT_CHANNELS, 128, kernel_size=3)
        self.backbone = SwinTransformerV3Stack(
            dim=128,
            depth=24,
            input_resolution=LATENT_SHAPE,
            num_heads=8,
            window_size=SWIN_WINDOW_SIZE,
            shift_size=SWIN_SHIFT_SIZE,
            mlp_ratio=4.0,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = CircConv2d(128, PHYSICAL_CHANNELS, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        batch_size, channels, height, width = encoded.shape
        tokens = encoded.flatten(2).transpose(1, 2)
        tokens = self.backbone(tokens)
        decoded = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.decoder(decoded)


class FuXiPerturbationModelV3(nn.Module):
    """High-fidelity perturbation prior/posterior model with no spatial compression."""

    def __init__(
        self,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        _validate_window_geometry()
        self.meteor_encoder = CircConv2d(INPUT_CHANNELS, 64, kernel_size=3)
        self.helper_encoder = CircConv2d(HELPER_CHANNELS, 64, kernel_size=3)
        self.backbone = SwinTransformerV3Stack(
            dim=128,
            depth=8,
            input_resolution=LATENT_SHAPE,
            num_heads=8,
            window_size=SWIN_WINDOW_SIZE,
            shift_size=SWIN_SHIFT_SIZE,
            mlp_ratio=4.0,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = CircConv2d(128, INPUT_CHANNELS * 2, kernel_size=3)

    def forward(
        self,
        meteorology: torch.Tensor,
        helpers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_met = self.meteor_encoder(meteorology)
        encoded_helper = self.helper_encoder(helpers)
        encoded = torch.cat([encoded_met, encoded_helper], dim=1)
        batch_size, channels, height, width = encoded.shape
        tokens = encoded.flatten(2).transpose(1, 2)
        tokens = self.backbone(tokens)
        decoded = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        stats = self.decoder(decoded)
        return torch.chunk(stats, chunks=2, dim=1)


class FuXiENSModelV3(nn.Module):
    """Composite high-fidelity FuXi-ENS-style ensemble model."""

    def __init__(
        self,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.forecast_model = FuXiForecastModelV3(
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
        )
        self.prior_model = FuXiPerturbationModelV3(
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
        )
        self.posterior_model = FuXiPerturbationModelV3(
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def sample_prior(
        self,
        window: torch.Tensor,
        helpers: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")

        if num_samples > 1:
            window = window.repeat_interleave(num_samples, dim=0)
            helpers = helpers.repeat_interleave(num_samples, dim=0)

        mu_p, logvar_p = self.prior_model(window, helpers)
        perturbation = self.reparameterize(mu_p, logvar_p)
        return self.forecast_model(window + perturbation)
