from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from upd_emulator.constants import INPUT_CHANNELS, PHYSICAL_CHANNELS
from upd_emulator.constants_v2 import LATENT_SHAPE, SWIN_SHIFT_SIZE, SWIN_WINDOW_SIZE
from upd_emulator.models.fuxi_ens_v3 import CircConv2d
from upd_emulator.models.fuxi_ens_v3_headonly_hierswin import HierarchicalSwinPerturbationModel
from upd_emulator.models.swin_v3 import SwinTransformerV3Stack





class FuXiForecastFeatureCoreV3(nn.Module):
    """V3 forecast core with explicit latent-feature access."""

    def __init__(
        self,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.2,
        latent_channels: int = 128,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.encoder = CircConv2d(INPUT_CHANNELS, self.latent_channels, kernel_size=3)
        self.backbone = SwinTransformerV3Stack(
            dim=self.latent_channels,
            depth=24,
            input_resolution=LATENT_SHAPE,
            num_heads=8,
            window_size=SWIN_WINDOW_SIZE,
            shift_size=SWIN_SHIFT_SIZE,
            mlp_ratio=4.0,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = CircConv2d(self.latent_channels, PHYSICAL_CHANNELS, kernel_size=3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def transform_encoded(self, encoded: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = encoded.shape
        tokens = encoded.flatten(2).transpose(1, 2)
        tokens = self.backbone(tokens)
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forecast_from_encoded(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decode(self.transform_encoded(encoded))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forecast_from_encoded(self.encode(x))


class LatentResidualPerturbationModel(HierarchicalSwinPerturbationModel):
    """Hierarchical Swin perturbation model with zero-initialized residual heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_dims: Sequence[int],
        stage_heads: Sequence[int],
        encoder_depths: Sequence[int],
        decoder_depths: Sequence[int],
        window_size: int,
        shift_size: int,
        drop_path_rate: float,
        use_checkpoint: bool,
        initial_logvar_bias: float = -4.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stage_dims=stage_dims,
            stage_heads=stage_heads,
            encoder_depths=encoder_depths,
            decoder_depths=decoder_depths,
            window_size=window_size,
            shift_size=shift_size,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )
        nn.init.zeros_(self.mu_head.conv.weight)
        if self.mu_head.conv.bias is not None:
            nn.init.zeros_(self.mu_head.conv.bias)
        nn.init.zeros_(self.logvar_head.conv.weight)
        if self.logvar_head.conv.bias is not None:
            nn.init.constant_(self.logvar_head.conv.bias, float(initial_logvar_bias))


class FuXiENSModelV3LatentHierSwin(nn.Module):
    """Frozen V3 deterministic core with latent-space hierarchical Swin perturbations."""

    def __init__(
        self,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.2,
        latent_channels: int = 128,
        perturb_stage_dims: Sequence[int] = (96, 128, 128, 160, 192),
        perturb_stage_heads: Sequence[int] = (8, 8, 8, 8, 8),
        perturb_encoder_depths: Sequence[int] = (1, 1, 2, 2, 2),
        perturb_decoder_depths: Sequence[int] = (1, 1, 1, 2),
        perturb_window_size: int = 8,
        perturb_shift_size: int = 4,
        perturb_drop_path_rate: float = 0.05,
        initial_logvar_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.perturb_stage_dims = tuple(int(v) for v in perturb_stage_dims)
        self.perturb_stage_heads = tuple(int(v) for v in perturb_stage_heads)
        self.perturb_encoder_depths = tuple(int(v) for v in perturb_encoder_depths)
        self.perturb_decoder_depths = tuple(int(v) for v in perturb_decoder_depths)
        self.perturb_window_size = int(perturb_window_size)
        self.perturb_shift_size = int(perturb_shift_size)
        self.perturb_drop_path_rate = float(perturb_drop_path_rate)
        self.initial_logvar_bias = float(initial_logvar_bias)

        self.forecast_model = FuXiForecastFeatureCoreV3(
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
            latent_channels=self.latent_channels,
        )
        self.prior_model = LatentResidualPerturbationModel(
            in_channels=self.latent_channels,
            out_channels=self.latent_channels,
            stage_dims=self.perturb_stage_dims,
            stage_heads=self.perturb_stage_heads,
            encoder_depths=self.perturb_encoder_depths,
            decoder_depths=self.perturb_decoder_depths,
            window_size=self.perturb_window_size,
            shift_size=self.perturb_shift_size,
            drop_path_rate=self.perturb_drop_path_rate,
            use_checkpoint=use_checkpoint,
            initial_logvar_bias=self.initial_logvar_bias,
        )
        self.posterior_model = LatentResidualPerturbationModel(
            in_channels=self.latent_channels * 2,
            out_channels=self.latent_channels,
            stage_dims=self.perturb_stage_dims,
            stage_heads=self.perturb_stage_heads,
            encoder_depths=self.perturb_encoder_depths,
            decoder_depths=self.perturb_decoder_depths,
            window_size=self.perturb_window_size,
            shift_size=self.perturb_shift_size,
            drop_path_rate=self.perturb_drop_path_rate,
            use_checkpoint=use_checkpoint,
            initial_logvar_bias=self.initial_logvar_bias,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def build_truth_window(window: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.cat([window[:, PHYSICAL_CHANNELS:, :, :], target], dim=1)

    def encode_current(self, window: torch.Tensor) -> torch.Tensor:
        return self.forecast_model.encode(window)

    def encode_truth_window(self, window: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        truth_window = self.build_truth_window(window, target)
        return self.forecast_model.encode(truth_window)

    def sample_prior(self, window: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        encoded = self.encode_current(window)
        if num_samples > 1:
            encoded = encoded.repeat_interleave(num_samples, dim=0)
        mu_p, logvar_p = self.prior_model(encoded)
        perturbation = self.reparameterize(mu_p, logvar_p)
        return self.forecast_model.forecast_from_encoded(encoded + perturbation)
