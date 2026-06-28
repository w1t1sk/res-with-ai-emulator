from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as f

from upd_emulator.constants import GRID_SHAPE, INPUT_CHANNELS
from upd_emulator.models.fuxi_ens_v3 import FuXiForecastModelV3
from upd_emulator.models.swin_v3 import SwinTransformerV3Stack


def _compute_stage_resolutions(num_stages: int) -> tuple[tuple[int, int], ...]:
    if num_stages < 2:
        raise ValueError("Need at least two hierarchy stages.")

    height, width = GRID_SHAPE
    resolutions = [(height, width)]
    current_height, current_width = height, width
    for _ in range(1, num_stages):
        if current_height % 2 != 0 or current_width % 2 != 0:
            raise ValueError(
                f"Cannot build {num_stages}-stage hierarchy from GRID_SHAPE={GRID_SHAPE}; "
                "intermediate resolutions must stay divisible by 2."
            )
        current_height //= 2
        current_width //= 2
        resolutions.append((current_height, current_width))
    return tuple(resolutions)


class CircConv2d(nn.Module):
    """Conv2d with circular longitude padding and zero latitude padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = f.pad(x, (0, 0, self.pad, self.pad), mode="constant", value=0)
        return self.conv(x)


class SwinFeatureStage(nn.Module):
    """Apply a Swin stack to a 2D feature map and preserve its spatial size."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        drop_path_rate: float,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.backbone = SwinTransformerV3Stack(
            dim=dim,
            depth=depth,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=4.0,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.backbone(tokens)
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = CircConv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.act1 = nn.GELU()
        self.conv2 = CircConv2d(out_channels, out_channels, kernel_size=3)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        return self.act2(self.conv2(x))


class UpsampleFuseBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = CircConv2d(in_channels + skip_channels, out_channels, kernel_size=3)
        self.act1 = nn.GELU()
        self.conv2 = CircConv2d(out_channels, out_channels, kernel_size=3)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = f.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act1(self.conv1(x))
        return self.act2(self.conv2(x))


class HierarchicalSwinPerturbationModel(nn.Module):
    """A coarse-to-fine perturbation pyramid with Swin blocks at each scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = INPUT_CHANNELS,
        stage_dims: Sequence[int] = (24, 32, 48, 64, 96),
        stage_heads: Sequence[int] = (4, 4, 4, 4, 8),
        encoder_depths: Sequence[int] = (1, 1, 1, 2, 2),
        decoder_depths: Sequence[int] = (1, 1, 1, 1),
        window_size: int = 8,
        shift_size: int = 4,
        drop_path_rate: float = 0.05,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.stage_dims = tuple(int(v) for v in stage_dims)
        self.stage_heads = tuple(int(v) for v in stage_heads)
        self.encoder_depths = tuple(int(v) for v in encoder_depths)
        self.decoder_depths = tuple(int(v) for v in decoder_depths)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.drop_path_rate = float(drop_path_rate)

        num_stages = len(self.stage_dims)
        if len(self.stage_heads) != num_stages:
            raise ValueError("stage_heads must match the number of stage_dims.")
        if len(self.encoder_depths) != num_stages:
            raise ValueError("encoder_depths must match the number of stage_dims.")
        if len(self.decoder_depths) != num_stages - 1:
            raise ValueError("decoder_depths must have len(stage_dims) - 1 entries.")
        for dim, heads in zip(self.stage_dims, self.stage_heads, strict=True):
            if dim % heads != 0:
                raise ValueError(f"Stage dim {dim} must be divisible by its head count {heads}.")

        self.stage_resolutions = _compute_stage_resolutions(num_stages)

        self.stem = nn.Sequential(
            CircConv2d(in_channels, self.stage_dims[0], kernel_size=3),
            nn.GELU(),
            CircConv2d(self.stage_dims[0], self.stage_dims[0], kernel_size=3),
            nn.GELU(),
        )

        self.encoder_stages = nn.ModuleList(
            [
                SwinFeatureStage(
                    dim=dim,
                    input_resolution=resolution,
                    depth=depth,
                    num_heads=heads,
                    window_size=self.window_size,
                    shift_size=self.shift_size,
                    drop_path_rate=self.drop_path_rate,
                    use_checkpoint=use_checkpoint,
                )
                for dim, resolution, depth, heads in zip(
                    self.stage_dims,
                    self.stage_resolutions,
                    self.encoder_depths,
                    self.stage_heads,
                    strict=True,
                )
            ]
        )

        self.down_blocks = nn.ModuleList(
            [
                DownsampleBlock(self.stage_dims[idx], self.stage_dims[idx + 1])
                for idx in range(num_stages - 1)
            ]
        )

        reversed_dims = list(reversed(self.stage_dims))
        reversed_heads = list(reversed(self.stage_heads))
        reversed_resolutions = list(reversed(self.stage_resolutions))

        self.up_blocks = nn.ModuleList(
            [
                UpsampleFuseBlock(
                    in_channels=reversed_dims[idx],
                    skip_channels=reversed_dims[idx + 1],
                    out_channels=reversed_dims[idx + 1],
                )
                for idx in range(num_stages - 1)
            ]
        )

        self.decoder_stages = nn.ModuleList(
            [
                SwinFeatureStage(
                    dim=reversed_dims[idx + 1],
                    input_resolution=reversed_resolutions[idx + 1],
                    depth=self.decoder_depths[idx],
                    num_heads=reversed_heads[idx + 1],
                    window_size=self.window_size,
                    shift_size=self.shift_size,
                    drop_path_rate=self.drop_path_rate,
                    use_checkpoint=use_checkpoint,
                )
                for idx in range(num_stages - 1)
            ]
        )

        self.mu_head = CircConv2d(self.stage_dims[0], out_channels, kernel_size=3)
        self.logvar_head = CircConv2d(self.stage_dims[0], out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips: list[torch.Tensor] = []

        x = self.stem(x)
        x = self.encoder_stages[0](x)
        skips.append(x)

        for down_block, stage in zip(self.down_blocks, self.encoder_stages[1:], strict=True):
            x = down_block(x)
            x = stage(x)
            skips.append(x)

        x = skips[-1]
        reversed_skips = list(reversed(skips[:-1]))
        for up_block, decoder_stage, skip in zip(self.up_blocks, self.decoder_stages, reversed_skips, strict=True):
            x = up_block(x, skip)
            x = decoder_stage(x)

        return self.mu_head(x), self.logvar_head(x)


class FuXiENSModelV3HeadOnlyHierSwin(nn.Module):
    """Frozen V3 deterministic core with a hierarchical Swin perturbation pyramid."""

    def __init__(
        self,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.2,
        perturb_stage_dims: Sequence[int] = (24, 32, 48, 64, 96),
        perturb_stage_heads: Sequence[int] = (4, 4, 4, 4, 8),
        perturb_encoder_depths: Sequence[int] = (1, 1, 1, 2, 2),
        perturb_decoder_depths: Sequence[int] = (1, 1, 1, 1),
        perturb_window_size: int = 8,
        perturb_shift_size: int = 4,
        perturb_drop_path_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.perturb_stage_dims = tuple(int(v) for v in perturb_stage_dims)
        self.perturb_stage_heads = tuple(int(v) for v in perturb_stage_heads)
        self.perturb_encoder_depths = tuple(int(v) for v in perturb_encoder_depths)
        self.perturb_decoder_depths = tuple(int(v) for v in perturb_decoder_depths)
        self.perturb_window_size = int(perturb_window_size)
        self.perturb_shift_size = int(perturb_shift_size)
        self.perturb_drop_path_rate = float(perturb_drop_path_rate)

        self.forecast_model = FuXiForecastModelV3(
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
        )
        self.prior_model = HierarchicalSwinPerturbationModel(
            in_channels=INPUT_CHANNELS,
            out_channels=INPUT_CHANNELS,
            stage_dims=self.perturb_stage_dims,
            stage_heads=self.perturb_stage_heads,
            encoder_depths=self.perturb_encoder_depths,
            decoder_depths=self.perturb_decoder_depths,
            window_size=self.perturb_window_size,
            shift_size=self.perturb_shift_size,
            drop_path_rate=self.perturb_drop_path_rate,
            use_checkpoint=use_checkpoint,
        )
        self.posterior_model = HierarchicalSwinPerturbationModel(
            in_channels=INPUT_CHANNELS + (INPUT_CHANNELS // 2),
            out_channels=INPUT_CHANNELS,
            stage_dims=self.perturb_stage_dims,
            stage_heads=self.perturb_stage_heads,
            encoder_depths=self.perturb_encoder_depths,
            decoder_depths=self.perturb_decoder_depths,
            window_size=self.perturb_window_size,
            shift_size=self.perturb_shift_size,
            drop_path_rate=self.perturb_drop_path_rate,
            use_checkpoint=use_checkpoint,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def sample_prior(self, window: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if num_samples > 1:
            window = window.repeat_interleave(num_samples, dim=0)
        mu_p, logvar_p = self.prior_model(window)
        perturbation = self.reparameterize(mu_p, logvar_p)
        return self.forecast_model(window + perturbation)
