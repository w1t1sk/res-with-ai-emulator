"""Model definitions for the updated latent-HierSwin emulator."""

from .fuxi_ens_v3 import CircConv2d, FuXiForecastModelV3
from .fuxi_ens_v3_headonly_hierswin import HierarchicalSwinPerturbationModel
from .fuxi_ens_v3_latent_hierswin import (
    FuXiENSModelV3LatentHierSwin,
    FuXiForecastFeatureCoreV3,
)

__all__ = [
    "CircConv2d",
    "FuXiForecastModelV3",
    "FuXiForecastFeatureCoreV3",
    "HierarchicalSwinPerturbationModel",
    "FuXiENSModelV3LatentHierSwin",
]
