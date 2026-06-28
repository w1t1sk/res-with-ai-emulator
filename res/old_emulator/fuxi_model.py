import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_layers import swintransformerblock


class CircConv2d(nn.Module):
    """
    A Conv2d wrapper that applies:
      - CIRCULAR padding on the longitude axis (width, dim=-1) to correctly
        model the periodic East-West boundary of the Earth.
      - ZERO (reflect-safe) padding on the latitude axis (height, dim=-2)
        because the poles are hard boundaries, NOT periodic.
    
    This is a drop-in replacement for nn.Conv2d(kernel_size=3, padding=1).
    The weight/bias shapes are identical, so pre-trained weights load cleanly.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        assert kernel_size == 3, "CircConv2d is designed for kernel_size=3"
        # padding=0 because we handle padding manually before the conv
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=bias)

    def forward(self, x):
        # Step 1: Circular pad ONLY in the longitude/width dimension (left and right)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        # Step 2: Zero pad in the latitude/height dimension (top and bottom)
        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        return self.conv(x)


class fuxibase(nn.Module):
    def __init__(self, in_channels=28, out_channels=14, embed_dim=64, img_size=(64, 128)):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        # Use custom CircConv2d for the embedding projection
        self.embed = CircConv2d(in_channels, embed_dim)
        self.stage1 = self._make_layer(embed_dim, img_size)
        self.stage2 = self._make_layer(embed_dim, img_size)
        self.stage3 = self._make_layer(embed_dim, img_size)
        self.stage4 = self._make_layer(embed_dim, img_size)
        self.stage5 = self._make_layer(embed_dim, img_size)
        # 1x1 final projection: no spatial padding needed
        self.final = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def _make_layer(self, dim, input_resolution):
        return nn.Sequential(
            swintransformerblock(dim=dim, input_resolution=input_resolution, num_heads=4, window_size=8, shift_size=0),
            swintransformerblock(dim=dim, input_resolution=input_resolution, num_heads=4, window_size=8, shift_size=4)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_emb = self.embed(x)
        x_flat = x_emb.flatten(2).transpose(1, 2)
        x1 = self.stage1(x_flat)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3 + x2)
        x5 = self.stage5(x4 + x1)
        x_out = x5.transpose(1, 2).view(b, self.embed_dim, h, w)
        return self.final(x_out)


class PerturbationModel(nn.Module):
    def __init__(self, in_channels, out_channels=28):
        super().__init__()
        # All 3x3 convolutions use CircConv2d for longitude-only circular padding
        self.conv1 = CircConv2d(in_channels, 64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = CircConv2d(64, 64)
        self.relu2 = nn.ReLU(inplace=True)
        self.mu_head = CircConv2d(64, out_channels)
        self.logvar_head = CircConv2d(64, out_channels)

    def forward(self, x):
        features = self.relu2(self.conv2(self.relu1(self.conv1(x))))
        return self.mu_head(features), self.logvar_head(features)


class fuxiens(nn.Module):
    def __init__(self, forecast_model, model_p, model_q):
        super().__init__()
        self.forecast_model = forecast_model
        self.model_p = model_p
        self.model_q = model_q

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, y=None, num_samples=1):
        b, c, h, w = x.shape
        x_expanded = x.repeat(num_samples, 1, 1, 1)

        if y is not None:
            mu_p, logvar_p = self.model_p(x)
            mu_q, logvar_q = self.model_q(torch.cat([x, y], dim=1))
            mu_q_exp = mu_q.repeat(num_samples, 1, 1, 1)
            logvar_q_exp = logvar_q.repeat(num_samples, 1, 1, 1)
            z = self.reparameterize(mu_q_exp, logvar_q_exp)
            x_perturbed = x_expanded + z
            pred = self.forecast_model(x_perturbed)
            return pred, mu_p, logvar_p, mu_q, logvar_q
        else:
            mu_p, logvar_p = self.model_p(x)
            mu_p_exp = mu_p.repeat(num_samples, 1, 1, 1)
            logvar_p_exp = logvar_p.repeat(num_samples, 1, 1, 1)
            z = self.reparameterize(mu_p_exp, logvar_p_exp)
            x_perturbed = x_expanded + z
            return self.forecast_model(x_perturbed)
