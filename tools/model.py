from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCM(nn.Module):
    # Spatial Compression Module
    def __init__(self, in_dim: int, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),                   # (C, H, W) -> (128, H, W)
            nn.Conv2d(128, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # (128, H, W) -> (64, H//2, W//2)
            nn.Conv2d(64, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),    # (64, H//2, W//2) -> (32, H//4, W//4)
            nn.Conv2d(32, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),    # (32, H//4, W//4) -> (16, H//8, W//8)
            nn.AdaptiveAvgPool2d(out_dim),                                                           # (16, H//8, W//8) -> (16,8,8)
            nn.Flatten(),                                                                            # (1024,)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PH(nn.Module):
    # Projection Head
    def __init__(self, in_dim=1024, hid_dim=512, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.ReLU(inplace=True), # (1024,) -> (512,)
            nn.Linear(hid_dim, out_dim), # (512,) -> (128,)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), p=2, dim=1) # L2 normalization : mlp(x) / ||mlp(x)||

class TLs(nn.Module):
    # Trainable Layers = SCM + PH
    def __init__(self, fm_dim: int):
        super().__init__()
        self.scm = SCM(in_dim=fm_dim)
        self.ph = PH()
    def forward(self, x: torch.Tensor, isTrain : bool = False) -> torch.Tensor:
        h = self.scm(x)
        return self.ph(h) if isTrain else h

# Loss : NT-Xent Loss(normalized temperature-scaled cross entropy loss)
# self-supervised learning, contrastive learning, metric learning : SimCLR(2020)
def nt_xent_loss(e_a, e_b, temperature: float = 0.05):
    # e_a, e_b: (B, 128) L2 normalized embeddings in positive pair relation
    B   = e_a.shape[0]
    z   = torch.cat([e_a, e_b], dim=0)          # (2B, d)
    sim = torch.mm(z, z.T) / temperature         # (2B, 2B)

    mask = torch.eye(2*B, dtype=bool).to(z.device)
    sim.masked_fill_(mask, float('-inf'))

    labels = torch.cat([
        torch.arange(B, 2*B),
        torch.arange(0,   B)
    ]).to(z.device)
    return F.cross_entropy(sim, labels)