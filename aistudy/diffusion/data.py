"""Procedural "shapes" dataset: infinite, labelled, generated on the fly in torch.

Four classes (circle, square, triangle, cross) with random position, size, rotation, colour
and background tint, rendered with anti-aliased signed-distance functions. It is a stand-in
for ImageNet that needs no download and is hard enough to show the effect of classifier-free
guidance, sampler order and EMA. To use real data, replace ``ShapesDataset.sample`` with a
DataLoader over your images scaled to [-1, 1] (and, at scale, encode them to a latent space
with a VAE / RAE first - see docs/03-diffusion-models.md).
"""

from __future__ import annotations

import math

import torch

CLASSES = ["circle", "square", "triangle", "cross"]


def _box_sdf(px, py, bx, by):
    qx, qy = px.abs() - bx, py.abs() - by
    outside = torch.sqrt(qx.clamp_min(0) ** 2 + qy.clamp_min(0) ** 2)
    inside = torch.maximum(qx, qy).clamp_max(0)
    return outside + inside


def render_shapes(labels: torch.Tensor, img_size: int = 32, generator: torch.Generator | None = None) -> torch.Tensor:
    """labels [N] -> images [N, 3, H, W] in [-1, 1]."""
    n = labels.numel()

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    cx, cy = 0.3 + 0.4 * rand(n), 0.3 + 0.4 * rand(n)
    r = 0.14 + 0.12 * rand(n)
    theta = 2 * math.pi * rand(n)
    color = 0.45 + 0.55 * rand(n, 3)
    bg = 0.25 * rand(n, 3)

    coords = (torch.arange(img_size, dtype=torch.float32) + 0.5) / img_size
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    c, s = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    px, py = c * dx + s * dy, -s * dx + c * dy            # rotate into the shape frame
    rr = r[:, None, None]

    sdf_circle = torch.sqrt(px ** 2 + py ** 2) - rr
    sdf_square = _box_sdf(px, py, 0.8 * rr, 0.8 * rr)
    normals = [(-math.pi / 2), (math.pi / 6), (5 * math.pi / 6)]     # equilateral triangle edges
    sdf_tri = torch.stack([math.cos(a) * px + math.sin(a) * py for a in normals]).amax(0) - 0.6 * rr
    sdf_cross = torch.minimum(_box_sdf(px, py, rr, rr / 2.4), _box_sdf(px, py, rr / 2.4, rr))
    sdfs = torch.stack([sdf_circle, sdf_square, sdf_tri, sdf_cross], dim=1)  # [N, 4, H, W]
    sdf = sdfs.gather(1, labels.view(-1, 1, 1, 1).expand(-1, 1, img_size, img_size).cpu())

    alpha = torch.sigmoid(-sdf * img_size * 2.0)          # ~1 pixel anti-aliased edge
    img = bg[:, :, None, None] * (1 - alpha) + color[:, :, None, None] * alpha
    return img * 2 - 1


class ShapesDataset:
    num_classes = len(CLASSES)

    def __init__(self, img_size: int = 32):
        self.img_size = img_size

    def sample(self, batch_size: int, device, generator: torch.Generator | None = None):
        y = torch.randint(0, self.num_classes, (batch_size,), generator=generator)
        x = render_shapes(y, self.img_size, generator)
        return x.to(device), y.to(device)
