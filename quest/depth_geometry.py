"""Raw OpenCV camera rays/projection for the versioned Depth-only frontend."""

import torch


def pack_distortion(value):
    value = torch.as_tensor(value, dtype=torch.float32).flatten()
    if value.numel() not in (4, 5, 8) or not bool(torch.isfinite(value).all()):
        raise ValueError("Depth geometry supports finite OpenCV 4/5/8 coefficients, not fisheye/tilt")
    result = torch.zeros(8, dtype=torch.float32)
    result[:value.numel()] = value
    return result


def _expand(value, dimensions):
    while value.ndim < dimensions:
        value = value.unsqueeze(-1)
    return value


def distort_normalized(xy, coefficients):
    """Brown-Conrady/rational order: k1,k2,p1,p2,k3,k4,k5,k6."""
    d = coefficients
    while d.ndim < xy.ndim:
        d = d.unsqueeze(-2)
    x, y = xy.unbind(-1)
    k1, k2, p1, p2, k3, k4, k5, k6 = d.unbind(-1)
    r2 = x.square() + y.square()
    denominator = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
    safe = torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / safe
    tangential = torch.stack((2 * p1 * x * y + p2 * (r2 + 2 * x.square()),
                              p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y), -1)
    distorted = xy * radial.unsqueeze(-1) + tangential
    valid = (denominator > 1e-8) & (radial > 1e-8) & torch.isfinite(distorted).all(-1)
    return distorted, radial, tangential, valid


def raw_camera_rays(intrinsics, distortion, grid_size, image_size, iterations=12):
    """Feature-cell centers map to original resized JPEG pixels, align_corners=False."""
    h, w = grid_size
    ih, iw = image_size
    y, x = torch.meshgrid(torch.arange(h, device=intrinsics.device, dtype=torch.float32),
                          torch.arange(w, device=intrinsics.device, dtype=torch.float32), indexing="ij")
    u, v = (x.flatten() + .5) * iw / w - .5, (y.flatten() + .5) * ih / h - .5
    k = intrinsics.float()
    target = torch.stack(((u - k[..., 0, 2, None]) / k[..., 0, 0, None],
                          (v - k[..., 1, 2, None]) / k[..., 1, 1, None]), -1)
    xy = target.clone()
    for _ in range(iterations):
        _, radial, tangential, valid = distort_normalized(xy, distortion.float())
        candidate = (target - tangential) / torch.where(valid, radial, torch.ones_like(radial)).unsqueeze(-1)
        xy = torch.where(valid[..., None] & torch.isfinite(candidate), candidate, xy)
    recovered, _, _, valid = distort_normalized(xy, distortion.float())
    pixel_error = (recovered - target) * torch.stack((k[..., 0, 0], k[..., 1, 1]), -1).unsqueeze(-2)
    valid = valid & torch.isfinite(xy).all(-1) & (pixel_error.norm(dim=-1) <= .25)
    rays = torch.cat((xy, torch.ones_like(xy[..., :1])), -1)
    return torch.where(valid[..., None], rays, torch.zeros_like(rays)), valid


def project_raw(points, intrinsics, distortion, image_size):
    z = points[..., 2]
    safe_z = torch.where(z > 1e-6, z, torch.ones_like(z))
    xy, _, _, valid = distort_normalized(points[..., :2] / safe_z[..., None], distortion.float())
    k = intrinsics.float()
    u = xy[..., 0] * _expand(k[..., 0, 0], z.ndim) + _expand(k[..., 0, 2], z.ndim)
    v = xy[..., 1] * _expand(k[..., 1, 1], z.ndim) + _expand(k[..., 1, 2], z.ndim)
    h, w = image_size
    valid = valid & (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    grid = torch.stack((2 * (u + .5) / w - 1, 2 * (v + .5) / h - 1), -1)
    valid = valid & torch.isfinite(grid).all(-1)
    return torch.where(valid[..., None], grid, torch.full_like(grid, 2.)), valid


@torch.no_grad()
def sampling_geometry(intrinsics, extrinsics, distortion, camera_mask, grid_size,
                      image_size, depths, neighbors):
    """Static calibrated ray sweeps; no teacher tensors or global ego position."""
    b, n = intrinsics.shape[:2]
    if (n < 2 or intrinsics.shape != (b, n, 3, 3) or extrinsics.shape != (b, n, 4, 4)
            or distortion.shape != (b, n, 8) or camera_mask.shape != (b, n)
            or not 1 <= neighbors < n):
        raise ValueError("Invalid Depth camera/calibration/mask dimensions")
    k, transforms, distortion = intrinsics.float(), extrinsics.float(), distortion.float()
    if not all(bool(torch.isfinite(x).all()) for x in (k, transforms, distortion)):
        raise ValueError("Nonfinite Depth calibration")
    if (not bool((k[..., (0, 1), (0, 1)] > 0).all())
            or not torch.allclose(k[..., 2, :], k.new_tensor([0, 0, 1]).expand(b, n, 3), atol=1e-5)
            or not bool((k[..., 0, 1].abs() < 1e-6).all() & (k[..., 1, 0].abs() < 1e-6).all())):
        raise ValueError("Depth requires positive focal, zero-skew pinhole K before distortion")
    rotation = transforms[..., :3, :3]
    eye = torch.eye(3, device=k.device).expand(b, n, 3, 3)
    if (not torch.allclose(rotation.transpose(-1, -2) @ rotation, eye, atol=1e-4, rtol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation), k.new_ones(b, n), atol=1e-4)
            or not torch.allclose(transforms[..., 3, :], k.new_tensor([0, 0, 0, 1]).expand(b, n, 4), atol=1e-5)):
        raise ValueError("extrinsics must be rigid sensor2lidar, not lidar2cam")
    rays, ray_valid = raw_camera_rays(k, distortion, grid_size, image_size)
    forward = rotation[..., :, 2]
    similarity = forward @ forward.transpose(1, 2)
    available = camera_mask[:, :, None] & camera_mask[:, None, :] & ~torch.eye(n, device=k.device, dtype=torch.bool)[None]
    scores, source_ids = similarity.masked_fill(~available, -float("inf")).topk(neighbors, dim=-1)
    batch = torch.arange(b, device=k.device)[:, None]
    camera_points = rays[..., None, :] * depths.float()[None, None, None, :, None]
    lidar_points = torch.einsum("bnij,bnpdj->bnpdi", rotation, camera_points) + transforms[..., None, None, :3, 3]
    grids, masks = [], []
    inverse = torch.linalg.inv(transforms)
    for rank in range(neighbors):
        source = source_ids[..., rank]
        t = inverse[batch, source]
        xyz = torch.einsum("bnij,bnpdj->bnpdi", t[..., :3, :3], lidar_points) + t[..., None, None, :3, 3]
        grid, valid = project_raw(xyz, k[batch, source], distortion[batch, source], image_size)
        valid = valid & ray_valid[..., None] & torch.isfinite(scores[..., rank, None, None])
        grids.append(grid)
        masks.append(valid)
    world_rays = torch.einsum("bnij,bnpj->bnpi", rotation, rays)
    origin = transforms[..., :3, 3].unsqueeze(-2).expand_as(world_rays) / depths[-1]
    return source_ids, grids, masks, torch.cat((world_rays, origin), -1), ray_valid
