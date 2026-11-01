"""Calibrated lenses for the sim cameras: render an undistorted pinhole view, then warp it into the real lens.

Isaac renders ideal pinhole images (Isaac Lab exposes no camera model matching OpenCV's fisheye or
radial-tangential distortion). A camera with calibrated intrinsics is therefore rendered as a "virtual"
pinhole camera with the same pose, square pixels, the optical axis at the image centre and a field of view
covering every pixel ray of the real camera; each output pixel then samples the virtual image where its ray
lands (bilinear). The result has the real camera's focal lengths, principal point and distortion.

Intrinsics use OpenCV conventions (pixel centres at integer coordinates):
    {"model": "pinhole" | "fisheye", "K": 3x3, "D": [...], "size": [width, height], "render_scale": 1.0}
"pinhole" is OpenCV's radial-tangential model (5, 8 or more coefficients), "fisheye" its equidistant
(Kannala-Brandt) model with 4 coefficients. render_scale > 1 renders the virtual camera finer than the real
camera's centre resolution (less aliasing where the warp compresses the image, more render time).
"""

import math

import cv2
import numpy as np
import torch


def pixel_rays(intrinsics):
    """Undistorted normalized coordinates (x/z, y/z) of every output pixel centre: [height, width, 2]."""
    width, height = intrinsics["size"]
    K = np.asarray(intrinsics["K"], np.float64)
    D = np.asarray(intrinsics["D"], np.float64)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    if intrinsics["model"] == "fisheye":
        rays = cv2.fisheye.undistortPoints(pixels, K, D.reshape(4, 1), criteria=criteria)
        points = np.concatenate((rays, np.ones_like(rays[..., :1])), axis=-1)
        back, _ = cv2.fisheye.projectPoints(points, np.zeros(3), np.zeros(3), K, D.reshape(4, 1))
    else:
        # OpenCV 4 names the iterative variant undistortPointsIter; OpenCV 5 adds criteria to undistortPoints.
        undistort = getattr(cv2, "undistortPointsIter", None)
        identity = np.eye(3)
        rays = undistort(pixels, K, D, identity, identity, criteria) if undistort else cv2.undistortPoints(
            pixels, K, D, R=identity, P=identity, criteria=criteria)
        points = np.concatenate((rays, np.ones_like(rays[..., :1])), axis=-1)
        back, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)
    error = np.abs(back.reshape(-1, 2) - pixels.reshape(-1, 2)).max()
    if error > 0.01:
        raise ValueError(f"Lens model is not invertible over the image (round trip error {error:.3f} px)")
    return rays.reshape(height, width, 2)


def virtual_camera(intrinsics):
    """Square-pixel, centred pinhole camera covering all rays: (fx = fy, width, height)."""
    rays = pixel_rays(intrinsics)
    K = np.asarray(intrinsics["K"], np.float64)
    focal = float(intrinsics.get("render_scale", 1.0)) * max(K[0, 0], K[1, 1])
    half_x, half_y = np.abs(rays[..., 0]).max(), np.abs(rays[..., 1]).max()
    # +2 px margin so bilinear sampling of the outermost rays stays inside; even sizes keep the centre exact.
    width = 2 * math.ceil(focal * half_x + 1.0) + 2
    height = 2 * math.ceil(focal * half_y + 1.0) + 2
    return focal, width, height


def isaac_intrinsic_matrix(focal, width, height):
    """Row-major intrinsic matrix for PinholeCameraCfg.from_intrinsic_matrix: optical axis at the image centre
    (Isaac measures the principal point from the image edge, OpenCV from the first pixel's centre)."""
    return [focal, 0.0, width / 2, 0.0, focal, height / 2, 0.0, 0.0, 1.0]


class LensWarp:
    """Warps virtual pinhole renders [N, Hv, Wv, C] (uint8) into the calibrated lens [N, H, W, C]."""

    def __init__(self, intrinsics):
        self.focal, self.render_width, self.render_height = virtual_camera(intrinsics)
        rays = pixel_rays(intrinsics)
        # Virtual camera in OpenCV pixel coordinates: centre at ((W-1)/2, (H-1)/2).
        u = self.focal * rays[..., 0] + (self.render_width - 1) / 2
        v = self.focal * rays[..., 1] + (self.render_height - 1) / 2
        grid = np.stack((2 * u / (self.render_width - 1) - 1, 2 * v / (self.render_height - 1) - 1), axis=-1)
        self.grid = torch.from_numpy(grid.astype(np.float32))[None]
        self.width, self.height = intrinsics["size"]

    def __call__(self, images):
        if self.grid.device != images.device:
            self.grid = self.grid.to(images.device)
        if images.shape[1:3] != (self.render_height, self.render_width):
            raise ValueError(f"Expected {self.render_width}x{self.render_height} renders, got {tuple(images.shape)}")
        x = images.permute(0, 3, 1, 2).float()
        warped = torch.nn.functional.grid_sample(
            x, self.grid.expand(len(images), -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True,
        )
        return warped.round().clamp(0, 255).to(images.dtype).permute(0, 2, 3, 1).contiguous()
