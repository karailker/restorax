import math

import numpy as np


def to_rgb(frame: np.ndarray, src: str) -> np.ndarray:
    """Convert frame to RGB uint8 from a given color space string."""
    import cv2

    if src == "rgb":
        return frame
    if src == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported source color space: {src}")


def from_rgb(frame: np.ndarray, dst: str) -> np.ndarray:
    """Convert RGB uint8 frame to destination color space."""
    import cv2

    if dst == "rgb":
        return frame
    if dst == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported destination color space: {dst}")


def pad_to_multiple(
    frame: np.ndarray, multiple: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Pad frame height and width up to the nearest multiple.

    Returns (padded_frame, (pad_top, pad_bottom, pad_left, pad_right)).
    """
    h, w = frame.shape[:2]
    pad_h = math.ceil(h / multiple) * multiple - h
    pad_w = math.ceil(w / multiple) * multiple - w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    padded = np.pad(
        frame,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="reflect",
    )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(frame: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    """Remove padding applied by pad_to_multiple."""
    pad_top, pad_bottom, pad_left, pad_right = pads
    h, w = frame.shape[:2]
    return frame[
        pad_top : h - pad_bottom if pad_bottom else h, pad_left : w - pad_right if pad_right else w
    ]


def tile_frame(
    frame: np.ndarray,
    tile_size: int,
    overlap: int,
) -> tuple[list[tuple[np.ndarray, tuple[int, int, int, int]]], int, int]:
    """
    Split a frame into overlapping tiles.

    Returns (tiles, n_tiles_h, n_tiles_w) where each tile entry is
    (tile_array, (y_start, y_end, x_start, x_end)) in the original frame.
    """
    h, w = frame.shape[:2]
    step = tile_size - overlap
    tiles = []
    ys = list(range(0, h - overlap, step))
    xs = list(range(0, w - overlap, step))
    for y in ys:
        for x in xs:
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)
            tiles.append((frame[y1:y2, x1:x2].copy(), (y1, y2, x1, x2)))
    return tiles, len(ys), len(xs)


def merge_tiles(
    tiles: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    output_h: int,
    output_w: int,
    scale: int,
    gaussian_blend: bool = True,
) -> np.ndarray:
    """
    Merge upscaled tiles back into a full frame.

    When gaussian_blend=True (default), overlap regions are blended with
    a 2D Gaussian window — this eliminates visible seam artifacts that occur
    with simple averaging, especially on high-frequency textures at tile edges.
    """
    c = tiles[0][0].shape[2] if tiles[0][0].ndim == 3 else 1
    canvas = np.zeros((output_h, output_w, c), dtype=np.float32)
    weight = np.zeros((output_h, output_w, 1), dtype=np.float32)

    for tile_arr, (y1, y2, x1, x2) in tiles:
        oy1, oy2 = y1 * scale, y2 * scale
        ox1, ox2 = x1 * scale, x2 * scale
        th = oy2 - oy1
        tw = ox2 - ox1
        tile_crop = tile_arr[:th, :tw].astype(np.float32)

        w = _gaussian_window(th, tw) if gaussian_blend else np.ones((th, tw, 1), dtype=np.float32)

        canvas[oy1:oy2, ox1:ox2] += tile_crop * w
        weight[oy1:oy2, ox1:ox2] += w

    result = np.clip(canvas / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8)
    return result


def _gaussian_window(h: int, w: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """
    Generate a 2D Gaussian window for smooth tile blending.

    sigma_ratio controls how much of the tile is the soft-edge region.
    Lower = sharper edges; higher = more blending.
    """
    sigma_y = h * sigma_ratio
    sigma_x = w * sigma_ratio
    cy, cx = h / 2.0, w / 2.0
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    gy = np.exp(-0.5 * ((ys - cy) / sigma_y) ** 2)
    gx = np.exp(-0.5 * ((xs - cx) / sigma_x) ** 2)
    window = np.outer(gy, gx)[:, :, np.newaxis]  # H W 1
    return window.astype(np.float32)
