
from pathlib import Path
import math
import numpy as np
from PIL import Image

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert to uint8. If float and not in [0,1], do per-channel min-max."""
    x = np.asarray(arr)
    if x.dtype == np.uint8:
        return x
    if x.ndim == 2:
        x = x[..., None]
    x = x.astype(np.float32)
    vmin = np.nanmin(x)
    vmax = np.nanmax(x)
    if np.isfinite(vmin) and np.isfinite(vmax) and 0.0 <= vmin and vmax <= 1.0:
        y = x * 255.0
    else:
        y = np.empty_like(x)
        for c in range(x.shape[-1]):
            xc = x[..., c]
            cmin, cmax = np.nanmin(xc), np.nanmax(xc)
            if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
                yc = np.zeros_like(xc)
            else:
                yc = (xc - cmin) / (cmax - cmin) * 255.0
            y[..., c] = yc
    y = np.nan_to_num(y, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)
    return y

def save_image(image_numpy: np.ndarray, image_path: str, aspect_ratio: float = 1.0):
    """
    Save a numpy image to disk.
    - Supports HxW, HxWx1, HxWx3, HxWx4
    - If float not in [0,1], min–max normalize per channel to uint8
    - aspect_ratio scales width: new_w = round(w * aspect_ratio); height unchanged
    """
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)

    img = np.asarray(image_numpy)
    if img.ndim == 2:
        mode = "L"
        img_u8 = _to_uint8(img)[..., 0]
    elif img.ndim == 3 and img.shape[2] in (1, 3, 4):
        img_u8 = _to_uint8(img)
        mode = {1: "L", 3: "RGB", 4: "RGBA"}[img_u8.shape[2]]
        if img_u8.shape[2] == 1:
            img_u8 = img_u8[..., 0]
    else:
        raise ValueError(f"Unsupported shape {img.shape}. Use split-save for C not in (1,3,4).")

    H, W = img_u8.shape[:2]
    im = Image.fromarray(img_u8, mode=mode)

    if aspect_ratio != 1.0:
        new_w = max(1, int(round(W * aspect_ratio)))
        new_h = H
        im = im.resize((new_w, new_h), Image.BICUBIC)

    im.save(image_path)

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert to uint8. If float and not in [0,1], do per-channel min-max."""
    x = np.asarray(arr)
    if x.dtype == np.uint8:
        return x
    if x.ndim == 2:
        x = x[..., None]
    x = x.astype(np.float32)
    vmin = np.nanmin(x)
    vmax = np.nanmax(x)
    if np.isfinite(vmin) and np.isfinite(vmax) and 0.0 <= vmin and vmax <= 1.0:
        y = x * 255.0
    else:
        y = np.empty_like(x)
        for c in range(x.shape[-1]):
            xc = x[..., c]
            cmin, cmax = np.nanmin(xc), np.nanmax(xc)
            if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
                yc = np.zeros_like(xc)
            else:
                yc = (xc - cmin) / (cmax - cmin) * 255.0
            y[..., c] = yc
    y = np.nan_to_num(y, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)
    return y

def save_image(image_numpy: np.ndarray, image_path: str, aspect_ratio: float = 1.0):
    """
    Save a numpy image to disk.
    - Supports HxW, HxWx1, HxWx3, HxWx4
    - If float not in [0,1], min–max normalize per channel to uint8
    - aspect_ratio scales width: new_w = round(w * aspect_ratio); height unchanged
    """
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)

    img = np.asarray(image_numpy)
    if img.ndim == 2:
        mode = "L"
        img_u8 = _to_uint8(img)[..., 0]
    elif img.ndim == 3 and img.shape[2] in (1, 3, 4):
        img_u8 = _to_uint8(img)
        mode = {1: "L", 3: "RGB", 4: "RGBA"}[img_u8.shape[2]]
        if img_u8.shape[2] == 1:
            img_u8 = img_u8[..., 0]
    else:
        # raise ValueError(f"Unsupported shape {img.shape}. Use split-save for C not in (1,3,4).")
        print(f"Unsupported shape {img.shape}. Use split-save for C not in (1,3,4).")

    H, W = img_u8.shape[:2]
    im = Image.fromarray(img_u8, mode=mode)

    if aspect_ratio != 1.0:
        new_w = max(1, int(round(W * aspect_ratio)))
        new_h = H
        im = im.resize((new_w, new_h), Image.BICUBIC)

    im.save(image_path)

def save_multichannel(arr: np.ndarray, stem: str):
    """
    Save arrays with arbitrary channels.
    - C in {1,3,4}: one image
    - otherwise: split every 3 channels to one RGB image: stem_part{k}.png
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        save_image(a, f"{stem}.png")
        return [f"{stem}.png"]

    H, W, C = a.shape
    if C in (1, 3, 4):
        save_image(a, f"{stem}.png")
        return [f"{stem}.png"]

    paths = []
    n_parts = math.ceil(C / 3)
    for k in range(n_parts):
        c0, c1 = k * 3, min((k + 1) * 3, C)
        chunk = a[..., c0:c1]
        # pad to 3 channels for PNG
        if chunk.shape[2] == 1:
            chunk = np.repeat(chunk, 3, axis=2)
        elif chunk.shape[2] == 2:
            z = np.zeros((H, W, 1), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, z], axis=2)
        out = f"{stem}_part{k}.png"
        save_image(chunk, out)
        paths.append(out)
    return paths


def save_image2(img_np, out_path: str, jpg_quality: int = 90):
    arr = np.asarray(img_np)  # Open3D -> RGB uint8
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    fmt = Path(out_path).suffix.lower()

    if fmt in (".jpg", ".jpeg"):
        # 可能是 RGBA，需要转成 RGB 才能存 JPG
        if arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr, mode="RGBA").convert("RGB")
        else:
            img = Image.fromarray(arr, mode="RGB")
        img.save(out_path, quality=jpg_quality, subsampling="4:2:0", optimize=True)
    elif fmt == ".webp":
        # 体积更小，但编码略慢；若极致省空间可用
        if arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr, mode="RGBA")
        else:
            img = Image.fromarray(arr, mode="RGB")
        img.save(out_path, quality=jpg_quality, method=6)
    else:
        print(f"Unsupported format {fmt}. Use jpg/webp.")
