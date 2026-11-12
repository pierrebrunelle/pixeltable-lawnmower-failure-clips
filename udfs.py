"""Pure-Python UDFs for the mower failure-clip pipeline (no table imports)."""

from __future__ import annotations

import numpy as np
import PIL.Image

import pixeltable as pxt

# Color signatures of the failure moments (mean RGB of a keyframe, 0..255).
SKY_BLUE = (135, 206, 235)
HOSE_ORANGE = (255, 127, 0)
FENCE_BROWN = (139, 90, 43)
DARK_LUMA = 40.0


def _rgb(img: PIL.Image.Image) -> np.ndarray:
    return np.asarray(img.convert('RGB').resize((64, 48)), dtype=np.float32)


def _share(a: np.ndarray, color: tuple[int, int, int], tol: float = 45.0) -> float:
    d = np.linalg.norm(a - np.array(color, dtype=np.float32), axis=-1)
    return float((d < tol).mean())


@pxt.udf
def luma(img: PIL.Image.Image) -> float:
    """Mean brightness (0..255) of a frame."""
    return float(np.asarray(img.convert('L'), dtype=np.float32).mean())


@pxt.udf
def color_shares(img: pxt.Image) -> pxt.Json[{'sky': pxt.Float, 'hose': pxt.Float, 'fence': pxt.Float}]:
    """Share of pixels close to the sky / hose / fence colors."""
    a = _rgb(img)
    return {'sky': _share(a, SKY_BLUE), 'hose': _share(a, HOSE_ORANGE, 60.0), 'fence': _share(a, FENCE_BROWN)}


@pxt.udf
def failure_label(shares: dict, brightness: float) -> str:
    """Rule-of-thumb label for a clip; the CLIP index finds look-alikes the rules miss."""
    if brightness < DARK_LUMA:
        return 'stall_dark'
    if shares['sky'] > 0.25:
        return 'tip_slope'
    if shares['hose'] > 0.05:
        return 'hose_snag'
    if shares['fence'] > 0.15:
        return 'fence_bump'
    return 'normal'


@pxt.udf
def gray_thumb(img: PIL.Image.Image) -> PIL.Image.Image:
    """Single-channel (mode 'L') 96 px thumbnail for the review queue."""
    t = img.convert('L')
    t.thumbnail((96, 96))
    return t


@pxt.udf
def overlay_mask(img: PIL.Image.Image, label: str) -> PIL.Image.Image:
    """RGBA review overlay: failure clips get a translucent red tint."""
    base = img.convert('RGBA')
    if label == 'normal':
        return base
    tint = PIL.Image.new('RGBA', base.size, (255, 0, 0, 90))
    return PIL.Image.alpha_composite(base, tint)


@pxt.udf
def firmware_family(firmware: str | None) -> str | None:
    """'4.2.1' -> '4.x'."""
    if firmware is None:
        return None
    return firmware.split('.')[0] + '.x'
