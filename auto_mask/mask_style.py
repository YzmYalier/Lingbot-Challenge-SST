"""How a kept-region mask is drawn onto camera frames (shared by pipeline_lerobot and composite_lerobot).

dim      pixels outside the mask x ``outside_scale`` (the original SAM datasets: 0.2)
outline  pixels inside the mask unchanged, a ``edge_width``-px ring just OUTSIDE the mask set
         to ``edge_color`` (white), the remaining outside pixels x ``outside_scale``

The ring is ``dilate(mask, ellipse(2w+1)) & ~mask``, so mask pixels are never altered.
The outside scaling is float multiply + truncation, identical to the original ``dim``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import cv2
import numpy as np

STYLES = ("dim", "outline")


@dataclass(frozen=True)
class MaskStyle:
    name: str = "dim"
    outside_scale: float = 0.2
    edge_width: int = 2
    edge_color: tuple = (255, 255, 255)

    def __post_init__(self):
        if self.name not in STYLES:
            raise ValueError(f"unknown style {self.name!r}; expected one of {STYLES}")
        if not 0.0 <= self.outside_scale <= 1.0:
            raise ValueError("outside_scale must be in [0, 1]")
        if self.name == "outline" and self.edge_width < 1:
            raise ValueError("edge_width must be >= 1 for the outline style")

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.name == "dim":  # edge settings do not apply
            d.pop("edge_width")
            d.pop("edge_color")
        else:
            d["edge_color"] = list(self.edge_color)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MaskStyle":
        d = dict(d)
        if "edge_color" in d:
            d["edge_color"] = tuple(int(c) for c in d["edge_color"])
        return cls(**d)

    def tag(self) -> str:
        """Short, filesystem-safe id, e.g. ``dim0.2`` or ``outline0.1_w2``."""
        s = f"{self.name}{self.outside_scale:g}"
        if self.name == "outline":
            s += f"_w{self.edge_width}"
            if self.edge_color != (255, 255, 255):
                s += "_c" + "-".join(map(str, self.edge_color))
        return s

    def ring(self, keep: np.ndarray) -> np.ndarray:
        """(T,H,W) bool: the outline pixels (outside the mask, within edge_width of it)."""
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.edge_width + 1,) * 2)
        out = np.empty_like(keep)
        for t in range(len(keep)):
            out[t] = cv2.dilate(keep[t].view(np.uint8), k).view(bool)
        out &= ~keep
        return out

    def render(self, frames: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """frames (T,H,W,3) uint8 RGB, keep (T,H,W) bool -> styled copy."""
        if keep.shape != frames.shape[:3] or keep.dtype != bool:
            raise ValueError(f"mask {keep.dtype} {keep.shape} vs frames {frames.shape}")
        out = frames.copy()
        outside = ~keep
        out[outside] = np.clip(out[outside].astype(np.float32) * self.outside_scale, 0, 255).astype(np.uint8)
        if self.name == "outline":
            out[self.ring(keep)] = np.asarray(self.edge_color, dtype=np.uint8)
        return out


def add_style_args(ap: argparse.ArgumentParser, default_style: str = "dim", default_scale: float = 0.2,
                   default_width: int = 2) -> None:
    ap.add_argument("--style", choices=STYLES, default=default_style,
                    help="dim: outside x scale; outline: white ring outside the mask + outside x scale")
    ap.add_argument("--outside-scale", type=float, default=default_scale)
    ap.add_argument("--edge-width", type=int, default=default_width, help="outline ring width in px")
    ap.add_argument("--edge-color", default="255,255,255", help="outline RGB")


def style_from_args(args) -> MaskStyle:
    color = tuple(int(c) for c in args.edge_color.split(","))
    if len(color) != 3 or not all(0 <= c <= 255 for c in color):
        raise SystemExit(f"bad --edge-color {args.edge_color!r}")
    return MaskStyle(args.style, float(args.outside_scale), int(args.edge_width), color)
