# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Lifecycle wrapper for the SAM 3 image model."""

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image


ImageInput = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]


class SAM3ImageWrapper:
    """Load, call, and unload a SAM 3 image/text segmentation model.

    By default this class only uses ``sam3/ckpt/sam3.pt`` and never starts a
    Hugging Face download. Pass ``load_from_hf=True`` to opt into that fallback.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        load_from_hf: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir
            else Path(__file__).resolve().parents[1] / "ckpt"
        )
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.confidence_threshold = float(confidence_threshold)
        self.load_from_hf = bool(load_from_hf)
        self.model = None
        self.processor = None
        self._loaded_checkpoint_path = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def _resolve_checkpoint(self) -> Optional[Path]:
        if self.checkpoint_path is not None:
            candidate = self.checkpoint_path
            if candidate.is_dir():
                candidate = candidate / "sam3.pt"
        else:
            candidate = self.checkpoint_dir / "sam3.pt"

        if candidate.is_file():
            return candidate
        if self.load_from_hf:
            return None

        partial_files = sorted(
            path
            for path in candidate.parent.glob("*")
            if path.is_file()
            and path.name.endswith((".incomplete", ".part", ".tmp"))
        )
        partial_hint = ""
        if partial_files:
            partial_hint = " Partial files found: " + ", ".join(
                str(path) for path in partial_files
            )
        raise FileNotFoundError(
            f"SAM3 checkpoint was not found at {candidate}. "
            f"The download may still be incomplete.{partial_hint} "
            "Finish downloading sam3.pt or pass checkpoint_path explicitly."
        )

    def load(self) -> "SAM3ImageWrapper":
        """Load the model once and return ``self``."""
        if self.is_loaded:
            return self

        checkpoint_path = self._resolve_checkpoint()
        # Delay heavyweight imports until load() is called.
        from .model.sam3_image_processor import Sam3Processor
        from .model_builder import build_sam3_image_model

        try:
            self.model = build_sam3_image_model(
                checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                load_from_HF=self.load_from_hf,
                device=self.device,
                eval_mode=True,
            )
            self.processor = Sam3Processor(
                self.model,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
            self._loaded_checkpoint_path = checkpoint_path
        except Exception:
            self.unload()
            raise
        return self

    @staticmethod
    def _prepare_image(image: ImageInput) -> Any:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return opened.convert("RGB").copy()
        if isinstance(image, Image.Image) and image.mode != "RGB":
            return image.convert("RGB")
        return image

    @staticmethod
    def _to_cpu(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: SAM3ImageWrapper._to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [SAM3ImageWrapper._to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(SAM3ImageWrapper._to_cpu(item) for item in value)
        return value

    @torch.inference_mode()
    def predict(
        self,
        image: ImageInput,
        text_prompt: str,
        *,
        return_cpu: bool = True,
        return_state: bool = False,
    ) -> Dict[str, Any]:
        """Run one text prompt and return masks, boxes, and scores."""
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")

        self.load()
        state = self.processor.set_image(self._prepare_image(image))
        state = self.processor.set_text_prompt(text_prompt, state)
        result: Dict[str, Any] = {
            key: state[key]
            for key in ("masks", "boxes", "scores", "masks_logits")
            if key in state
        }
        if return_state:
            result["state"] = state
        return self._to_cpu(result) if return_cpu else result

    __call__ = predict

    def unload(self) -> None:
        """Release model references and clear CUDA's cached allocations."""
        model = self.model
        self.model = None
        self.processor = None
        self._loaded_checkpoint_path = None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "SAM3ImageWrapper":
        return self.load()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.unload()

    def __del__(self) -> None:
        try:
            self.unload()
        except Exception:
            pass

