from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Sequence
import torch
import gc
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor, AutoImageProcessor,
    CLIPModel, ResNetModel, EfficientNetModel,
    SamModel, Sam2Model,
)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

ImageInput = Union[Image.Image, Sequence[Image.Image]]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# modle_id : (model_cls, has_pooled)
MODELS = {
    "clip":         (CLIPModel,         True),
    "resnet":       (ResNetModel,       True),
    "efficientnet": (EfficientNetModel, True),
    "sam":          (SamModel,          False),
    "sam2":         (Sam2Model,         False),
}


@dataclass
class EncoderFeatures:
    spatial_emb: torch.Tensor                 # (B, C, H, W)
    pooled_emb: Optional[torch.Tensor]     # (B, D), native global 없으면 None

class ImageEncoderFeatureExtractor:
    def __init__(
        self, model_name: str, checkpoint: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name.lower()
        cls, self.has_pooled = MODELS[self.model_name]
        self.checkpoint = checkpoint
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype == torch.float16 and self.device.type != "cuda":
            dtype = None
        self.dtype = dtype
        kw = {"dtype": dtype} if dtype is not None else {}
        self.model = cls.from_pretrained(checkpoint, **kw).to(self.device).eval()
        try:
            self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(checkpoint)
    def _pixel_values(self, images: ImageInput) -> torch.Tensor:
        pv = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        return pv.to(self.dtype) if self.dtype is not None else pv

    @torch.no_grad()
    def extract(self, images: ImageInput) -> EncoderFeatures:
        return getattr(self, f"_extract_{self.model_name}")(images)

    def _extract_clip(self, images):
        pv = self._pixel_values(images)
        vout = self.model.vision_model(pixel_values=pv)
        spatials = vout.last_hidden_state[:, 1:, :]
        p = self.model.config.vision_config.patch_size             
        b, n, d = spatials.shape     
        h, w = pv.shape[-2] // p, pv.shape[-1] // p                                     # spatial : (B, (h*w)+1, 768) 
        if h * w != n:                                                                  #   ~~ exclude CLS token & reshape ~~
            h = w = int(round(n ** 0.5))                                                #      -->  (B, 768, h, w) : (B, 768, 14, 14)
        return EncoderFeatures(spatials.transpose(1, 2).reshape(b, d, h, w),              # pooled  : (B, 512)
            self.model.visual_projection(vout.pooler_output))

    def _extract_resnet(self, images):
        out = self.model(pixel_values=self._pixel_values(images))                       # spatial : (B, 2048, 7, 7)
        return EncoderFeatures(out.last_hidden_state, out.pooler_output.flatten(1))     # pooled  : (B, 2048)

    def _extract_efficientnet(self, images):
        out = self.model(pixel_values=self._pixel_values(images))                       # spatial : (B, 2560, 19, 19)
        return EncoderFeatures(out.last_hidden_state, out.pooler_output.flatten(1))     # pooled  : (B , 2560)

    def _extract_sam(self, images):
        return EncoderFeatures(
            self.model.get_image_embeddings(self._pixel_values(images)), None)          # spatial : (B,256,64,64)

    def _extract_sam2(self, images):
        vout = self.model.get_image_features(pixel_values=self._pixel_values(images))   # spatial : (h*w, B, 256)
        feat = vout.fpn_hidden_states[-1]                                               #           ~~ reshape ~~
        hw, b, c = feat.shape                                                           #      -->  (B, 256, h, w) : (B, 256, 64, 64)
        side = int(round(hw ** 0.5))
        return EncoderFeatures(feat.permute(1, 2, 0).reshape(b, c, side, side).contiguous(), None)


def encode_directory(
    model_name: str, image_dir: Union[str, Path], output_dir: Union[str, Path], checkpoint: str,
     *, batch_size: int = 32, device: Optional[str] = None, half: bool = True,
) -> None:
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images in {image_dir}")
    N = len(paths)

    names = [p.relative_to(image_dir).with_suffix("").as_posix() for p in paths]
    names_dir = Path(output_dir) / "names.txt"
    if names_dir.exists():
        if "\n".join(name for name in names) != names_dir.read_text(encoding='utf-8') :
            (Path(output_dir) / f"names_{model_name}.txt").write_text("\n".join(names), encoding="utf-8")
    else:
        names_dir.write_text("\n".join(names), encoding="utf-8")
    
    ext = ImageEncoderFeatureExtractor(model_name, checkpoint, device,
                                       torch.float16 if half else None)
    print(f"[START] {model_name} :")

    spatial_mm = pooled_mm = None
    for start in range(0, N, batch_size):
        batch = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        embs = ext.extract(images)
        end = start + len(batch)

        spatial_np = embs.spatial_emb.half().cpu().numpy()
        if spatial_mm is None:
            spatial_mm = np.lib.format.open_memmap(
                out_dir / "spatial.npy", mode="w+",
                dtype=np.float16, shape=(N, *spatial_np.shape[1:]))
        spatial_mm[start:end] = spatial_np

        if ext.has_pooled:
            pooled_np = embs.pooled_emb.half().cpu().numpy()
            if pooled_mm is None:
                pooled_mm = np.lib.format.open_memmap(
                    out_dir / "pooled.npy", mode="w+",
                    dtype=np.float16, shape=(N, pooled_np.shape[1]))
            pooled_mm[start:end] = pooled_np

        print(f"... {end:5}/{N}", end="\r")

    spatial_shape = tuple(spatial_mm.shape)
    pooled_shape = tuple(pooled_mm.shape) if ext.has_pooled else None
    spatial_mm.flush()
    pooled_mm.flush() if ext.has_pooled else None

    print(f"[FINISH] {N} embeddings\n"
        f"  L spatial : {spatial_shape}\n"
        f"  L pooled : {pooled_shape}\n\n")

    gc.collect()
    torch.cuda.empty_cache()