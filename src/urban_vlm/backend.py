"""Lazy dependency import. Hugging Face image/text logits without fine-tuning."""
from __future__ import annotations
from PIL import Image


class SiglipBackend:
    def __init__(self, model_id: str, device: str = "auto", revision: str | None = None, seed: int = 42):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError('Install inference dependencies: pip install -e ".[inference]"') from exc
        self.torch = torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        kwargs = {"revision": revision} if revision else {}
        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModel.from_pretrained(model_id, **kwargs).to(self.device).eval()
        self.model_id = model_id
        self.revision = revision
        self.commit = getattr(self.model.config, "_commit_hash", None)

    def predict_logits(self, image: Image.Image, texts: list[str]) -> list[float]:
        torch = self.torch
        # SigLIP 2 uses 64-token fixed padding; the processor handles RGB resize.
        inputs = self.processor(
            images=image, text=texts, padding="max_length", max_length=64,
            truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        logits = outputs.logits_per_image.detach().float().cpu().numpy()
        if logits.shape != (1, len(texts)):
            raise ValueError(f"Unexpected logits shape {logits.shape}")
        return logits[0].tolist()
