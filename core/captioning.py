"""Local captioning for character-sheet images (BLIP with templated fallback)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


def template_caption(trigger_word: str, index: int, total: int) -> str:
    poses = [
        "portrait photo",
        "looking at camera",
        "three-quarter view",
        "side profile",
        "candid expression",
        "smiling",
        "neutral expression",
        "full body standing",
        "close-up face",
        "natural lighting",
    ]
    pose = poses[index % len(poses)]
    return f"a photo of {trigger_word}, {pose}, high quality reference ({index + 1}/{total})"


class Captioner:
    """BLIP captioner with graceful fallback to templates."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipe = None
        self._failed = False

    def _ensure(self) -> bool:
        if self._pipe is not None:
            return True
        if self._failed:
            return False
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            model_id = "Salesforce/blip-image-captioning-base"
            self._processor = BlipProcessor.from_pretrained(model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(model_id)
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
            self._pipe = True
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[captioning] BLIP unavailable ({exc}); using templated captions.")
            self._failed = True
            return False

    def caption_image(self, image: Image.Image, trigger_word: str) -> str:
        if not self._ensure():
            return f"a photo of {trigger_word}"
        import torch

        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=40)
        raw = self._processor.decode(out[0], skip_special_tokens=True).strip()
        # Keep identity token front-and-center for LoRA training.
        if trigger_word.lower() not in raw.lower():
            return f"a photo of {trigger_word}, {raw}"
        return raw

    def caption_folder(
        self,
        images_dir: Path,
        trigger_word: str,
        *,
        use_blip: bool = True,
        overwrite: bool = False,
    ) -> list[tuple[Path, str]]:
        images = list_images(images_dir)
        results: list[tuple[Path, str]] = []
        for i, img_path in enumerate(images):
            cap_path = img_path.with_suffix(".txt")
            if cap_path.exists() and not overwrite:
                caption = cap_path.read_text(encoding="utf-8").strip()
            elif use_blip:
                with Image.open(img_path) as im:
                    caption = self.caption_image(im, trigger_word)
                cap_path.write_text(caption + "\n", encoding="utf-8")
            else:
                caption = template_caption(trigger_word, i, len(images))
                cap_path.write_text(caption + "\n", encoding="utf-8")
            results.append((img_path, caption))
        return results


def caption_scene(image: Image.Image) -> str:
    """Short scene description used to ground character generation in a background."""
    cap = Captioner()
    if not cap._ensure():
        return "cinematic scene matching the reference background lighting and color grade"
    import torch

    inputs = cap._processor(images=image.convert("RGB"), return_tensors="pt").to(cap.device)
    with torch.inference_mode():
        out = cap._model.generate(**inputs, max_new_tokens=32)
    return cap._processor.decode(out[0], skip_special_tokens=True).strip()
