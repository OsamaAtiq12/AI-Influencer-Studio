"""Hosted Krea 2 API backend (optional fallback when local GPU is insufficient)."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .config import KREA_API_BASE, KREA_API_KEY


class KreaAPIError(RuntimeError):
    pass


class KreaAPIClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or KREA_API_KEY
        self.base_url = (base_url or KREA_API_BASE).rstrip("/")
        if not self.api_key:
            raise KreaAPIError(
                "KREA_API_KEY (or KREA_API_TOKEN) is required for --use-api. "
                "Create a token at https://www.krea.ai/app/api/tokens"
            )

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def upload_asset(self, image: Image.Image, description: str = "style reference") -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        resp = requests.post(
            f"{self.base_url}/assets",
            headers=self.headers,
            files={"file": ("reference.png", buf, "image/png")},
            data={"description": description},
            timeout=120,
        )
        if resp.status_code >= 400:
            raise KreaAPIError(f"Asset upload failed: {resp.status_code} {resp.text}")
        data = resp.json()
        url = data.get("image_url") or data.get("url")
        if not url:
            raise KreaAPIError(f"Unexpected asset response: {data}")
        return url

    @staticmethod
    def image_to_data_uri(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _poll_job(self, job_id: str, timeout_s: float = 300.0) -> dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout_s:
            resp = requests.get(
                f"{self.base_url}/jobs/{job_id}",
                headers=self.headers,
                timeout=60,
            )
            if resp.status_code >= 400:
                raise KreaAPIError(f"Job poll failed: {resp.status_code} {resp.text}")
            data = resp.json()
            status = (data.get("status") or data.get("state") or "").lower()
            if status in {"completed", "succeeded", "success", "done"}:
                return data
            if status in {"failed", "error", "cancelled"}:
                raise KreaAPIError(f"Job {job_id} failed: {data}")
            time.sleep(2.0)
        raise KreaAPIError(f"Timed out waiting for job {job_id}")

    def generate(
        self,
        prompt: str,
        *,
        background: Image.Image | None = None,
        style_strength: float = 0.6,
        resolution: str = "1K",
        creativity: str | float = "medium",
        style_id: str | None = None,
        lora_strength: float = 0.85,
        seed: int | None = None,
        aspect_ratio: str = "1:1",
        model: str = "krea-2/medium",
    ) -> Image.Image:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution if isinstance(resolution, str) else f"{resolution}",
            "creativity": _map_creativity(creativity),
        }
        if seed is not None:
            payload["seed"] = seed

        if background is not None:
            # Prefer data URI to avoid depending on asset persistence.
            payload["image_style_references"] = [
                {"url": self.image_to_data_uri(background), "strength": float(style_strength)}
            ]

        if style_id:
            payload["styles"] = [{"id": style_id, "strength": float(lora_strength)}]

        endpoint = f"{self.base_url}/generate/image/krea/{model}"
        resp = requests.post(
            endpoint,
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if resp.status_code >= 400:
            raise KreaAPIError(f"Generate failed: {resp.status_code} {resp.text}")
        data = resp.json()

        # Sync SDK-style responses may embed urls immediately.
        urls = _extract_urls(data)
        if not urls and "job_id" in data:
            job = self._poll_job(data["job_id"])
            urls = _extract_urls(job)
        if not urls:
            raise KreaAPIError(f"No image URL in API response: {data}")

        img_resp = requests.get(urls[0], timeout=120)
        img_resp.raise_for_status()
        return Image.open(io.BytesIO(img_resp.content)).convert("RGB")

    def train_style(
        self,
        name: str,
        image_paths: list[Path],
        *,
        trigger_word: str | None = None,
        timeout_s: float = 3600.0,
    ) -> str:
        """Optional hosted LoRA/style training. Returns style id."""
        files = []
        handles = []
        try:
            for p in image_paths:
                fh = open(p, "rb")
                handles.append(fh)
                files.append(("images", (p.name, fh, "image/png")))
            data = {"name": name}
            if trigger_word:
                data["trigger_word"] = trigger_word
            resp = requests.post(
                f"{self.base_url}/styles/train",
                headers=self.headers,
                files=files,
                data=data,
                timeout=120,
            )
            if resp.status_code >= 400:
                raise KreaAPIError(f"Style train failed: {resp.status_code} {resp.text}")
            body = resp.json()
            job_id = body.get("job_id") or body.get("id")
            if not job_id:
                # Some deployments return style id directly
                style_id = body.get("style_id") or body.get("id")
                if style_id:
                    return str(style_id)
                raise KreaAPIError(f"Unexpected train response: {body}")
            job = self._poll_job(str(job_id), timeout_s=timeout_s)
            style_id = (
                job.get("style_id")
                or (job.get("result") or {}).get("style_id")
                or (job.get("data") or {}).get("id")
            )
            if not style_id:
                raise KreaAPIError(f"Train finished but no style_id: {job}")
            return str(style_id)
        finally:
            for fh in handles:
                fh.close()


def _map_creativity(value: str | float) -> str:
    if isinstance(value, str):
        return value
    # Map 0..1 slider → API enum
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "medium"
    return "high"


def _extract_urls(data: dict[str, Any]) -> list[str]:
    if not isinstance(data, dict):
        return []
    for key in ("urls", "images"):
        val = data.get(key)
        if isinstance(val, list) and val:
            if isinstance(val[0], str):
                return val
            if isinstance(val[0], dict) and "url" in val[0]:
                return [v["url"] for v in val]
    nested = data.get("data") or data.get("result") or data.get("output") or {}
    if isinstance(nested, dict):
        return _extract_urls(nested)
    return []
