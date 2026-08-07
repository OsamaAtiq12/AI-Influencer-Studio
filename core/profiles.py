"""Character profile helpers (profile.json + paths)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import character_dir, slugify


@dataclass
class CharacterProfile:
    slug: str
    name: str
    trigger_word: str
    created_at: str
    training: dict[str, Any] = field(default_factory=dict)
    sample_images: list[str] = field(default_factory=list)
    lora_path: str = "lora.safetensors"
    notes: str = ""

    @property
    def dir(self) -> Path:
        return character_dir(self.slug)

    @property
    def lora_file(self) -> Path:
        return self.dir / self.lora_path

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "profile.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, slug: str) -> "CharacterProfile":
        path = character_dir(slug) / "profile.json"
        if not path.exists():
            raise FileNotFoundError(f"No profile for slug '{slug}' at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def create(
        cls,
        name: str,
        slug: str | None = None,
        trigger_word: str | None = None,
        **training: Any,
    ) -> "CharacterProfile":
        s = slugify(slug or name)
        tw = trigger_word or f"ohwx {s.replace('-', ' ')}"
        return cls(
            slug=s,
            name=name,
            trigger_word=tw,
            created_at=datetime.now(timezone.utc).isoformat(),
            training=dict(training),
        )


def list_characters() -> list[CharacterProfile]:
    from .config import CHARACTERS_DIR

    CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
    profiles: list[CharacterProfile] = []
    for child in sorted(CHARACTERS_DIR.iterdir()):
        if child.is_dir() and (child / "profile.json").exists():
            profiles.append(CharacterProfile.load(child.name))
    return profiles
