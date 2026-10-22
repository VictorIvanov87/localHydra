"""Runtime settings loaded from an optional validated JSON config file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .contracts.config import AppConfig, ConversationConfig, ProviderConfig, ServerConfig
from .contracts.models import ModelProfile
from .contracts.privacy import PrivacyConfig


class ConfigError(Exception):
    """Startup configuration is invalid; the process must exit non-zero, not guess."""


@dataclass(frozen=True)
class Settings:
    server: ServerConfig = field(default_factory=ServerConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    models: tuple[ModelProfile, ...] = ()
    # False when no config file was supplied: no models, nothing guessed or defaulted.
    configured: bool = False


def load_settings(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    try:
        cfg = AppConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid config {path}:\n{e}") from e
    return Settings(
        server=cfg.server,
        provider=cfg.provider,
        privacy=cfg.privacy,
        conversation=cfg.conversation,
        models=tuple(cfg.models),
        configured=True,
    )
