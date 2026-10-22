"""Validated application configuration (JSON file passed via --config / LOCALHYDRA_CONFIG).

Local-only boundary (an application rule, not OS-enforced network isolation):
- The backend binds to a loopback IP literal only.
- The provider base URL must be http(s) with a loopback host (127.0.0.0/8, ::1 or `localhost`),
  no embedded credentials, and redirects are never followed. LH-002 must additionally check every
  *resolved* address with `all_loopback()` before connecting, and re-check on each redirect hop
  (there are none to follow in v1, so a redirect response is an error).
- Models: 4-5 exact `name:tag` profiles; missing models are reported unavailable and are never
  pulled. There is deliberately no field anywhere that enables downloads.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from .common import Contract
from .models import ModelConfig
from .privacy import PrivacyConfig
from .runs import DEFAULT_MAX_CONVERSATION_TURNS_IN_MEMORY

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def is_loopback_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value.strip("[]")).is_loopback
    except ValueError:
        return False


def all_loopback(resolved_addresses: Iterable[str]) -> bool:
    """True only if at least one address was resolved and every one is loopback."""
    addrs = list(resolved_addresses)
    return bool(addrs) and all(is_loopback_ip(a) for a in addrs)


def validate_loopback_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("provider URL scheme must be http or https")
    if parts.username or parts.password:
        raise ValueError("provider URL must not embed credentials")
    host = parts.hostname
    if host is None:
        raise ValueError("provider URL has no host")
    if host != "localhost" and not is_loopback_ip(host):
        raise ValueError(f"provider host {host!r} is not loopback; only local Ollama is allowed")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("provider URL must be a bare origin (no path, query or fragment)")
    return url.rstrip("/")


class ServerConfig(Contract):
    host: str = DEFAULT_HOST
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def _loopback_literal(cls, v: str) -> str:
        if not is_loopback_ip(v):
            raise ValueError("server.host must be a loopback IP literal (e.g. 127.0.0.1 or ::1)")
        return v


class ProviderConfig(Contract):
    kind: Literal["ollama"] = "ollama"
    base_url: str = DEFAULT_OLLAMA_URL
    follow_redirects: Literal[False] = False

    @field_validator("base_url")
    @classmethod
    def _loopback(cls, v: str) -> str:
        return validate_loopback_url(v)


class ConversationConfig(Contract):
    max_turns_in_memory: int = Field(default=DEFAULT_MAX_CONVERSATION_TURNS_IN_MEMORY, ge=1, le=1000)


class AppConfig(ModelConfig):
    """On-disk config. Inherits the strict 4-5 unique exact-tag `models` list."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
