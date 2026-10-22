"""Loopback-only target enforcement for the Ollama adapter.

Builds on `contracts.config.validate_loopback_url` / `all_loopback` and closes the gaps those
leave (they are pure string/IP checks):

- `localhost` is accepted by name, so the name is resolved by us on EVERY request and all
  resolved addresses must be loopback (`all_loopback`). A hosts-file or DNS answer that mixes in a
  non-loopback address is rejected whole.
- The connection is then made to the already-validated IP literal (pinning), with the original
  authority in the `Host` header. The name is never resolved a second time by the HTTP stack, so
  the answer cannot change between check and connect (DNS rebinding / TOCTOU).
- The URL port is validated here (the contract helper does not look at it).
- Redirects are never followed (see `ollama.py`); proxies from the environment are ignored.

This is an application-level boundary, not OS-enforced network isolation.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from ..contracts.config import all_loopback, is_loopback_ip, validate_loopback_url
from .errors import ProviderError, ProviderErrorKind

Resolver = Callable[[str, int], Awaitable[Sequence[str]]]


async def system_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


@dataclass(frozen=True)
class LoopbackTarget:
    scheme: str
    host: str  # lowercase, no brackets
    port: int

    @classmethod
    def parse(cls, url: str) -> LoopbackTarget:
        try:
            validate_loopback_url(url)
            parts = urlsplit(url)
            port = parts.port  # raises ValueError for non-numeric / out-of-range ports
            host = parts.hostname
        except ValueError as exc:
            raise ProviderError(
                ProviderErrorKind.target_not_loopback,
                "provider target rejected: not a valid loopback origin",
                details={"reason": str(exc)[:200]},
            ) from exc
        if host is None:
            raise ProviderError(ProviderErrorKind.target_not_loopback, "provider target has no host")
        if port is not None and port == 0:
            raise ProviderError(
                ProviderErrorKind.target_not_loopback,
                "provider target rejected: port 0",
                details={"reason": "port 0"},
            )
        return cls(scheme=parts.scheme, host=host, port=port or _default_port(parts.scheme))

    @property
    def host_is_ip(self) -> bool:
        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            return False
        return True

    @property
    def authority(self) -> str:
        """Original authority, sent as the Host header (and TLS SNI name) after pinning."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return host if self.port == _default_port(self.scheme) else f"{host}:{self.port}"

    def url_for(self, ip: str, path: str) -> str:
        host = f"[{ip}]" if ":" in ip else ip
        return f"{self.scheme}://{host}:{self.port}{path}"

    async def resolve(self, resolver: Resolver) -> list[str]:
        """Loopback addresses to try in order; raises if any resolved address is not loopback."""
        if self.host_is_ip:
            addrs: list[str] = [self.host]
        else:
            try:
                addrs = [str(a) for a in await resolver(self.host, self.port)]
            except OSError as exc:
                raise ProviderError(
                    ProviderErrorKind.dns_failure,
                    "could not resolve the provider host",
                    details={"errno": getattr(exc, "errno", None)},
                ) from exc
            if not addrs:
                raise ProviderError(
                    ProviderErrorKind.dns_failure, "provider host resolved to no addresses"
                )
        if not all_loopback(addrs):
            raise ProviderError(
                ProviderErrorKind.target_not_loopback,
                "provider host resolved to a non-loopback address; refusing to connect",
                details={"non_loopback": [a for a in addrs if not is_loopback_ip(a)][:8]},
            )
        return addrs
