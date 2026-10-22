"""Combine configured profiles with what the provider reports, using the frozen contract.

Read-only: lists installed and loaded models. It cannot trigger a download.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts.models import ModelProfile, ModelStatus, ProbeStatus, resolve_model_statuses
from .errors import ProviderError, ProviderErrorKind
from .provider import ModelProvider

# A failure to learn the inventory is "unreachable" (the contract has no other failed-probe
# state). A rejected target is a configuration error and is never swallowed.
_PROPAGATE = {ProviderErrorKind.target_not_loopback, ProviderErrorKind.closed}


@dataclass(frozen=True)
class ModelProbe:
    probe_status: ProbeStatus
    statuses: list[ModelStatus]
    error: ProviderError | None = None
    loaded_state_error: ProviderError | None = None
    # Installed tags that Ollama proxies to a remote host (cloud). They are NOT usable for local
    # inference and are reported in `statuses` as unavailable/not_installed, because the frozen
    # contract has no 'remote' reason. Consumers must show/handle them via this field.
    remote_tags: frozenset[str] = frozenset()


async def probe_models(provider: ModelProvider, profiles: Sequence[ModelProfile]) -> ModelProbe:
    try:
        installed = await provider.list_models()
    except ProviderError as err:
        if err.kind in _PROPAGATE:
            raise
        return ModelProbe(
            probe_status=ProbeStatus.unreachable,
            statuses=resolve_model_statuses(profiles, None),
            error=err,
        )
    loaded_tags: frozenset[str] | None = None
    loaded_error: ProviderError | None = None
    try:
        loaded = await provider.list_loaded()
        loaded_tags = loaded.tags  # None when the server has no loaded-state endpoint
    except ProviderError as err:
        if err.kind in _PROPAGATE:
            raise
        loaded_error = err
    return ModelProbe(
        probe_status=ProbeStatus.ok,
        statuses=resolve_model_statuses(
            profiles, installed.local_tags, loaded_tags, observed_at=installed.observed_at
        ),
        loaded_state_error=loaded_error,
        remote_tags=installed.remote_tags,
    )
