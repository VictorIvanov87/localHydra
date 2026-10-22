"""Model profiles, validated model configuration and model availability.

Profiles are operator-provided. Nothing here names a real model, and capability numbers
are never defaults: they come from the operator's config or from later measurements.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from pydantic import Field, model_validator

from .common import Contract, ModelProfileId, OllamaTag, UtcDatetime

MIN_MODEL_PROFILES = 4
MAX_MODEL_PROFILES = 5


class TaskType(str, Enum):
    chat = "chat"
    code = "code"
    reasoning = "reasoning"
    summarization = "summarization"
    extraction = "extraction"


class Modality(str, Enum):
    text = "text"
    image = "image"


class SuitabilitySource(str, Enum):
    operator_configured = "operator_configured"
    measured = "measured"


class TaskSuitability(Contract):
    score: float = Field(ge=0.0, le=1.0)
    source: SuitabilitySource
    note: str | None = Field(default=None, max_length=500)


class LatencySource(str, Enum):
    unknown = "unknown"
    configured = "configured"
    measured = "measured"


class LatencyProfile(Contract):
    ttft_ms: float | None = Field(default=None, gt=0)
    tokens_per_second: float | None = Field(default=None, gt=0)
    source: LatencySource = LatencySource.unknown

    @model_validator(mode="after")
    def _source_matches_values(self) -> LatencyProfile:
        has_value = self.ttft_ms is not None or self.tokens_per_second is not None
        if self.source is LatencySource.unknown and has_value:
            raise ValueError("latency values require source 'configured' or 'measured'")
        if self.source is not LatencySource.unknown and not has_value:
            raise ValueError("latency source 'configured'/'measured' requires at least one value")
        return self


class ModelProfile(Contract):
    id: ModelProfileId
    tag: OllamaTag
    enabled: bool = True
    display_name: str | None = Field(default=None, max_length=100)
    context_window: int = Field(ge=512, le=10_000_000)
    max_output_tokens: int = Field(ge=1)
    input_modalities: list[Modality] = Field(default_factory=lambda: [Modality.text], min_length=1)
    supports_tools: bool = False
    task_suitability: dict[TaskType, TaskSuitability] = Field(default_factory=dict)
    latency: LatencyProfile = Field(default_factory=LatencyProfile)

    @model_validator(mode="after")
    def _consistent(self) -> ModelProfile:
        if self.max_output_tokens >= self.context_window:
            raise ValueError("max_output_tokens must be smaller than context_window")
        if Modality.text not in self.input_modalities:
            raise ValueError("input_modalities must include 'text'")
        if len(set(self.input_modalities)) != len(self.input_modalities):
            raise ValueError("input_modalities must not repeat")
        return self


class ModelConfig(Contract):
    """The `models` section of the application config: 4-5 exact, unique profiles."""

    models: list[ModelProfile] = Field(min_length=MIN_MODEL_PROFILES, max_length=MAX_MODEL_PROFILES)

    @model_validator(mode="after")
    def _unique(self) -> ModelConfig:
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError("model profile ids must be unique")
        tags = [m.tag.lower() for m in self.models]
        if len(set(tags)) != len(tags):
            raise ValueError("model tags must be unique (case-insensitive)")
        return self


class Availability(str, Enum):
    available = "available"
    unavailable = "unavailable"
    unknown = "unknown"


class UnavailableReason(str, Enum):
    not_installed = "not_installed"
    disabled = "disabled"
    provider_not_probed = "provider_not_probed"


class LoadedState(str, Enum):
    loaded = "loaded"
    not_loaded = "not_loaded"
    unknown = "unknown"


class ModelStatus(Contract):
    profile: ModelProfile
    availability: Availability
    availability_reason: UnavailableReason | None = None
    loaded_state: LoadedState = LoadedState.unknown
    # When loaded_state/availability were last observed from the provider; null if never.
    observed_at: UtcDatetime | None = None


class ProbeStatus(str, Enum):
    not_probed = "not_probed"
    ok = "ok"
    unreachable = "unreachable"


class ModelsResponse(Contract):
    configured: bool = Field(description="False when no model config was supplied.")
    provider_probe: ProbeStatus
    # Constant: the application never pulls/downloads models. Missing models stay unavailable.
    auto_download: bool = Field(default=False)
    models: list[ModelStatus]

    @model_validator(mode="after")
    def _never_auto_download(self) -> ModelsResponse:
        if self.auto_download:
            raise ValueError("auto_download is always false in the MVP")
        return self


def resolve_model_statuses(
    profiles: Iterable[ModelProfile],
    installed_tags: Iterable[str] | None,
    loaded_tags: Iterable[str] | None = None,
    observed_at: UtcDatetime | None = None,
) -> list[ModelStatus]:
    """Combine configured profiles with observed provider state.

    installed_tags=None means the provider was not probed: availability is 'unknown', never
    optimistically 'available'. A configured tag absent from a successful probe is
    'unavailable/not_installed'; nothing ever triggers a download. Matching is exact.
    """
    installed = None if installed_tags is None else set(installed_tags)
    loaded = None if loaded_tags is None else set(loaded_tags)
    out: list[ModelStatus] = []
    for p in profiles:
        if not p.enabled:
            availability, reason = Availability.unavailable, UnavailableReason.disabled
        elif installed is None:
            availability, reason = Availability.unknown, UnavailableReason.provider_not_probed
        elif p.tag in installed:
            availability, reason = Availability.available, None
        else:
            availability, reason = Availability.unavailable, UnavailableReason.not_installed
        if loaded is None or installed is None:
            state = LoadedState.unknown
        else:
            state = LoadedState.loaded if p.tag in loaded else LoadedState.not_loaded
        out.append(
            ModelStatus(
                profile=p,
                availability=availability,
                availability_reason=reason,
                loaded_state=state,
                observed_at=observed_at if installed is not None else None,
            )
        )
    return out
