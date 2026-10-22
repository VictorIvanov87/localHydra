"""Shared primitives for the frozen v1 contracts."""

from __future__ import annotations

from typing import Annotated, Generic, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

# Bump rules are documented in docs/contracts.md (contract change process).
CONTRACT_VERSION = "1.0.0"
EVENT_SCHEMA_VERSION = 1

RunId = Annotated[str, StringConstraints(pattern=r"^run_[0-9a-f]{32}$")]
ConversationId = Annotated[str, StringConstraints(pattern=r"^conv_[0-9a-f]{32}$")]
PhaseId = Annotated[str, StringConstraints(pattern=r"^ph_[0-9a-f]{32}$")]
ModelProfileId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
# Explicit "name:tag" only. A bare name implies ":latest" and is not an exact tag.
OllamaTag = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*){0,2}:[A-Za-z0-9][A-Za-z0-9._-]*$",
        max_length=200,
    ),
]

UtcDatetime = AwareDatetime


class Contract(BaseModel):
    """Base for every contract model: unknown fields are rejected, never ignored."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


T = TypeVar("T")

MAX_PAGE_LIMIT = 100
DEFAULT_PAGE_LIMIT = 20


class Page(Contract, Generic[T]):
    """Cursor pagination. Order is (created_at desc, id desc); cursors are opaque."""

    items: list[T]
    limit: int = Field(ge=1, le=MAX_PAGE_LIMIT)
    next_cursor: str | None = Field(
        default=None, description="Opaque. Null exactly when has_more is false."
    )
    has_more: bool

    def model_post_init(self, __context: object) -> None:
        if self.has_more != (self.next_cursor is not None):
            raise ValueError("next_cursor must be set if and only if has_more is true")
        if len(self.items) > self.limit:
            raise ValueError("items must not exceed limit")


class PageQuery(Contract):
    limit: int = Field(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT)
    cursor: str | None = Field(default=None, min_length=1, max_length=512)
