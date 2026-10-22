"""Context-fit estimation contract (method `ascii3_nonascii1_v1`).

This is not a tokenizer. It is a deliberately conservative upper-bound heuristic whose method
and uncertainty are disclosed with every estimate, and which routing must use for hard
eligibility:

  point  = ceil(ascii_chars / 3 + non_ascii_chars * 1) + 8 * message_count + 16
  upper  = ceil(point * 1.25)
  need   = upper + output_reserve
  fits   = need <= context_window

`output_reserve` is the caller's requested max output tokens, or the profile's
max_output_tokens when none was requested. Requesting more than the profile's
max_output_tokens is rejected as `output_budget_exceeded`, not clamped.
Characters are Unicode code points. The heuristic may still under-estimate pathological
inputs; that residual risk is stated in `uncertainty`, and provider-reported prompt_tokens
(NormalizedMetrics) are the ground truth after the fact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

from pydantic import Field

from .common import Contract
from .models import ModelProfile

METHOD_ID = "ascii3_nonascii1_v1"
UNCERTAINTY = (
    "Heuristic upper bound (x1.25 margin), not a tokenizer; may under-estimate unusual "
    "inputs. Provider-reported prompt_tokens are authoritative after generation."
)
PER_MESSAGE_OVERHEAD = 8
FIXED_OVERHEAD = 16
MARGIN = 1.25


class ContextEstimate(Contract):
    method: str = METHOD_ID
    message_count: int = Field(ge=0)
    point_tokens: int = Field(ge=0)
    upper_bound_tokens: int = Field(ge=0)
    margin: float = MARGIN
    uncertainty: str = UNCERTAINTY


class FitFailure(str, Enum):
    context_exceeded = "context_exceeded"
    output_budget_exceeded = "output_budget_exceeded"


class ContextFit(Contract):
    fits: bool
    context_window: int = Field(ge=1)
    upper_bound_tokens: int = Field(ge=0)
    output_reserve: int = Field(ge=0)
    required_tokens: int = Field(ge=0)
    failure: FitFailure | None = None


def estimate_context(contents: Sequence[str]) -> ContextEstimate:
    ascii_chars = sum(1 for c in "".join(contents) if ord(c) < 128)
    non_ascii = sum(len(c) for c in contents) - ascii_chars
    point = (
        math.ceil(ascii_chars / 3 + non_ascii)
        + PER_MESSAGE_OVERHEAD * len(contents)
        + FIXED_OVERHEAD
    )
    return ContextEstimate(
        message_count=len(contents),
        point_tokens=point,
        upper_bound_tokens=math.ceil(point * MARGIN),
    )


def check_context_fit(
    estimate: ContextEstimate,
    profile: ModelProfile,
    requested_max_output_tokens: int | None = None,
) -> ContextFit:
    reserve = (
        profile.max_output_tokens
        if requested_max_output_tokens is None
        else requested_max_output_tokens
    )
    required = estimate.upper_bound_tokens + reserve
    failure: FitFailure | None = None
    if reserve > profile.max_output_tokens:
        failure = FitFailure.output_budget_exceeded
    elif required > profile.context_window:
        failure = FitFailure.context_exceeded
    return ContextFit(
        fits=failure is None,
        context_window=profile.context_window,
        upper_bound_tokens=estimate.upper_bound_tokens,
        output_reserve=reserve,
        required_tokens=required,
        failure=failure,
    )
