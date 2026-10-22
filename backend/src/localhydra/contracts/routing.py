"""Routing contract: requirements, candidate evaluation and decision records.

Shape and invariants only. The routing policy itself belongs to LH-003.

Invariants enforced here (so every implementation is held to them):
- Hard eligibility happens before ranking; an ineligible candidate has no score.
- A selected model must be an eligible candidate. Manual override and sticky conversation
  selection never bypass eligibility.
- No eligible candidate => outcome `no_eligible_model`, nothing selected (the run then fails
  with ErrorCode.no_eligible_model).
- Classifier output (RoutingRequirements) is separate from model mapping: it has no field that
  can name a model, endpoint or permission.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import Field, model_validator

from .common import Contract, ModelProfileId, OllamaTag, UtcDatetime
from .context import ContextEstimate, ContextFit
from .models import Modality, TaskType


class RoutingMode(str, Enum):
    auto = "auto"  # deterministic rules
    classifier = "classifier"  # optional local classifier supplies requirements (LH-006)
    manual = "manual"  # explicit model override; still subject to eligibility


class RequirementsSource(str, Enum):
    rules = "rules"
    classifier = "classifier"
    operator = "operator"


class RoutingRequirements(Contract):
    task_type: TaskType
    required_modalities: list[Modality] = Field(default_factory=lambda: [Modality.text])
    requires_tools: bool = False
    source: RequirementsSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    classifier_id: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def _source_fields(self) -> RoutingRequirements:
        if self.source is not RequirementsSource.classifier and (
            self.confidence is not None or self.classifier_id is not None
        ):
            raise ValueError("confidence/classifier_id are only valid when source is 'classifier'")
        return self


class RejectionReason(str, Enum):
    disabled = "disabled"
    not_installed = "not_installed"
    availability_unknown = "availability_unknown"
    modality_unsupported = "modality_unsupported"
    tools_unsupported = "tools_unsupported"
    context_exceeded = "context_exceeded"
    output_budget_exceeded = "output_budget_exceeded"


class RankingComponent(Contract):
    name: str = Field(min_length=1, max_length=64)
    value: float
    weight: float
    contribution: float

    @model_validator(mode="after")
    def _product(self) -> RankingComponent:
        if not math.isclose(self.contribution, self.value * self.weight, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("contribution must equal value * weight")
        return self


class CandidateEvaluation(Contract):
    model_id: ModelProfileId
    tag: OllamaTag
    eligible: bool
    rejections: list[RejectionReason] = Field(default_factory=list)
    context_fit: ContextFit | None = None
    components: list[RankingComponent] = Field(default_factory=list)
    score: float | None = None

    @model_validator(mode="after")
    def _consistent(self) -> CandidateEvaluation:
        if self.eligible == bool(self.rejections):
            raise ValueError("eligible must be true exactly when there are no rejections")
        if not self.eligible and (self.score is not None or self.components):
            raise ValueError("ineligible candidates are never ranked: no score or components")
        if self.eligible and self.components and self.score is not None:
            if not math.isclose(
                self.score, sum(c.contribution for c in self.components), rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError("score must equal the sum of component contributions")
        return self


class RoutingOutcome(str, Enum):
    selected = "selected"
    no_eligible_model = "no_eligible_model"


class RoutingReason(str, Enum):
    manual_override = "manual_override"
    sticky_conversation = "sticky_conversation"
    highest_score = "highest_score"
    only_eligible_candidate = "only_eligible_candidate"
    tie_break_profile_order = "tie_break_profile_order"


class RoutingDecision(Contract):
    policy_version: str = Field(min_length=1, max_length=64)
    mode: RoutingMode
    requirements: RoutingRequirements
    context_estimate: ContextEstimate
    candidates: list[CandidateEvaluation] = Field(min_length=1)
    outcome: RoutingOutcome
    selected_model_id: ModelProfileId | None = None
    reasons: list[RoutingReason] = Field(default_factory=list)
    override_applied: bool = False
    sticky_applied: bool = False
    decided_at: UtcDatetime

    @model_validator(mode="after")
    def _consistent(self) -> RoutingDecision:
        ids = [c.model_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidates must be unique by model_id")
        by_id = {c.model_id: c for c in self.candidates}
        if self.outcome is RoutingOutcome.selected:
            chosen = by_id.get(self.selected_model_id or "")
            if chosen is None:
                raise ValueError("selected_model_id must be one of the evaluated candidates")
            if not chosen.eligible:
                raise ValueError("an ineligible candidate can never be selected")
            if not self.reasons:
                raise ValueError("a selection must record at least one reason")
        else:
            if self.selected_model_id is not None:
                raise ValueError("no_eligible_model must not select a model")
            if any(c.eligible for c in self.candidates):
                raise ValueError("no_eligible_model requires every candidate to be ineligible")
        if self.override_applied != (RoutingReason.manual_override in self.reasons):
            raise ValueError("override_applied must match the manual_override reason")
        if self.sticky_applied != (RoutingReason.sticky_conversation in self.reasons):
            raise ValueError("sticky_applied must match the sticky_conversation reason")
        if self.mode is RoutingMode.manual and self.outcome is RoutingOutcome.selected:
            if not self.override_applied:
                raise ValueError("manual mode selections must record manual_override")
        if self.override_applied and self.mode is not RoutingMode.manual:
            raise ValueError("override_applied requires manual mode")
        return self
