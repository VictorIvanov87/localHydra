"""Privacy defaults: nothing containing prompt/response content is retained unless opted in.

Retention is an operator (server config) decision snapshotted onto each run at creation.
There is no per-request opt-in in v1.

- prompts=true    persist request messages (RunDetail.messages, DB rows).
- responses=true  persist generated text (RunDetail.response_text) and delta text in stored events.
- Streaming is always live: SSE clients receive delta text as it is generated regardless of
  retention ("ephemeral streaming"). With responses=false the *persisted* copy of each delta has
  text=null and redacted=true, so a client that reconnects with Last-Event-ID receives redacted
  placeholders for deltas it missed; metadata history stays replayable, transcript replay does not.
- log_content=false: application logs never include prompt/response text or request bodies.
- export_content=false: exports omit content even when retained; true is only honored for
  fields that were actually retained.
- Conversation context for multi-turn chat is held in bounded process memory (see
  ConversationDetail) and is lost on restart; that is independent of the retention flags and is
  never written to the DB, events, logs or exports unless the flags above say so.
"""

from __future__ import annotations

from pydantic import model_validator

from .common import Contract


class Retention(Contract):
    prompts: bool = False
    responses: bool = False


class PrivacyConfig(Contract):
    retain_prompts: bool = False
    retain_responses: bool = False
    log_content: bool = False
    export_content: bool = False

    @model_validator(mode="after")
    def _export_needs_retention(self) -> PrivacyConfig:
        if self.export_content and not (self.retain_prompts or self.retain_responses):
            raise ValueError("export_content requires retain_prompts or retain_responses")
        return self

    def retention(self) -> Retention:
        return Retention(prompts=self.retain_prompts, responses=self.retain_responses)
