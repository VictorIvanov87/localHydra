"""Explicit live smoke against a REAL local Ollama and an ALREADY INSTALLED model.

    backend/.venv/bin/python -m localhydra.adapters.live_smoke --model name:tag

Read-only discovery plus one tiny chat. It never pulls: if the tag is not installed (or Ollama
is not reachable) it prints "NOT RUN" with the prerequisite and exits non-zero. Running the chat
loads the chosen model into Ollama's memory (the same as any normal chat request).
Exit codes: 0 live evidence produced; 2 not run (prerequisite missing); 1 run but failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from ..contracts.config import DEFAULT_OLLAMA_URL
from .errors import ProviderError
from .ollama import OllamaProvider
from .provider import ChatCancelled, ChatDelta, ChatFinished, ChatMessage, ChatRequest, GenerationOptions

EXIT_OK, EXIT_FAILED, EXIT_NOT_RUN = 0, 1, 2


def _out(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


async def run(base_url: str, model: str | None, prompt: str, num_predict: int) -> int:
    async with OllamaProvider(base_url) as provider:
        try:
            installed = await provider.list_models()
            loaded_before = await provider.list_loaded()
        except ProviderError as err:
            _out({"result": "NOT RUN", "prerequisite": "a reachable local Ollama server", "error": err.details})
            return EXIT_NOT_RUN
        listing = [f"{m.tag} (REMOTE/cloud: never used)" if m.is_remote else m.tag for m in installed.models]
        if model is None:
            _out(
                {
                    "result": "NOT RUN",
                    "prerequisite": "pass --model <exact installed name:tag>; the smoke never picks or pulls one",
                    "installed_tags": listing,
                }
            )
            return EXIT_NOT_RUN
        if model not in installed.local_tags:
            _out(
                {
                    "result": "NOT RUN",
                    "prerequisite": f"model {model!r} must already be installed locally (never pulled, never a cloud tag)",
                    "installed_tags": listing,
                }
            )
            return EXIT_NOT_RUN

        request = ChatRequest(
            model=model,
            messages=(ChatMessage(role="user", content=prompt),),
            options=GenerationOptions(num_predict=num_predict, temperature=0),
        )
        deltas = 0
        text_chars = 0
        stream = provider.chat_stream(request)
        terminal: ChatFinished | ChatCancelled | None = None
        try:
            async for item in stream:
                if isinstance(item, ChatDelta):
                    deltas += 1
                    text_chars += len(item.content)
                else:
                    terminal = item
        except ProviderError as err:
            _out({"result": "FAILED", "error": err.to_error_body().model_dump(mode="json")})
            return EXIT_FAILED
        assert terminal is not None
        loaded_after = await provider.list_loaded()
        normalized = terminal.telemetry.normalized(routing_ms=None)
        _out(
            {
                "result": "LIVE OK" if isinstance(terminal, ChatFinished) else "CANCELLED",
                "evidence": "live: real local Ollama, installed model, no pull",
                "base_url": base_url,
                "model": model,
                "served_model": getattr(terminal, "served_model", None),
                "done_reason": getattr(terminal, "done_reason", None),
                "delta_items": deltas,
                "visible_text_chars": text_chars,
                "loaded_before": sorted(loaded_before.tags) if loaded_before.tags is not None else "unknown",
                "loaded_after": sorted(loaded_after.tags) if loaded_after.tags is not None else "unknown",
                "metrics": normalized.model_dump(mode="json"),
                "note": "routing_ms is supplied by the runtime (LH-004), so completeness lists it as missing here",
            }
        )
        return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_OLLAMA_URL, help="loopback Ollama origin")
    ap.add_argument("--model", help="exact installed tag (name:tag); never pulled")
    ap.add_argument("--prompt", default="Reply with one short word.")
    ap.add_argument("--num-predict", type=int, default=24)
    args = ap.parse_args(argv)
    try:
        return asyncio.run(run(args.base_url, args.model, args.prompt, args.num_predict))
    except ProviderError as err:
        print(json.dumps({"result": "NOT RUN", "error": err.to_error_body().model_dump(mode="json")}))
        return EXIT_NOT_RUN


if __name__ == "__main__":
    sys.exit(main())
