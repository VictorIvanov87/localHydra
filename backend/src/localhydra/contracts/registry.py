"""Schema registry and exporter.

`python -m localhydra.contracts.registry`          regenerate contracts/schemas/*.json + MANIFEST.json
`python -m localhydra.contracts.registry --check`  fail if committed files differ from the code

Request/config models export their *validation* schema (defaults optional). Everything the server
emits exports its *serialization* schema (fields with defaults are always present), so frontend
types and fixtures describe real payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from . import api, config, context, errors, events, lifecycle, metrics, models, privacy, routing, runs
from .common import CONTRACT_VERSION, EVENT_SCHEMA_VERSION

# name -> (python type, mode)
REGISTRY: dict[str, tuple[Any, str]] = {
    # requests / config (validation mode)
    "CreateRunRequest": (runs.CreateRunRequest, "validation"),
    "ListRunsQuery": (runs.ListRunsQuery, "validation"),
    "AppConfig": (config.AppConfig, "validation"),
    "ModelConfig": (models.ModelConfig, "validation"),
    "ModelProfile": (models.ModelProfile, "validation"),
    "PrivacyConfig": (privacy.PrivacyConfig, "validation"),
    # responses / records (serialization mode)
    "ErrorEnvelope": (errors.ErrorEnvelope, "serialization"),
    "HealthResponse": (api.HealthResponse, "serialization"),
    "ModelsResponse": (models.ModelsResponse, "serialization"),
    "ModelStatus": (models.ModelStatus, "serialization"),
    "CreateRunResponse": (runs.CreateRunResponse, "serialization"),
    "RunSummary": (runs.RunSummary, "serialization"),
    "RunDetail": (runs.RunDetail, "serialization"),
    "RunPage": (runs.RunPage, "serialization"),
    "CancelResponse": (lifecycle.CancelResponse, "serialization"),
    "ConversationDetail": (runs.ConversationDetail, "serialization"),
    "ConversationResetResponse": (runs.ConversationResetResponse, "serialization"),
    "Event": (events.Event, "serialization"),
    "RoutingRequirements": (routing.RoutingRequirements, "serialization"),
    "RoutingDecision": (routing.RoutingDecision, "serialization"),
    "ContextEstimate": (context.ContextEstimate, "serialization"),
    "ContextFit": (context.ContextFit, "serialization"),
    "RawOllamaMetrics": (metrics.RawOllamaMetrics, "serialization"),
    "NormalizedMetrics": (metrics.NormalizedMetrics, "serialization"),
    "ProgressMetrics": (metrics.ProgressMetrics, "serialization"),
    "Endpoints": (list[api.Endpoint], "serialization"),
}


def schema_for(name: str) -> dict[str, Any]:
    typ, mode = REGISTRY[name]
    if isinstance(typ, type) and issubclass(typ, BaseModel):
        schema = typ.model_json_schema(mode=mode)  # type: ignore[arg-type]
    else:
        schema = TypeAdapter(typ).json_schema(mode=mode)  # type: ignore[arg-type]
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
    schema["title"] = name
    return schema


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def endpoints_json() -> Any:
    return [e.model_dump(mode="json") for e in api.ENDPOINTS]


def build_files() -> dict[str, str]:
    files: dict[str, str] = {}
    manifest_schemas: dict[str, Any] = {}
    for name, (_, mode) in REGISTRY.items():
        text = _dump(schema_for(name))
        filename = f"{name}.schema.json"
        files[filename] = text
        manifest_schemas[name] = {
            "file": filename,
            "mode": mode,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    files["endpoints.json"] = _dump(endpoints_json())
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "schemas": manifest_schemas,
        "endpoints_sha256": hashlib.sha256(files["endpoints.json"].encode()).hexdigest(),
    }
    files["MANIFEST.json"] = _dump(manifest)
    return files


def find_schema_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "contracts" / "schemas").is_dir():
            return parent / "contracts" / "schemas"
    raise FileNotFoundError("contracts/schemas not found above " + str(here))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="localhydra.contracts.registry")
    ap.add_argument("--check", action="store_true", help="verify committed schemas match the code")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or find_schema_dir()
    files = build_files()
    if args.check:
        problems = []
        for fname, text in files.items():
            p = out / fname
            if not p.exists() or p.read_text() != text:
                problems.append(f"out of date: {p}")
        extra = {p.name for p in out.glob("*.json")} - set(files)
        problems += [f"unexpected file: {out / n}" for n in sorted(extra)]
        if problems:
            print("\n".join(problems), file=sys.stderr)
            print("Run: scripts/contracts.sh generate  (and follow docs/contracts.md)", file=sys.stderr)
            return 1
        print(f"contracts OK: {len(files)} files match code (contract {CONTRACT_VERSION})")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        (out / fname).write_text(text)
    print(f"wrote {len(files)} files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
