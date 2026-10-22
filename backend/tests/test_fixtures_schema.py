"""Every schema-level fixture case must behave as labeled under the pydantic contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from localhydra.contracts.registry import REGISTRY
from tests.conftest import FIXTURES, SCHEMA_CASE_FILES, load_fixture


def _validate(name: str, data):
    typ, _ = REGISTRY[name]
    if isinstance(typ, type) and issubclass(typ, BaseModel):
        return typ.model_validate(data)
    return TypeAdapter(typ).validate_python(data)


def _all_cases():
    for fname in SCHEMA_CASE_FILES:
        doc = load_fixture(fname)
        assert doc["synthetic"] is True and "SYNTHETIC" in doc["label"]
        for c in doc["cases"]:
            yield pytest.param(c, id=f"{fname}::{c['name']}")


@pytest.mark.parametrize("case", list(_all_cases()))
def test_case(case):
    if case["valid"]:
        _validate(case["schema"], case["data"])
    else:
        with pytest.raises(ValidationError):
            _validate(case["schema"], case["data"])


def test_valid_cases_roundtrip_without_loss():
    for fname in SCHEMA_CASE_FILES:
        for c in load_fixture(fname)["cases"]:
            if not c["valid"]:
                continue
            obj = _validate(c["schema"], c["data"])
            dumped = (
                obj.model_dump(mode="json")
                if isinstance(obj, BaseModel)
                else TypeAdapter(REGISTRY[c["schema"]][0]).dump_python(obj, mode="json")
            )
            # Serialization-mode fixtures list every field, so a round trip reproduces them
            # (timestamps are compared after normalizing the fractional-second spelling).
            if REGISTRY[c["schema"]][1] == "serialization":
                assert _norm(dumped) == _norm(c["data"]), c["name"]


def _norm(v):
    if isinstance(v, str) and v.startswith("2026-") and v.endswith("Z"):
        head, _, frac = v[:-1].partition(".")
        return head + "." + (frac or "0").ljust(6, "0") + "Z"
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, float):
        return round(v, 9)
    return v


def test_every_case_references_a_registered_schema():
    for fname in SCHEMA_CASE_FILES:
        for c in load_fixture(fname)["cases"]:
            assert c["schema"] in REGISTRY, c["name"]


def test_every_schema_has_a_valid_and_an_invalid_fixture():
    exempt = {"Endpoints", "ModelConfig", "ModelStatus", "ContextEstimate", "ContextFit"}
    valid, invalid = set(), set()
    for fname in SCHEMA_CASE_FILES:
        for c in load_fixture(fname)["cases"]:
            (valid if c["valid"] else invalid).add(c["schema"])
    missing_valid = set(REGISTRY) - exempt - valid
    assert not missing_valid, f"no valid fixture for {sorted(missing_valid)}"
    # Pure output records need no negative case.
    optional_invalid = {"RunDetail", "ConversationDetail"} | {
        n for n in REGISTRY if n in ("CreateRunResponse", "ModelProfile")
    }
    missing_invalid = set(REGISTRY) - exempt - invalid - optional_invalid
    assert not missing_invalid, f"no invalid fixture for {sorted(missing_invalid)}"


def test_fixture_files_are_labeled_synthetic():
    for p in sorted(FIXTURES.glob("*.json")):
        doc = json.loads(p.read_text())
        assert doc.get("synthetic") is True, p.name
        assert "SYNTHETIC" in doc.get("label", ""), p.name
