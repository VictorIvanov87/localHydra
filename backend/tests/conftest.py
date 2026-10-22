from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "contracts" / "fixtures"
SCHEMAS = ROOT / "contracts" / "schemas"
EXAMPLE_CONFIG = ROOT / "config" / "localhydra.example.json"

SCHEMA_CASE_FILES = ["models.json", "metrics.json", "routing.json", "events.json", "api.json"]


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def approx_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(approx_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(approx_equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="session")
def app_client():
    from fastapi.testclient import TestClient

    from localhydra.app import create_app

    return TestClient(create_app(), base_url="http://127.0.0.1")
