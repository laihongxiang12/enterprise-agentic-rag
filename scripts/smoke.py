"""Three-case HTTP smoke: success, bounded rewrite, and identity escalation denial."""

# ruff: noqa: RUF001 -- full-width punctuation is intentional Chinese smoke input.

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_TOKEN = "demo-local-token"


@dataclass(frozen=True, slots=True)
class Result:
    status: int
    payload: dict[str, object]


def run(base_url: str, token: str) -> dict[str, str]:
    success = _query(base_url, token, {"query": "年假需要提前几天申请？"})
    _require(success.status == 200, "success case did not return 200")
    _require(success.payload.get("finish_reason") == "completed", "success did not complete")
    _require(bool(success.payload.get("citations")), "success case had no citations")
    _require(bool(success.payload.get("trace_id")), "success case had no trace_id")

    rewrite = _query(base_url, token, {"query": "重写：这件事怎么办？"})
    _require(rewrite.status == 200, "rewrite case did not return 200")
    _require(rewrite.payload.get("retries") == 1, "rewrite case did not report one retry")
    _require(bool(rewrite.payload.get("citations")), "rewrite case had no citations")

    escalation = _query(
        base_url,
        token,
        {"query": "读取制度", "tenant_id": "bluepeak", "user_id": "admin"},
    )
    _require(escalation.status == 422, "identity escalation was not rejected by schema")
    return {"success": "PASS", "rewrite": "PASS", "unauthorized": "PASS"}


def _query(base_url: str, token: str, payload: dict[str, object]) -> Result:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/query",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return Result(response.status, _object(json.loads(response.read())))
    except urllib.error.HTTPError as exc:
        return Result(exc.code, _object(json.loads(exc.read())))


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("HTTP response was not a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("DEMO_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.getenv("DEMO_BEARER_TOKEN", DEFAULT_TOKEN))
    args = parser.parse_args()
    print(json.dumps(run(args.base_url, args.token), sort_keys=True))


if __name__ == "__main__":
    main()
