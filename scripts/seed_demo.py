"""Idempotently seed the local Fake Demo through its public HTTP contract."""

# ruff: noqa: RUF001 -- full-width punctuation is intentional Chinese fixture content.

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

DEFAULT_TOKEN = "demo-local-token"
POLICY = """# Northstar 年假申请制度

年假应至少提前三个工作日提交，直属经理在两个工作日内审批。
本文件为公开演示使用的原创虚构制度，不包含真实企业数据。
""".encode()


def seed(base_url: str, token: str) -> dict[str, object]:
    first = _upload(base_url, token)
    second = _upload(base_url, token)
    if first.get("document_id") != second.get("document_id"):
        raise RuntimeError("idempotent seed returned different document IDs")
    return {
        "status": "seeded",
        "document_id": first["document_id"],
        "job_id": first["job_id"],
        "idempotent": True,
    }


def _upload(base_url: str, token: str) -> dict[str, object]:
    boundary = "enterprise-rag-stage20-boundary"
    fields = (
        _part(boundary, "workspace_id", "policies"),
        _part(boundary, "visibility", "restricted"),
        _part(boundary, "acl_groups", "hr"),
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="annual-leave.md"\r\n'
            "Content-Type: text/markdown\r\n\r\n"
        ).encode()
        + POLICY
        + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/documents",
        data=b"".join(fields),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Idempotency-Key": "northstar-demo-policy-v1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 202:
                raise RuntimeError(f"seed returned HTTP {response.status}")
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"seed failed with HTTP {exc.code}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("seed response was not an object")
    return payload


def _part(boundary: str, name: str, value: str) -> bytes:
    return (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("DEMO_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.getenv("DEMO_BEARER_TOKEN", DEFAULT_TOKEN))
    args = parser.parse_args()
    print(json.dumps(seed(args.base_url, args.token), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
