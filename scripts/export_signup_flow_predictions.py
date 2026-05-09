"""Utilities for export signup flow predictions."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


EXPECTED_SIGNUP_DOMAINS = ["personality"]


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Run signup flow inference through the API and save predictions JSON in artifacts/."
    )
    parser.add_argument("--email", required=True, help="Email used for /auth/signup and /auth/signin")
    parser.add_argument("--password", required=True, help="Password used for /auth/signup and /auth/signin")
    parser.add_argument("--reddit-username", required=True, help="Reddit username for signup inference")
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Max polling time for inference job completion (default: 180)",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="Polling interval for /inference/jobs/{job_id} (default: 2)",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: artifacts/signup_flow/<reddit_username>_<timestamp>.json",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    """Handle UTC now iso."""
    return datetime.now(timezone.utc).isoformat()


def _raise_for_unexpected_status(response: httpx.Response, allowed: set[int], context: str) -> None:
    """Handle raise for unexpected status."""
    if response.status_code not in allowed:
        raise RuntimeError(
            f"{context} failed with status={response.status_code}, body={response.text[:500]}"
        )


def ensure_user_exists(client: httpx.Client, *, email: str, password: str) -> dict[str, Any]:
    """Ensure user exists."""
    payload = {"email": email, "password": password}
    response = client.post("/api/v1/auth/signup", json=payload)
    if response.status_code == 201:
        return {"status": "created", "payload": response.json()}
    if response.status_code == 409:
        return {"status": "already_exists"}
    _raise_for_unexpected_status(response, {201, 409}, "auth signup")
    return {"status": "unknown"}


def sign_in(client: httpx.Client, *, email: str, password: str) -> str:
    """Handle sign in."""
    payload = {"email": email, "password": password}
    response = client.post("/api/v1/auth/signin", json=payload)
    _raise_for_unexpected_status(response, {200}, "auth signin")
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("auth signin response did not include access_token")
    return str(token)


def trigger_signup_inference(
    client: httpx.Client,
    *,
    token: str,
    reddit_username: str,
) -> dict[str, Any]:
    """Trigger signup inference."""
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post(
        "/api/v1/users/me/reddit",
        json={"reddit_username": reddit_username},
        headers=headers,
    )
    _raise_for_unexpected_status(response, {200, 202}, "signup inference trigger")
    return response.json()


def poll_job_until_finished(
    client: httpx.Client,
    *,
    token: str,
    job_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Handle poll job until finished."""
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/inference/jobs/{job_id}", headers=headers)
        _raise_for_unexpected_status(response, {200}, "job status poll")
        payload = response.json()
        last_payload = payload

        if "domain" in payload:
            return payload
        if payload.get("status") == "failed":
            return payload

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out waiting for job_id={job_id}. Last payload={json.dumps(last_payload or {}, default=str)}"
    )


def default_output_path(reddit_username: str) -> Path:
    """Handle default output path."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "signup_flow" / f"{reddit_username}_{timestamp}.json"


def main() -> int:
    """Run the module entry point."""
    args = parse_args()
    started_at = _utc_now_iso()
    base_url = args.api_base_url.rstrip("/")
    output_path = Path(args.output) if args.output else default_output_path(args.reddit_username)

    with httpx.Client(base_url=base_url, timeout=args.request_timeout_seconds) as client:
        signup_status = ensure_user_exists(client, email=args.email, password=args.password)
        token = sign_in(client, email=args.email, password=args.password)
        trigger_payload = trigger_signup_inference(
            client,
            token=token,
            reddit_username=args.reddit_username,
        )

        if "domain" in trigger_payload or trigger_payload.get("status") == "failed":
            final_payload = trigger_payload
        else:
            job_id = str(trigger_payload.get("job_id", ""))
            if not job_id:
                raise RuntimeError(f"Queued signup response did not include job_id: {trigger_payload}")
            final_payload = poll_job_until_finished(
                client,
                token=token,
                job_id=job_id,
                timeout_seconds=args.timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
            )

    output_payload: dict[str, Any] = {
        "meta": {
            "generated_at": _utc_now_iso(),
            "started_at": started_at,
            "api_base_url": base_url,
            "expected_domains": EXPECTED_SIGNUP_DOMAINS,
        },
        "user": {
            "email": args.email,
            "reddit_username": args.reddit_username,
            "signup_status": signup_status,
        },
        "signup_trigger_response": trigger_payload,
        "signup_result": final_payload,
    }

    actual_domains = output_payload.get("signup_result", {}).get("domains")
    if isinstance(actual_domains, list):
        missing_domains = [d for d in EXPECTED_SIGNUP_DOMAINS if d not in actual_domains]
        output_payload["meta"]["missing_domains"] = missing_domains

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, default=str)

    print(f"Saved signup flow predictions JSON: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
