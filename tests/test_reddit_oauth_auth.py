from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.routes import auth as auth_route
from app.db.deps import get_db
from app.services.reddit_oauth_service import RedditOAuthAuthResult


async def _override_get_db():
    yield object()


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth_route.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = _override_get_db
    return app


def test_reddit_authorize_endpoint_returns_url(monkeypatch) -> None:
    expected_state = "signed-state"
    expected_url = "https://reddit.example/auth?state=signed-state"

    def fake_authorize_payload(*, redirect_uri: str, intent: str) -> dict:
        assert redirect_uri == "http://localhost:3000/reddit/callback"
        assert intent == "signin"
        return {
            "authorization_url": expected_url,
            "state": expected_state,
            "expires_in_seconds": 600,
        }

    monkeypatch.setattr(auth_route, "build_reddit_authorization_payload", fake_authorize_payload)

    client = TestClient(_build_test_app())
    response = client.post(
        "/api/v1/auth/reddit/authorize",
        json={
            "redirect_uri": "http://localhost:3000/reddit/callback",
            "intent": "signin",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["authorization_url"] == expected_url
    assert payload["state"] == expected_state
    assert payload["expires_in_seconds"] == 600


def test_reddit_signup_endpoint_returns_token(monkeypatch) -> None:
    user_id = uuid4()
    state: dict[str, object] = {}

    async def fake_complete_reddit_oauth(**kwargs) -> RedditOAuthAuthResult:
        assert kwargs["intent"] == "signup"
        return RedditOAuthAuthResult(
            user=SimpleNamespace(id=user_id),
            reddit_username="sample_redditor",
            is_new_user=True,
        )

    async def fake_queue_signup_inference_for_user(*, db, user_id, reddit_username, wait_for_completion):
        state["queued"] = {
            "db": db,
            "user_id": user_id,
            "reddit_username": reddit_username,
            "wait_for_completion": wait_for_completion,
        }
        return False, {"status": "queued"}

    monkeypatch.setattr(auth_route, "complete_reddit_oauth", fake_complete_reddit_oauth)
    monkeypatch.setattr(auth_route, "queue_signup_inference_for_user", fake_queue_signup_inference_for_user)

    client = TestClient(_build_test_app())
    response = client.post(
        "/api/v1/auth/reddit/signup",
        json={
            "code": "reddit-code",
            "state": "signed-state",
            "redirect_uri": "http://localhost:3000/reddit/callback",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["reddit_username"] == "sample_redditor"
    assert payload["is_new_user"] is True
    assert isinstance(payload["access_token"], str)
    assert state["queued"]["user_id"] == user_id
    assert state["queued"]["reddit_username"] == "sample_redditor"
    assert state["queued"]["wait_for_completion"] is False
