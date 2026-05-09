"""Services for domain inference utils."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def select_current_week_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select current week posts."""
    if not posts:
        return []

    dated_posts = []
    for post in posts:
        dt = parse_post_datetime(post)
        if dt is None:
            continue
        dated_posts.append((dt, post))

    if not dated_posts:
        return posts

    latest_dt = max(dt for dt, _ in dated_posts)
    week_start = latest_dt - timedelta(days=7)
    selected = [post for dt, post in dated_posts if dt >= week_start]
    return selected or posts


def parse_post_datetime(post: dict[str, Any]) -> datetime | None:
    """Parse post datetime."""
    created_at = post.get("created_at")
    if isinstance(created_at, str) and created_at:
        value = created_at.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass

    created_utc = post.get("created_utc")
    if created_utc is not None:
        try:
            return datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
        except Exception:
            return None
    return None
