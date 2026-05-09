"""Extraction helpers for reddit."""
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import requests

from src.extractors.base import ExtractionError, SocialPostExtractor
from src.extractors.models import SocialAccountLink

logger = logging.getLogger(__name__)


class RedditPostExtractor(SocialPostExtractor):
    """Extract all submitted posts for a Reddit account via Reddit's public JSON listing."""

    provider = "reddit"
    base_url = "https://www.reddit.com"

    def __init__(
        self,
        user_agent: Optional[str] = None,
        page_size: int = 100,
        request_timeout_seconds: float = 30.0,
        page_delay_seconds: float = 0.2,
        max_pages: Optional[int] = None,
        include_comments: bool = True,
        max_comments_per_post: Optional[int] = 25,
        comments_sort: str = "top",
    ):
        """Initialize the reddit post extractor."""
        self.user_agent = (
            user_agent
            or os.getenv("REDDIT_USER_AGENT")
            or "MindMirrorRedditExtractor/1.0"
        )
        self.page_size = max(1, min(page_size, 100))
        self.request_timeout_seconds = request_timeout_seconds
        self.page_delay_seconds = max(0.0, page_delay_seconds)
        self.max_pages = max_pages
        self.include_comments = include_comments
        self.max_comments_per_post = (
            max(1, max_comments_per_post) if max_comments_per_post is not None else None
        )
        self.comments_sort = comments_sort
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self._oauth_token: Optional[str] = None
        self._oauth_token_expires_at: float = 0.0

    def extract_posts(self, account: SocialAccountLink) -> List[Dict[str, Any]]:
        """Extract posts."""
        username = account.external_username.strip()
        if not username:
            raise ExtractionError("Reddit username is required.")

        posts: List[Dict[str, Any]] = []
        next_token: Optional[str] = None
        pages_fetched = 0

        while True:
            listing = self._fetch_listing_page(username=username, after=next_token)
            data = listing.get("data", {})
            children = data.get("children", [])
            for item in children:
                if item.get("kind") != "t3":
                    continue
                post_data = item.get("data", {})
                post = self._normalize_post(post_data, username=username)
                if self.include_comments:
                    try:
                        comments = self._fetch_post_comments(
                            username=username,
                            post_data=post_data,
                        )
                    except ExtractionError as exc:
                        logger.warning(
                            "Failed to fetch comments for post %s by %s: %s",
                            post.get("post_id"),
                            username,
                            exc,
                        )
                        comments = []
                    post["comments"] = comments
                    post["comments_extracted_count"] = len(comments)
                posts.append(post)

            next_token = data.get("after")
            pages_fetched += 1
            if not next_token:
                break
            if self.max_pages is not None and pages_fetched >= self.max_pages:
                logger.info(
                    "Stopping Reddit extraction for %s after %s pages due to max_pages.",
                    username,
                    pages_fetched,
                )
                break
            if self.page_delay_seconds:
                time.sleep(self.page_delay_seconds)

        logger.info(
            "Extracted %s Reddit posts for %s across %s page(s).",
            len(posts),
            username,
            pages_fetched,
        )
        return posts

    def _fetch_listing_page(
        self,
        username: str,
        after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch listing page."""
        query = {
            "limit": self.page_size,
            "raw_json": 1,
        }
        if after:
            query["after"] = after

        url = f"{self.base_url}/user/{username}/submitted.json?{urlencode(query)}"
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code == 404:
                raise ExtractionError(f"Reddit user '{username}' was not found.") from exc
            if exc.code in {403, 429}:
                logger.info(
                    "Public Reddit listing request blocked for %s (HTTP %s); trying OAuth fallback.",
                    username,
                    exc.code,
                )
                if self._has_oauth_credentials():
                    return self._fetch_listing_page_via_oauth(username=username, after=after)
                raise ExtractionError(
                    f"Reddit rejected the request for '{username}' (HTTP {exc.code})."
                ) from exc
            raise ExtractionError(
                f"Reddit request failed for '{username}' with HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise ExtractionError(
                f"Unable to reach Reddit for '{username}': {exc.reason}"
            ) from exc

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"Invalid JSON returned by Reddit for '{username}'."
            ) from exc

    def _fetch_listing_page_via_oauth(
        self,
        username: str,
        after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch listing page via OAuth."""
        token = self._get_oauth_token()
        query = {
            "limit": self.page_size,
            "raw_json": 1,
        }
        if after:
            query["after"] = after

        response = requests.get(
            f"https://oauth.reddit.com/user/{username}/submitted",
            params=query,
            headers={
                "Authorization": f"bearer {token}",
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            timeout=self.request_timeout_seconds,
        )
        if response.status_code == 404:
            raise ExtractionError(f"Reddit user '{username}' was not found.")
        if response.status_code >= 400:
            raise ExtractionError(
                f"Reddit rejected the OAuth request for '{username}' (HTTP {response.status_code})."
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ExtractionError(
                f"Invalid JSON returned by Reddit for '{username}'."
            ) from exc

    def _fetch_post_comments(
        self,
        username: str,
        post_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Fetch post comments."""
        permalink = post_data.get("permalink")
        if not permalink:
            return []

        permalink_path = str(permalink).rstrip("/")
        if not permalink_path.startswith("/"):
            permalink_path = f"/{permalink_path}"
        # Reddit permalinks can contain non-ASCII title slugs; urllib requires ASCII-safe URLs.
        permalink_path = quote(permalink_path, safe="/%")

        query = {
            "raw_json": 1,
            "sort": self.comments_sort,
            "limit": self._comment_page_size(),
            "depth": 10,
        }
        url = f"{self.base_url}{permalink_path}.json?{urlencode(query)}"
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code == 404:
                return []
            if exc.code in {403, 429} and self._has_oauth_credentials():
                return self._fetch_post_comments_via_oauth(
                    username=username,
                    permalink_path=permalink_path,
                    post_id=post_data.get("id"),
                )
            raise ExtractionError(
                f"Reddit comments request failed for '{username}' with HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise ExtractionError(
                f"Unable to reach Reddit comments for '{username}': {exc.reason}"
            ) from exc

        try:
            response_payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"Invalid comments JSON returned by Reddit for '{username}'."
            ) from exc

        return self._normalize_comment_payload(
            payload=response_payload,
            username=username,
            post_id=post_data.get("id"),
        )

    def _fetch_post_comments_via_oauth(
        self,
        username: str,
        permalink_path: str,
        post_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Fetch post comments via OAuth."""
        token = self._get_oauth_token()
        response = requests.get(
            f"https://oauth.reddit.com{permalink_path}.json",
            params={
                "raw_json": 1,
                "sort": self.comments_sort,
                "limit": self._comment_page_size(),
                "depth": 10,
            },
            headers={
                "Authorization": f"bearer {token}",
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            timeout=self.request_timeout_seconds,
        )
        if response.status_code == 404:
            return []
        if response.status_code >= 400:
            raise ExtractionError(
                f"Reddit rejected the OAuth comments request for '{username}' (HTTP {response.status_code})."
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise ExtractionError(
                f"Invalid comments JSON returned by Reddit for '{username}'."
            ) from exc

        return self._normalize_comment_payload(
            payload=response_payload,
            username=username,
            post_id=post_id,
        )

    def _get_oauth_token(self) -> str:
        """Get OAuth token."""
        now = time.time()
        if self._oauth_token and now < self._oauth_token_expires_at:
            return self._oauth_token

        if not self._has_oauth_credentials():
            raise ExtractionError("Reddit OAuth credentials are not configured.")

        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"},
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            timeout=self.request_timeout_seconds,
        )
        if response.status_code >= 400:
            raise ExtractionError(
                f"Failed to authenticate with Reddit (HTTP {response.status_code})."
            )

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise ExtractionError("Reddit OAuth token response did not include an access token.")

        expires_in = int(payload.get("expires_in") or 3600)
        self._oauth_token = str(access_token)
        self._oauth_token_expires_at = now + max(expires_in - 60, 60)
        return self._oauth_token

    def _has_oauth_credentials(self) -> bool:
        """Return whether OAuth credentials."""
        return bool(self.client_id and self.client_secret)

    def _comment_page_size(self) -> int:
        """Handle comment page size."""
        if self.max_comments_per_post is None:
            return 100
        return max(1, min(100, self.max_comments_per_post))

    def _fetch_listing_page_via_oauth(
        self,
        username: str,
        after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch listing page via OAuth."""
        token = self._get_oauth_token()
        query = {
            "limit": self.page_size,
            "raw_json": 1,
        }
        if after:
            query["after"] = after

        response = requests.get(
            f"https://oauth.reddit.com/user/{username}/submitted",
            params=query,
            headers={
                "Authorization": f"bearer {token}",
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            timeout=self.request_timeout_seconds,
        )
        if response.status_code == 404:
            raise ExtractionError(f"Reddit user '{username}' was not found.")
        if response.status_code >= 400:
            raise ExtractionError(
                f"Reddit rejected the OAuth request for '{username}' (HTTP {response.status_code})."
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ExtractionError(
                f"Invalid JSON returned by Reddit for '{username}'."
            ) from exc

    def _get_oauth_token(self) -> str:
        """Get OAuth token."""
        now = time.time()
        if self._oauth_token and now < self._oauth_token_expires_at:
            return self._oauth_token

        if not self._has_oauth_credentials():
            raise ExtractionError("Reddit OAuth credentials are not configured.")

        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"},
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            timeout=self.request_timeout_seconds,
        )
        if response.status_code >= 400:
            raise ExtractionError(
                f"Failed to authenticate with Reddit (HTTP {response.status_code})."
            )

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise ExtractionError("Reddit OAuth token response did not include an access token.")

        expires_in = int(payload.get("expires_in") or 3600)
        self._oauth_token = str(access_token)
        self._oauth_token_expires_at = now + max(expires_in - 60, 60)
        return self._oauth_token

    def _has_oauth_credentials(self) -> bool:
        """Return whether OAuth credentials."""
        return bool(self.client_id and self.client_secret)

    def _normalize_post(self, post_data: Dict[str, Any], username: str) -> Dict[str, Any]:
        """Normalize post."""
        created_utc = post_data.get("created_utc")
        created_at = self._utc_to_iso(created_utc)

        permalink = post_data.get("permalink")
        absolute_permalink = None
        if permalink:
            absolute_permalink = f"{self.base_url}{permalink}"

        return {
            "provider": self.provider,
            "provider_username": username,
            "post_id": post_data.get("id"),
            "fullname": post_data.get("name"),
            "title": post_data.get("title") or "",
            "selftext": post_data.get("selftext") or "",
            "subreddit": post_data.get("subreddit"),
            "created_utc": created_utc,
            "created_at": created_at,
            "permalink": absolute_permalink,
            "url": post_data.get("url"),
            "score": post_data.get("score"),
            "upvote_ratio": post_data.get("upvote_ratio"),
            "num_comments": post_data.get("num_comments"),
            "is_self": post_data.get("is_self"),
            "is_video": post_data.get("is_video"),
            "over_18": post_data.get("over_18"),
            "link_flair_text": post_data.get("link_flair_text"),
            "domain": post_data.get("domain"),
            "is_gallery": post_data.get("is_gallery"),
            "media_metadata": post_data.get("media_metadata"),
            "preview": post_data.get("preview"),
            "retrieved_from": "submitted",
        }

    def _normalize_comment_payload(
        self,
        payload: Any,
        username: str,
        post_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Normalize comment payload."""
        if not isinstance(payload, list) or len(payload) < 2:
            return []

        comments_listing = payload[1]
        if not isinstance(comments_listing, dict):
            return []
        data = comments_listing.get("data")
        if not isinstance(data, dict):
            return []
        children = data.get("children")
        if not isinstance(children, list):
            return []

        return self._normalize_comments_tree(
            children=children,
            username=username,
            post_id=post_id,
            remaining=self.max_comments_per_post,
        )

    def _normalize_comments_tree(
        self,
        children: List[Dict[str, Any]],
        username: str,
        post_id: Optional[str],
        remaining: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Normalize comments tree."""
        if remaining is not None and remaining <= 0:
            return []

        comments: List[Dict[str, Any]] = []
        for child in children:
            if remaining is not None and len(comments) >= remaining:
                break
            if child.get("kind") != "t1":
                continue
            comment_data = child.get("data", {})
            comments.append(
                self._normalize_comment(
                    comment_data=comment_data,
                    username=username,
                    post_id=post_id,
                )
            )

            replies = comment_data.get("replies")
            if not isinstance(replies, dict):
                continue
            replies_data = replies.get("data")
            if not isinstance(replies_data, dict):
                continue
            replies_children = replies_data.get("children")
            if not isinstance(replies_children, list):
                continue

            next_remaining = None
            if remaining is not None:
                next_remaining = remaining - len(comments)
            comments.extend(
                self._normalize_comments_tree(
                    children=replies_children,
                    username=username,
                    post_id=post_id,
                    remaining=next_remaining,
                )
            )

        return comments

    def _normalize_comment(
        self,
        comment_data: Dict[str, Any],
        username: str,
        post_id: Optional[str],
    ) -> Dict[str, Any]:
        """Normalize comment."""
        created_utc = comment_data.get("created_utc")
        comment_permalink = comment_data.get("permalink")
        absolute_permalink = None
        if comment_permalink:
            absolute_permalink = f"{self.base_url}{comment_permalink}"

        return {
            "provider": self.provider,
            "provider_username": username,
            "post_id": post_id,
            "comment_id": comment_data.get("id"),
            "fullname": comment_data.get("name"),
            "parent_id": comment_data.get("parent_id"),
            "author": comment_data.get("author"),
            "body": comment_data.get("body") or "",
            "subreddit": comment_data.get("subreddit"),
            "score": comment_data.get("score"),
            "created_utc": created_utc,
            "created_at": self._utc_to_iso(created_utc),
            "permalink": absolute_permalink,
            "is_submitter": comment_data.get("is_submitter"),
            "distinguished": comment_data.get("distinguished"),
            "retrieved_from": "comments",
        }

    def _utc_to_iso(self, created_utc: Any) -> Optional[str]:
        """Handle UTC to iso."""
        if created_utc is None:
            return None
        try:
            return datetime.fromtimestamp(
                float(created_utc),
                tz=timezone.utc,
            ).isoformat()
        except (TypeError, ValueError, OSError):
            return None
