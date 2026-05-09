"""Services for reddit extractor."""
from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from src.extractors.models import SocialAccountLink
from src.extractors.reddit import RedditPostExtractor


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
STATIC_REDDIT_USERNAME = "ok_celery_4705"
STATIC_REDDIT_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "data" / "data.json"


class _VisionCaptioner:
    """Generate vision captions."""
    _model = None
    _processor = None
    _torch = None
    _image = None
    _device = "cpu"
    _available: bool | None = None

    @classmethod
    def generate_caption(cls, image_path: str) -> str | None:
        """Generate caption."""
        if not cls._ensure_loaded():
            return None
        if not os.path.exists(image_path):
            return None
        try:
            image = cls._image.open(image_path).convert("RGB")
            inputs = cls._processor(image, return_tensors="pt").to(cls._device)
            with cls._torch.no_grad():
                output = cls._model.generate(**inputs, max_new_tokens=50)
            return str(cls._processor.decode(output[0], skip_special_tokens=True)).strip() or None
        except Exception as exc:
            logger.warning("Failed to generate media caption for %s: %s", image_path, exc)
            return None

    @classmethod
    def _ensure_loaded(cls) -> bool:
        """Ensure loaded."""
        if cls._available is False:
            return False
        if cls._model is not None and cls._processor is not None:
            return True

        try:
            import torch
            from PIL import Image
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            cls._available = False
            logger.warning("Vision dependencies unavailable; Reddit media summaries will be skipped: %s", exc)
            return False

        try:
            cls._torch = torch
            cls._image = Image
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            cls._model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(cls._device)
            cls._available = True
            return True
        except Exception as exc:
            cls._available = False
            logger.warning("Failed to load vision captioning model; Reddit media summaries will be skipped: %s", exc)
            return False


def extract_reddit_user_bundle(
    reddit_username: str,
    *,
    include_media_summaries: bool = True,
) -> dict[str, Any]:
    """Extract reddit user bundle."""
    if reddit_username.strip().lower() == STATIC_REDDIT_USERNAME:
        return _load_static_reddit_bundle(reddit_username)

    cached_path = Path("artifacts") / "extractions" / reddit_username / "reddit" / "posts.json"
    if cached_path.exists():
        with cached_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    else:
        extractor = RedditPostExtractor(page_delay_seconds=0.0)
        posts = extractor.extract_posts(
            SocialAccountLink(
                provider="reddit",
                external_username=reddit_username,
            )
        )
        payload = {
            "platform_username": reddit_username,
            "provider": "reddit",
            "external_username": reddit_username,
            "profile": {"username": reddit_username},
            "posts_count": len(posts),
            "posts": posts,
        }

    payload.setdefault("profile", {"username": reddit_username})
    payload.setdefault("posts", [])
    payload.setdefault("posts_count", len(payload["posts"]))
    payload.setdefault("external_username", reddit_username)
    payload.setdefault("platform_username", reddit_username)
    if include_media_summaries:
        if _enrich_media_summaries(payload=payload, reddit_username=reddit_username):
            _write_bundle_cache(cached_path=cached_path, payload=payload)
    return payload


def _load_static_reddit_bundle(reddit_username: str) -> dict[str, Any]:
    """Load static reddit bundle."""
    with STATIC_REDDIT_BUNDLE_PATH.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    payload.setdefault("profile", {"username": reddit_username})
    payload.setdefault("posts", [])
    payload.setdefault("posts_count", len(payload["posts"]))
    payload.setdefault("external_username", reddit_username)
    payload.setdefault("platform_username", reddit_username)
    return payload


def _enrich_media_summaries(*, payload: dict[str, Any], reddit_username: str) -> bool:
    """Handle enrich media summaries."""
    posts = payload.get("posts") or []
    if not isinstance(posts, list):
        return False

    changed = False
    media_dir = _media_download_dir(reddit_username)
    for post in posts:
        if not isinstance(post, dict):
            continue
        if _has_media_descriptions(post.get("media_summary")):
            continue

        summaries = []
        for index, url in enumerate(_extract_media_urls(post)):
            local_path = _download_media(
                url=url,
                base_dir=media_dir,
                filename_prefix=f"post_{post.get('post_id') or post.get('id') or index}_{index}",
            )
            if not local_path:
                continue

            description = _caption_media(local_path)
            if not description:
                continue

            summaries.append(
                {
                    "url": url,
                    "file_path": str(local_path),
                    "description": description,
                }
            )

        if summaries:
            post["media_summary"] = summaries
            changed = True

    return changed


def _media_download_dir(reddit_username: str) -> Path:
    """Handle media download dir."""
    assets_dir = Path(os.getenv("ML_ASSETS_DIR") or "assets")
    return assets_dir / "downloads" / reddit_username


def _extract_media_urls(post: dict[str, Any]) -> list[str]:
    """Extract media URLs."""
    urls: list[str] = []
    post_url = str(post.get("url") or "").strip()
    if _is_direct_image_url(post_url):
        urls.append(post_url)

    media_metadata = post.get("media_metadata")
    if isinstance(media_metadata, dict):
        for item in media_metadata.values():
            if not isinstance(item, dict):
                continue
            if item.get("status") != "valid":
                continue
            source = item.get("s")
            if not isinstance(source, dict):
                continue
            image_url = str(source.get("u") or "").strip()
            if image_url:
                urls.append(html.unescape(image_url))

    preview = post.get("preview")
    if isinstance(preview, dict):
        for image in preview.get("images") or []:
            if not isinstance(image, dict):
                continue
            source = image.get("source")
            if isinstance(source, dict):
                image_url = str(source.get("url") or "").strip()
                if image_url:
                    urls.append(html.unescape(image_url))

    return _dedupe(urls)


def _is_direct_image_url(url: str) -> bool:
    """Return whether direct image URL."""
    if not url:
        return False
    suffix = Path(urlsplit(url).path).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def _download_media(*, url: str, base_dir: Path, filename_prefix: str) -> Path | None:
    """Download media."""
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        extension = Path(urlsplit(url).path).suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            extension = ".jpg"

        file_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        filepath = base_dir / f"{_safe_filename(filename_prefix)}_{file_hash}{extension}"
        if filepath.exists():
            return filepath

        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            logger.warning("Failed to download Reddit media %s: HTTP %s", url, response.status_code)
            return None

        with filepath.open("wb") as file_handle:
            for chunk in response.iter_content(1024):
                if chunk:
                    file_handle.write(chunk)
        return filepath
    except Exception as exc:
        logger.warning("Failed to download Reddit media %s: %s", url, exc)
        return None


def _caption_media(path: Path) -> str | None:
    """Caption media."""
    return _VisionCaptioner.generate_caption(str(path))


def _has_media_descriptions(media_summary: Any) -> bool:
    """Return whether media descriptions."""
    if isinstance(media_summary, list):
        return any(
            isinstance(item, dict) and str(item.get("description") or "").strip()
            for item in media_summary
        )
    if isinstance(media_summary, dict):
        return bool(str(media_summary.get("description") or "").strip())
    return bool(str(media_summary or "").strip())


def _safe_filename(value: Any) -> str:
    """Handle safe filename."""
    text = str(value or "media")
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return safe.strip("_") or "media"


def _dedupe(values: list[str]) -> list[str]:
    """Deduplicate the requested data."""
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _write_bundle_cache(*, cached_path: Path, payload: dict[str, Any]) -> None:
    """Write bundle cache."""
    try:
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with cached_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to write enriched Reddit bundle cache %s: %s", cached_path, exc)
