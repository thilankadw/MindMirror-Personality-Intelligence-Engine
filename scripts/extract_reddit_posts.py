"""Utilities for extract reddit posts."""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.extractors import (
    ExtractionRequest,
    InMemorySocialIdentityRegistry,
    JsonFileExtractionStorage,
    PlatformUserIdentity,
    RedditPostExtractor,
    SocialAccountLink,
    SocialExtractionService,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract all Reddit posts for a provider username and store them as JSON "
            "under the platform user's extraction directory."
        )
    )
    parser.add_argument(
        "--platform-username",
        required=True,
        help="The MindMirror platform username.",
    )
    parser.add_argument(
        "--reddit-username",
        required=True,
        help="The Reddit username linked to the platform user.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. Defaults to artifacts/extractions/<platform_username>/reddit/posts.json",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Reddit listing page size (max 100).",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout per Reddit request.",
    )
    parser.add_argument(
        "--page-delay-seconds",
        type=float,
        default=0.2,
        help="Delay between paginated Reddit requests.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Optional safety cap on fetched pages.",
    )
    parser.add_argument(
        "--include-comments",
        dest="include_comments",
        action="store_true",
        help="Include comments for each extracted post (default: enabled).",
    )
    parser.add_argument(
        "--no-comments",
        dest="include_comments",
        action="store_false",
        help="Disable comments extraction and fetch posts only.",
    )
    parser.set_defaults(include_comments=True)
    parser.add_argument(
        "--max-comments-per-post",
        type=int,
        default=25,
        help="Maximum comments to keep per post (set 0 to disable cap).",
    )
    parser.add_argument(
        "--comments-sort",
        default="top",
        help="Reddit comment sort order (e.g., top, new, best).",
    )
    return parser


def main() -> None:
    """Run the module entry point."""
    parser = build_parser()
    args = parser.parse_args()

    registry = InMemorySocialIdentityRegistry()
    registry.register(
        PlatformUserIdentity(
            platform_username=args.platform_username,
            social_accounts=[
                SocialAccountLink(
                    provider="reddit",
                    external_username=args.reddit_username,
                )
            ],
        )
    )

    service = SocialExtractionService(
        extractors=[
            RedditPostExtractor(
                page_size=args.page_size,
                request_timeout_seconds=args.request_timeout_seconds,
                page_delay_seconds=args.page_delay_seconds,
                max_pages=args.max_pages,
                include_comments=args.include_comments,
                max_comments_per_post=(
                    None if args.max_comments_per_post == 0 else args.max_comments_per_post
                ),
                comments_sort=args.comments_sort,
            )
        ],
        identity_registry=registry,
        storage=JsonFileExtractionStorage(),
    )
    result = service.extract_posts(
        ExtractionRequest(
            platform_username=args.platform_username,
            provider="reddit",
            output_path=args.output,
        )
    )

    print(
        json.dumps(
            {
                "platform_username": result.platform_username,
                "provider": result.provider,
                "external_username": result.external_username,
                "posts_count": result.posts_count,
                "output_path": result.output_path,
                "extracted_at": result.extracted_at,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
