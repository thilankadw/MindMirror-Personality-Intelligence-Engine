import json

from src.extractors import (
    ExtractionRequest,
    InMemorySocialIdentityRegistry,
    JsonFileExtractionStorage,
    PlatformUserIdentity,
    RedditPostExtractor,
    SocialAccountLink,
    SocialExtractionService,
)


def test_social_extraction_service_uses_platform_mapping(tmp_path, monkeypatch):
    registry = InMemorySocialIdentityRegistry()
    registry.register(
        PlatformUserIdentity(
            platform_username="mindmirror_user",
            social_accounts=[
                SocialAccountLink(provider="reddit", external_username="reddit_handle")
            ],
        )
    )

    extractor = RedditPostExtractor(page_delay_seconds=0.0)

    def fake_extract_posts(account):
        return [
            {
                "provider": "reddit",
                "provider_username": account.external_username,
                "post_id": "abc123",
                "title": "Example",
            }
        ]

    monkeypatch.setattr(extractor, "extract_posts", fake_extract_posts)

    service = SocialExtractionService(
        extractors=[extractor],
        identity_registry=registry,
        storage=JsonFileExtractionStorage(base_dir=str(tmp_path)),
    )
    result = service.extract_posts(
        ExtractionRequest(
            platform_username="mindmirror_user",
            provider="reddit",
        )
    )

    assert result.external_username == "reddit_handle"
    assert result.posts_count == 1

    with open(result.output_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    assert payload["platform_username"] == "mindmirror_user"
    assert payload["external_username"] == "reddit_handle"
    assert payload["posts"][0]["post_id"] == "abc123"


def test_reddit_post_extractor_paginates_and_normalizes(monkeypatch):
    extractor = RedditPostExtractor(page_delay_seconds=0.0)
    responses = [
        {
            "data": {
                "after": "t3_next",
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "one",
                            "name": "t3_one",
                            "title": "First",
                            "selftext": "Body 1",
                            "subreddit": "test",
                            "created_utc": 1,
                            "permalink": "/r/test/comments/one",
                            "url": "https://reddit.com/r/test/comments/one",
                            "score": 10,
                            "upvote_ratio": 0.9,
                            "num_comments": 2,
                            "is_self": True,
                            "is_video": False,
                            "over_18": False,
                            "link_flair_text": None,
                            "domain": "self.test",
                        },
                    }
                ],
            }
        },
        {
            "data": {
                "after": None,
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "two",
                            "name": "t3_two",
                            "title": "Second",
                            "selftext": "",
                            "subreddit": "test",
                            "created_utc": 2,
                            "permalink": "/r/test/comments/two",
                            "url": "https://reddit.com/r/test/comments/two",
                            "score": 20,
                            "upvote_ratio": 0.8,
                            "num_comments": 3,
                            "is_self": True,
                            "is_video": False,
                            "over_18": False,
                            "link_flair_text": "flair",
                            "domain": "self.test",
                        },
                    }
                ],
            }
        },
    ]

    def fake_fetch_listing_page(username, after=None):
        if after is None:
            return responses[0]
        return responses[1]

    def fake_fetch_post_comments(username, post_data):
        return [
            {
                "provider": "reddit",
                "provider_username": username,
                "post_id": post_data.get("id"),
                "comment_id": f"c_{post_data.get('id')}",
                "body": "Example comment",
            }
        ]

    monkeypatch.setattr(extractor, "_fetch_listing_page", fake_fetch_listing_page)
    monkeypatch.setattr(extractor, "_fetch_post_comments", fake_fetch_post_comments)

    posts = extractor.extract_posts(
        SocialAccountLink(provider="reddit", external_username="reddit_handle")
    )

    assert [post["post_id"] for post in posts] == ["one", "two"]
    assert posts[0]["provider_username"] == "reddit_handle"
    assert posts[0]["permalink"] == "https://www.reddit.com/r/test/comments/one"
    assert posts[0]["comments"][0]["comment_id"] == "c_one"
    assert posts[1]["comments_extracted_count"] == 1


def test_reddit_post_extractor_comment_normalization_flattens_with_cap():
    extractor = RedditPostExtractor(page_delay_seconds=0.0, max_comments_per_post=2)
    payload = [
        {},
        {
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "id": "c1",
                            "name": "t1_c1",
                            "parent_id": "t3_one",
                            "author": "author_1",
                            "body": "Root",
                            "subreddit": "test",
                            "score": 5,
                            "created_utc": 1,
                            "permalink": "/r/test/comments/one/x/c1/",
                            "is_submitter": False,
                            "distinguished": None,
                            "replies": {
                                "data": {
                                    "children": [
                                        {
                                            "kind": "t1",
                                            "data": {
                                                "id": "c2",
                                                "name": "t1_c2",
                                                "parent_id": "t1_c1",
                                                "author": "author_2",
                                                "body": "Reply",
                                                "subreddit": "test",
                                                "score": 3,
                                                "created_utc": 2,
                                                "permalink": "/r/test/comments/one/x/c2/",
                                                "is_submitter": False,
                                                "distinguished": None,
                                            },
                                        }
                                    ]
                                }
                            },
                        },
                    },
                    {
                        "kind": "t1",
                        "data": {
                            "id": "c3",
                            "name": "t1_c3",
                            "parent_id": "t3_one",
                            "author": "author_3",
                            "body": "Second root",
                            "subreddit": "test",
                            "score": 1,
                            "created_utc": 3,
                            "permalink": "/r/test/comments/one/x/c3/",
                            "is_submitter": False,
                            "distinguished": None,
                        },
                    },
                ]
            }
        },
    ]

    comments = extractor._normalize_comment_payload(
        payload=payload,
        username="reddit_handle",
        post_id="one",
    )

    assert [comment["comment_id"] for comment in comments] == ["c1", "c2"]
