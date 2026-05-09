"""Services for extraction."""
import os
import logging
import asyncio
import concurrent.futures
import requests
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.user import User
from app.db.models.post import Post
from app.db.models.comment import Comment
from app.db.models.media import Media
from app.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


class ExtractionService:
    """Provide extraction services."""
    def __init__(self, user_repo: UserRepository, session: AsyncSession):
        """Initialize the extraction service."""
        self.user_repo = user_repo
        self.session = session
        try:
            import praw
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing optional dependency 'praw'. Install it to use extraction endpoints."
            ) from exc
        self.reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT
        )

    def _timestamp_to_str(self, ts: float) -> str:
        """Handle timestamp to str."""
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ""

    def download_media(self, url: str, base_dir: str, filename_prefix: str) -> Optional[str]:
        """Download media."""
        try:
            os.makedirs(base_dir, exist_ok=True)
            ext = os.path.splitext(url.split('?')[0])[1]
            if not ext:
                ext = '.jpg'
            
            filepath = os.path.join(base_dir, f"{filename_prefix}{ext}")
            
            if os.path.exists(filepath):
                return filepath
                
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return filepath
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
        return None

    async def extract_user_data(self, username: str, posts_limit: Optional[int] = None, comments_limit: Optional[int] = None):
        """Extract user data."""
        one_year_ago_ts = (datetime.utcnow() - timedelta(days=365)).timestamp()
        now = datetime.utcnow()

        # ensure user exists
        db_user = await self.user_repo.get_by_username(username)
        if not db_user:
            logger.error(f"User {username} not found for extraction")
            return

        db_user.pipeline_step = "GATHERING_DATA"
        db_user.pipeline_progress = 0.05
        db_user.pipeline_message = "Connecting to Reddit API..."
        await self.session.commit()

        # Identity vs Data Source
        reddit_id = db_user.reddit_username or username
        logger.info(f"Starting extraction for display user: {username} using Reddit ID: {reddit_id}")

        # get async session
        db = self.session

        redditor = self.reddit.redditor(reddit_id)

        # verify Reddit user
        try:
            _ = redditor.comment_karma
        except Exception as e:
            logger.error(f"Error fetching Reddit user {username}: {e}")
            return {
                "username": username,
                "posts_added": 0,
                "comments_added": 0,
                "status": "error",
                "error": str(e),
            }

        # ext posts
        logger.info(f"Extracting posts for {username}...")
        db_user.pipeline_message = "Scanning Reddit posts..."
        db_user.pipeline_progress = 0.08
        await db.commit()

        new_posts_count = 0
        skipped_posts = 0
        media_tasks = []
        oldest_post_date = None
        user_media_dir = os.path.join(settings.ML_ASSETS_DIR, "downloads", username)

        # run blocking PRAW thread
        submissions = await asyncio.to_thread(
            lambda: list(redditor.submissions.new(limit=posts_limit))
        )
        total_submissions = len(submissions)

        for idx, s in enumerate(submissions):
            if s.created_utc < one_year_ago_ts:
                break

            # track date range
            post_date = datetime.fromtimestamp(s.created_utc)
            if oldest_post_date is None or post_date < oldest_post_date:
                oldest_post_date = post_date
            days_scanned = (now - oldest_post_date).days if oldest_post_date else 0

            # check duplicate
            existing = await db.execute(
                select(Post).where(Post.reddit_id == s.id)
            )
            if existing.scalars().first():
                skipped_posts += 1
                continue

            new_post = Post(
                reddit_id=s.id,
                author_username=username,
                title=s.title,
                selftext=s.selftext,
                subreddit=str(s.subreddit),
                score=float(s.score),
                upvote_ratio=float(s.upvote_ratio),
                num_comments=float(s.num_comments),
                created_utc=s.created_utc,
                created_date=self._timestamp_to_str(s.created_utc),
                url=s.url,
                permalink=f"https://reddit.com{s.permalink}",
                is_self=s.is_self,
                is_video=getattr(s, "is_video", False),
                is_nsfw=s.over_18,
                is_spoiler=getattr(s, "spoiler", False),
                stickied=s.stickied,
                is_original_content=s.is_original_content,
            )
            db.add(new_post)
            # flush for UUID
            await db.flush()
            new_posts_count += 1

            # check media in post
            if not s.is_self:
                url_lower = s.url.lower()
                if any(url_lower.endswith(x) for x in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
                    logger.info(f"Found image: {s.url}")
                    media_tasks.append((s.url, user_media_dir, f"post_{s.id}", new_post.id))
                
                elif hasattr(s, 'is_gallery') and s.is_gallery:
                    logger.info(f"Found gallery in post {s.id}")
                    if hasattr(s, 'media_metadata'):
                        for i, (item_id, item) in enumerate(s.media_metadata.items()):
                            if item['status'] == 'valid' and 's' in item:
                                img_url = item['s']['u'].replace('&amp;', '&')
                                media_tasks.append((img_url, user_media_dir, f"post_{s.id}_img_{i}", new_post.id))

            # Live progress update every 5 posts
            if new_posts_count % 5 == 0 or idx == len(submissions) - 1:
                progress = 0.08 + (0.17 * min((idx + 1) / max(total_submissions, 1), 1.0))
                db_user.pipeline_progress = round(progress, 2)
                db_user.pipeline_message = f"Scanned {new_posts_count} posts across {days_scanned} days (of last 365)..."
                await db.commit()

        # Progress after posts
        days_covered = (now - oldest_post_date).days if oldest_post_date else 0
        db_user.pipeline_progress = 0.28
        db_user.pipeline_message = f"Found {new_posts_count} posts across {days_covered} days. Scanning comments..."
        await db.commit()

        # concurrent media dl
        download_results = []
        if media_tasks:
            logger.info(f"Downloading {len(media_tasks)} media files...")
            db_user.pipeline_progress = 0.52
            db_user.pipeline_message = f"Downloading {len(media_tasks)} media files..."
            await db.commit()
            
            def download_all(tasks):
                """Download all queued media assets in a thread pool."""
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(self.download_media, t[0], t[1], t[2]): t[3] for t in tasks}
                    for f in concurrent.futures.as_completed(futures):
                        post_db_id = futures[f]
                        path = f.result()
                        results.append((post_db_id, path))
                return results

            download_results = await asyncio.to_thread(download_all, media_tasks)
            
            for post_db_id, path in download_results:
                if path:
                    logger.info(f"Saved media to {path}")
                    db.add(Media(post_id=post_db_id, file_path=path, media_type="image"))

            saved_media = sum(1 for _, path in download_results if path)
            db_user.pipeline_progress = 0.58
            db_user.pipeline_message = f"Downloaded {saved_media}/{len(media_tasks)} media files. Preparing analysis..."
            await db.commit()

        # ext comments
        logger.info(f"Extracting comments for {username}...")
        new_comments_count = 0

        comments = await asyncio.to_thread(
            lambda: list(redditor.comments.new(limit=comments_limit))
        )
        total_comments = len(comments)

        for idx, c in enumerate(comments):
            if c.created_utc < one_year_ago_ts:
                break

            existing = await db.execute(
                select(Comment).where(Comment.comment_id == c.id)
            )
            if existing.scalars().first():
                continue

            new_comment = Comment(
                comment_id=c.id,
                author_username=username,
                body=c.body,
                subreddit=str(c.subreddit),
                link_id=c.link_id,
                parent_id=c.parent_id,
                score=float(c.score),
                created_utc=c.created_utc,
                created_date=self._timestamp_to_str(c.created_utc),
                permalink=f"https://reddit.com{c.permalink}",
            )
            db.add(new_comment)
            new_comments_count += 1

            # Live progress update every 5 comments
            if new_comments_count % 5 == 0 or idx == len(comments) - 1:
                progress = 0.28 + (0.17 * min((idx + 1) / max(total_comments, 1), 1.0))
                db_user.pipeline_progress = round(progress, 2)
                db_user.pipeline_message = f"Processing comments: {new_comments_count} found so far..."
                await db.commit()

        if new_posts_count == 0 and new_comments_count == 0:
            db_user.pipeline_step = "COMPLETED_NO_DATA"
            db_user.pipeline_progress = 1.0
            db_user.pipeline_message = "No public Reddit activity found in the last year. Please ensure your account has public posts or comments."
            await db.commit()
            logger.warning(f"No data found for {username}")
            return {
                "username": username,
                "posts_added": 0,
                "comments_added": 0,
                "status": "completed_no_data",
            }

        # Extraction is the personality-domain terminal step in this repository copy.
        db_user.personal_data_present = True
        db_user.personal_model_training = False
        db_user.pipeline_step = "COMPLETED"
        db_user.pipeline_progress = 1.0
        db_user.pipeline_message = (
            f"Extracted {new_posts_count} posts and {new_comments_count} comments. "
            "Personality inference is ready."
        )
        await db.commit()

        logger.info(
            f"Extraction complete for {username}: "
            f"{new_posts_count} posts, {new_comments_count} comments"
            f", {len([r for r in download_results if r[1]]) if download_results else 0} media files"
        )

        return {
            "username": username,
            "posts_added": new_posts_count,
            "comments_added": new_comments_count,
            "media_downloaded": sum(1 for _, path in download_results if path) if media_tasks else 0,
            "status": "completed",
        }
