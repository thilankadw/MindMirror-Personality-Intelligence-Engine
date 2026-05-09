"""Services for vision."""
# app/services/vision.py
import os
import hashlib
import logging
import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User
from app.db.models.post import Post
from app.db.models.media import Media

logger = logging.getLogger(__name__)

try:
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("Vision dependencies (torch, transformers, PIL) not available. Vision endpoints will return errors.")


class VisionService:
    """Provide vision services."""
    _instance = None
    _model = None
    _processor = None

    def __init__(self, device: str = None):
        """Initialize the vision service."""
        if not VISION_AVAILABLE:
            self.device = "cpu"
            return
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if VisionService._model is None:
            VisionService._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            VisionService._model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def _get_file_hash(self, filepath):
        """Get file hash."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def generate_caption(self, image_path: str) -> str:
        """Generate caption."""
        if not VISION_AVAILABLE:
            return None
        if not os.path.exists(image_path):
            return None
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=50)
            return self._processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return None

    async def process_user_media(self, username: str, user_repo, session: AsyncSession):
        """Process user media."""
        if not VISION_AVAILABLE:
            return {"error": "Vision service unavailable (torch not installed)"}

        db_user = await user_repo.get_by_username(username)
        if not db_user:
            return {"error": "User not found"}

        target_result = await session.execute(
            select(Media)
            .join(Post)
            .where(Post.author_username == username, Media.caption.is_(None))
        )
        target_media = target_result.scalars().all()

        captioned = 0
        reused = 0
        for media in target_media:
            if not os.path.exists(media.file_path):
                continue
                
            f_hash = await asyncio.to_thread(self._get_file_hash, media.file_path)
            media.file_hash = f_hash
            
            existing = await session.execute(
                select(Media.caption).where(
                    Media.file_hash == f_hash,
                    Media.caption.isnot(None),
                )
            )
            existing_row = existing.first()
            if existing_row:
                media.caption = existing_row[0]
                reused += 1
            else:
                caption = await asyncio.to_thread(self.generate_caption, media.file_path)
                if caption:
                    media.caption = caption
                    captioned += 1
            
            await session.commit()

        return {"captioned": captioned, "reused": reused}
