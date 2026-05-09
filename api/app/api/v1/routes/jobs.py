"""API routes for jobs."""
from __future__ import annotations
import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.deps import get_db
from app.core.security import get_current_user as get_current_username
from app.schemas.job import JobResponse
from app.repositories.user_repository import UserRepository
from app.services.core.extraction import ExtractionService
from app.services.core.vision import VisionService

from app.api.deps import get_current_user
 
from app.db.models.user import User
from app.schemas.job import InferenceJobFailedOut, InferenceJobQueuedOut, PersonalityJobResultOut
from app.services.job_service import get_job_status_payload


router = APIRouter(prefix="/inference")

@router.post("/extract", response_model=JobResponse)
async def trigger_extraction(
    background_tasks: BackgroundTasks,
    current_username: str = Depends(get_current_username),
    db: AsyncSession = Depends(get_db),
):

    """Trigger extraction."""
    repo = UserRepository(db)
    service = ExtractionService(repo, db)

    job_uuid = str(uuid.uuid4())

    result = await service.extract_user_data(current_username)

    return {
        "job_id": job_uuid,
        "status": result["status"],
        "result": result,
    }


 
@router.post("/caption", response_model=JobResponse)
async def trigger_captioning(
    current_username: str = Depends(get_current_username),
    db: AsyncSession = Depends(get_db),
):

    """Trigger captioning."""
    user_repo = UserRepository(db)
    service = VisionService()

    result = await service.process_user_media(current_username, user_repo, db)

    job_uuid = str(uuid.uuid4())

    return {
        "job_id": job_uuid,
        "status": "completed" if "error" not in result else "failed",
        "result": result,
    }



@router.get(
    "/jobs/{job_id}",
    response_model=PersonalityJobResultOut | InferenceJobFailedOut | InferenceJobQueuedOut,
)
async def get_inference_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PersonalityJobResultOut | InferenceJobFailedOut | InferenceJobQueuedOut:
    """Get inference job."""
    payload = await get_job_status_payload(
        session,
        user_id=current_user.id,
        job_id=job_id,
    )
    if "domain" in payload:
        return PersonalityJobResultOut.model_validate(payload)
    if payload["status"] == "failed":
        return InferenceJobFailedOut.model_validate(payload)
    return InferenceJobQueuedOut.model_validate(payload)
