"""API routes for users."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user as get_current_username
from app.schemas.user import UserResponse
from app.repositories.user_repository import UserRepository

from app.api.deps import get_current_user
from app.db.deps import get_db
from app.db.models.user import User
from app.repositories.user_repository import get_latest_personality_run, get_platform_identity
from app.schemas.user_reddit import RedditUsernameIn, SignupInferenceOut, UserMeOut
from app.services.signup_inference_service import (
    SignupQueueError,
    queue_signup_inference_for_user,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users")


 
@router.get("/me/basic", response_model=UserResponse)
async def read_users_me(
    current_username: str = Depends(get_current_username),
    db: AsyncSession = Depends(get_db),
):
    """Read users me."""
    repo = UserRepository(db)
    user = await repo.get_by_username(current_username)
    return user


 
@router.get("/me/pipeline", response_model=UserResponse)
async def get_pipeline_status(
    current_username: str = Depends(get_current_username),
    db: AsyncSession = Depends(get_db),
):
    """Get pipeline status."""
    repo = UserRepository(db)
    user = await repo.get_by_username(current_username)
    return user



@router.post("/me/reddit", response_model=SignupInferenceOut)
async def save_reddit_username_and_run_signup_inference(
    payload: RedditUsernameIn,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SignupInferenceOut | JSONResponse:
    """Save reddit username and run signup inference."""
    try:
        completed, result = await queue_signup_inference_for_user(
            db=session,
            user_id=current_user.id,
            reddit_username=payload.reddit_username,
        )
    except ValueError as exc:
        logger.warning(
            "signup inference rejected user_id=%s reddit_username=%s reason=%s",
            current_user.id,
            payload.reddit_username,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except SignupQueueError as exc:
        logger.exception(
            "signup inference publish failed user_id=%s reddit_username=%s",
            current_user.id,
            payload.reddit_username,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "job_id": str(exc.job_id),
                "detail": "Unable to queue signup inference",
            },
        )

    if completed and "domain" in result:
        return SignupInferenceOut.model_validate(result)

    if completed and result.get("status") == "failed":
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "job_id": str(result["job_id"]),
                "domains": result.get("domains", []),
                "detail": "Unable to compute signup inference right now",
            },
        )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": str(result["job_id"]),
            "status": result["status"],
            "result_url": result.get("result_url"),
            "domains": result.get("domains", []),
        },
    )


@router.get("/me", response_model=UserMeOut)
async def get_me(
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserMeOut:
    """Get me."""
    identity = await get_platform_identity(session, current_user.id, "reddit")
    latest_run = await get_latest_personality_run(session, current_user.id)

    return UserMeOut(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        reddit_username=identity.platform_username if identity else None,
        last_signup_computed_at=latest_run.computed_at if latest_run else None,
    )
