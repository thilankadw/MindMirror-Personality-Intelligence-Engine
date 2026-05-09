"""API routes for auth."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.deps import get_db
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    LoginRequest,
    RedditOAuthAuthorizeIn,
    RedditOAuthAuthorizeOut,
    RedditOAuthExchangeIn,
    RedditOAuthToken,
    Token,
)
from app.schemas.user import UserCreate
from app.services.auth_service import (
    authenticate_user,
    create_user,
    get_user_by_email,
    issue_access_token,
)
from app.services.core.extraction import ExtractionService
from app.services.reddit_oauth_service import (
    RedditOAuthError,
    build_reddit_authorization_payload,
    complete_reddit_oauth,
)
from app.services.signup_inference_service import SignupQueueError, queue_signup_inference_for_user
from app.utils.security import create_access_token

router = APIRouter(prefix="/auth")
logger = logging.getLogger(__name__)


@router.post("/token", response_model=Token)
async def login_for_access_token(
    background_tasks: BackgroundTasks,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Log in for access token."""
    repo = UserRepository(db)
    user = await repo.get_by_username(form_data.username)

    if not user:
        user = await repo.create(form_data.username)

    if not user.personal_data_present:
        service = ExtractionService(repo, db)
        background_tasks.add_task(service.extract_user_data, user.username)

    access_token = create_access_token(subject=user.username)
    return Token(
        access_token=access_token,
        token_type="bearer",
        reddit_username=user.reddit_username or user.username,
    )


@router.post("/register", response_model=Token)
async def register_user(
    background_tasks: BackgroundTasks,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
    reddit_username: str | None = None,
) -> Token:
    """Register user."""
    repo = UserRepository(db)

    existing = await repo.get_by_username(form_data.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )

    if reddit_username:
        existing_reddit = await repo.get_by_reddit_username(reddit_username.strip().lower())
        if existing_reddit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This Reddit username is already registered",
            )

    user = await repo.create(
        username=form_data.username,
        reddit_username=reddit_username,
    )

    if not user.personal_data_present:
        service = ExtractionService(repo, db)
        background_tasks.add_task(service.extract_user_data, user.username)

    access_token = create_access_token(subject=user.username)
    return Token(
        access_token=access_token,
        token_type="bearer",
        reddit_username=user.reddit_username or user.username,
    )


@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserCreate, session: AsyncSession = Depends(get_db)) -> Token:
    """Sign up the requested data."""
    logger.info("auth.signup.request.start email=%s reddit_username=%s", payload.email, payload.reddit_username)
    existing = await get_user_by_email(session, payload.email)
    if existing:
        logger.warning("auth.signup.request.duplicate_email email=%s", payload.email)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    if payload.reddit_username:
        existing_reddit = await UserRepository(session).get_by_reddit_username(payload.reddit_username)
        if existing_reddit:
            logger.warning(
                "auth.signup.request.duplicate_reddit reddit_username=%s",
                payload.reddit_username,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Reddit username already registered",
            )
    user = await create_user(session, payload)
    logger.info("auth.signup.user.created user_id=%s username=%s", user.id, user.username)

    reddit_username = payload.reddit_username or user.reddit_username or user.username
    if reddit_username:
        try:
            logger.info(
                "auth.signup.inference.queue.start user_id=%s reddit_username=%s",
                user.id,
                reddit_username,
            )
            await queue_signup_inference_for_user(
                db=session,
                user_id=user.id,
                reddit_username=reddit_username,
                wait_for_completion=False,
            )
            logger.info(
                "auth.signup.inference.queue.done user_id=%s reddit_username=%s",
                user.id,
                reddit_username,
            )
        except (SignupQueueError, ValueError) as exc:
            logger.warning(
                "auth.signup.inference.queue.failed user_id=%s reddit_username=%s reason=%s",
                user.id,
                reddit_username,
                exc,
            )
    else:
        logger.warning("auth.signup.inference.queue.skipped user_id=%s reason=no_reddit_username", user.id)
    return Token(
        access_token=issue_access_token(str(user.id)),
        token_type="bearer",
        reddit_username=user.reddit_username,
    )


@router.post("/signin", response_model=Token)
async def signin(payload: LoginRequest, session: AsyncSession = Depends(get_db)) -> Token:
    """Sign in the requested data."""
    user = await authenticate_user(session, payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    access_token = issue_access_token(str(user.id))
    return Token(
        access_token=access_token,
        token_type="bearer",
        reddit_username=user.reddit_username,
    )


@router.post("/reddit/authorize", response_model=RedditOAuthAuthorizeOut)
async def reddit_authorize(payload: RedditOAuthAuthorizeIn) -> RedditOAuthAuthorizeOut:
    """Handle reddit authorize."""
    try:
        auth_payload = build_reddit_authorization_payload(
            redirect_uri=payload.redirect_uri,
            intent=payload.intent,
        )
    except RedditOAuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return RedditOAuthAuthorizeOut.model_validate(auth_payload)


@router.post("/reddit/signup", response_model=RedditOAuthToken)
async def reddit_signup(
    payload: RedditOAuthExchangeIn,
    session: AsyncSession = Depends(get_db),
) -> RedditOAuthToken:
    """Handle reddit signup."""
    logger.info("auth.reddit_signup.request.start redirect_uri=%s", payload.redirect_uri)
    try:
        result = await complete_reddit_oauth(
            session=session,
            code=payload.code,
            state=payload.state,
            redirect_uri=payload.redirect_uri,
            intent="signup",
        )
        logger.info(
            "auth.reddit_signup.oauth.done user_id=%s reddit_username=%s is_new_user=%s",
            result.user.id,
            result.reddit_username,
            result.is_new_user,
        )
    except RedditOAuthError as exc:
        logger.warning("auth.reddit_signup.oauth.failed status_code=%s reason=%s", exc.status_code, exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    try:
        logger.info(
            "auth.reddit_signup.inference.queue.start user_id=%s reddit_username=%s",
            result.user.id,
            result.reddit_username,
        )
        await queue_signup_inference_for_user(
            db=session,
            user_id=result.user.id,
            reddit_username=result.reddit_username,
            wait_for_completion=False,
        )
        logger.info(
            "auth.reddit_signup.inference.queue.done user_id=%s reddit_username=%s",
            result.user.id,
            result.reddit_username,
        )
    except (SignupQueueError, ValueError) as exc:
        logger.warning(
            "reddit signup inference trigger failed user_id=%s reddit_username=%s reason=%s",
            result.user.id,
            result.reddit_username,
            exc,
        )

    return RedditOAuthToken(
        access_token=issue_access_token(str(result.user.id)),
        token_type="bearer",
        reddit_username=result.reddit_username,
        is_new_user=result.is_new_user,
    )


# @router.post("/reddit/signin", response_model=RedditOAuthToken)
# async def reddit_signin(
#     payload: RedditOAuthExchangeIn,
#     session: AsyncSession = Depends(get_db),
# ) -> RedditOAuthToken:
#     try:
#         result = await complete_reddit_oauth(
#             session=session,
#             code=payload.code,
#             state=payload.state,
#             redirect_uri=payload.redirect_uri,
#             intent="signin",
#         )
#     except RedditOAuthError as exc:
#         raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

#     return RedditOAuthToken(
#         access_token=issue_access_token(str(result.user.id)),
#         token_type="bearer",
#         reddit_username=result.reddit_username,
#         is_new_user=result.is_new_user,
#     )
