"""
main.py
-------
FastAPI application entry point for the Socratic AI dialogue system.

Endpoints:
  POST /api/chat       — Send a message, get a Socratic response
  POST /api/reset      — Clear conversation history
  GET  /api/health     — Health check (for Vercel / uptime monitors)
  GET  /               — Serves the frontend index.html
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gemini_client import (
    ConversationHistory,
    GeminiAPIError,
    GeminiClient,
    GeminiRateLimitError,
    GeminiSafetyError,
)
from nlp_utils import preprocess

# Logging setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load env + initialise Gemini client (once at startup)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# In-memory session store: session_id → ConversationHistory
# For production you'd use Redis; for this project, in-process is fine.
_sessions: dict[str, ConversationHistory] = {}
_gemini: Optional[GeminiClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _gemini
    logger.info("Starting Socratic AI backend…")
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set — /api/chat will return errors.")
    else:
        _gemini = GeminiClient(api_key=GEMINI_API_KEY)
        logger.info("Gemini client ready.")
    yield
    logger.info("Shutting down.")

# App definition

app = FastAPI(
    title="Socratic AI",
    description="A Socratic dialogue engine powered by Google Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend origin (update for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend static files from the same directory (flat structure)
FRONTEND_DIR = Path(__file__).parent

# Pydantic models

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    response: str
    suggestions: list[str] = []
    session_id: str
    model_used: str = "gemini-2.0-flash"
    turn_count: int


class ResetRequest(BaseModel):
    session_id: str


class ErrorDetail(BaseModel):
    error: str
    error_type: str
    retry_after: Optional[int] = None


# Helper

def get_or_create_session(session_id: Optional[str]) -> tuple[str, ConversationHistory]:
    """Return existing session or create a new one."""
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    _sessions[new_id] = ConversationHistory()
    # Limit total sessions in memory (LRU-lite: drop oldest when > 500)
    if len(_sessions) > 500:
        oldest = next(iter(_sessions))
        del _sessions[oldest]
    return new_id, _sessions[new_id]


# Routes

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main HTML page."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Socratic AI API is running. Frontend not found."})


@app.get("/api/health")
async def health():
    """Health check — used by Vercel and uptime monitors."""
    return {
        "status": "ok",
        "gemini_ready": _gemini is not None,
        "active_sessions": len(_sessions),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """
    Main dialogue endpoint.

    1. Validate input
    2. Run NLP preprocessing
    3. Send to Gemini with conversation history
    4. Return Socratic response + session metadata
    """
    if _gemini is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                error="GEMINI_API_KEY is not configured on the server.",
                error_type="configuration_error",
            ).model_dump(),
        )

    # 1. Validate
    message = req.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorDetail(
                error="Message cannot be empty.",
                error_type="validation_error",
            ).model_dump(),
        )

    # 2. Session management
    session_id, history = get_or_create_session(req.session_id)

    # 3. NLP preprocessing
    try:
        preprocessed = preprocess(message)
        logger.info(
            "Session=%s | words=%d | concepts=%s | sentiment=%s | question=%s",
            session_id[:8],
            preprocessed.word_count,
            preprocessed.key_concepts,
            preprocessed.sentiment,
            preprocessed.is_question,
        )
    except Exception as e:
        logger.error("NLP preprocessing failed: %s", e)
        # Don't fail the whole request — fall back to raw message
        from nlp_utils import PreprocessedInput
        preprocessed = PreprocessedInput(
            original=message,
            cleaned=message.lower(),
            lemmatized=message,
            key_concepts=[],
            sentiment="neutral",
            is_question=message.strip().endswith("?"),
            word_count=len(message.split()),
        )

    # 4. Gemini call
    try:
        response_text, suggestions = _gemini.get_response(preprocessed, history)
        return ChatResponse(
            response=response_text,
            suggestions=suggestions,
            session_id=session_id,
            turn_count=len(history.turns) // 2,
        )

    except GeminiRateLimitError as e:
        logger.warning("Rate limit hit for session %s", session_id[:8])
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(e.retry_after or 30)},
            detail=ErrorDetail(
                error="The AI is thinking deeply and needs a moment. Please wait a few seconds and try again.",
                error_type="rate_limit",
                retry_after=e.retry_after or 30,
            ).model_dump(),
        )

    except GeminiSafetyError as e:
        logger.warning("Safety filter triggered for session %s: %s", session_id[:8], e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                error="Your message touched on a topic I cannot explore. Perhaps we could approach it from a different angle?",
                error_type="safety_filter",
            ).model_dump(),
        )

    except GeminiAPIError as e:
        logger.error("Gemini API error for session %s: %s", session_id[:8], e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=ErrorDetail(
                error="The AI encountered an unexpected issue. Please try again.",
                error_type="api_error",
            ).model_dump(),
        )


@app.post("/api/reset")
async def reset_session(req: ResetRequest):
    """Clear conversation history for a session."""
    if req.session_id in _sessions:
        _sessions[req.session_id].clear()
        logger.info("Session %s reset.", req.session_id[:8])
    return {"status": "ok", "message": "Conversation reset."}


# Global exception handler (catch-all)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(
            error="An unexpected server error occurred.",
            error_type="internal_error",
        ).model_dump(),
    )


# Static file serving — mounted LAST so API routes take priority
# Serves style.css, app.js, etc. at their root paths (e.g. /style.css)

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# Dev entry point

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
