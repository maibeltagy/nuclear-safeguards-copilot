"""FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from Backend.routes.chat import router as chat_router
from Configuration.settings import settings

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "Frontend"

app = FastAPI(
    title=settings.project.name,
    version=settings.project.version,
    description="RAG copilot for IAEA Nuclear Safeguards documentation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_frontend() -> FileResponse:
    """Serve the Bootstrap chat UI."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.project.name}
