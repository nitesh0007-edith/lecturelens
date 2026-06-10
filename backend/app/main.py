from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.ask import router as ask_router
from app.api.recommend import router as recommend_router
from app.workspaces.routes import router as workspaces_router
from app.config import settings
from app.middleware.tracing import setup_trace_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.workspaces.models import create_db_and_tables

    settings.workspace_data_dir.mkdir(parents=True, exist_ok=True)
    settings.bm25_index_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
    create_db_and_tables()
    setup_trace_logging()
    yield


app = FastAPI(
    title="LectureLens",
    description="Hybrid RAG learning copilot for university course materials",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (slowapi) — per workspace token or IP
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except ImportError:
    pass


@app.get("/health")
def health_root():
    return {"status": "ok", "service": "lecturelens"}


app.include_router(health_router, tags=["health"])
app.include_router(ask_router, prefix="/api", tags=["qa"])
app.include_router(recommend_router, prefix="/api", tags=["recommend"])
app.include_router(workspaces_router, prefix="/api/workspaces", tags=["workspaces"])
