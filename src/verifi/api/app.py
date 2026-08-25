"""FastAPI application with pipeline lifecycle management."""
from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from verifi.api.routes.analysis import router as analysis_router
from verifi.api.schemas import HealthResponse
from verifi.config import AppConfig
from verifi.pipeline.orchestrator import VeriFiPipeline

logger = structlog.get_logger()

_pipeline: VeriFiPipeline | None = None
_config: AppConfig | None = None


def get_pipeline() -> VeriFiPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized — server still starting?")
    return _pipeline


def get_config() -> AppConfig:
    if _config is None:
        return AppConfig()
    return _config


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _config
    logger.info("server_starting")
    _config = AppConfig()
    _pipeline = VeriFiPipeline(_config)
    _pipeline.load_models()
    logger.info("server_ready")
    yield
    logger.info("server_stopping")
    if _pipeline is not None:
        _pipeline.unload_models()
        _pipeline = None
    logger.info("server_stopped")


app = FastAPI(
    title="VeriFi",
    description="Forensic AI-generated video detection API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _pipeline and _pipeline._models_loaded else "loading",
        models_loaded=_pipeline._models_loaded if _pipeline else False,
        device=_config.device.resolve() if _config else "unknown",
    )
