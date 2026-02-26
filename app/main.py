import time
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.matching_config import matching_config
from app.core.middleware import register_middlewares
from app.api.routes.server_metrics import ServerMetrics
from app.api.models.model_init import create_all_tables
from app.api.routes.matching_engine import MatchingEngineRouter
from app.api.routes.mock_candidates_response import MockCandidatesResponseRouter


@asynccontextmanager
async def app_lifespan(_apps: FastAPI):
    print("🚀 Starting CivicMatch Policy Matching Engine...")
    print(f"📋 Configuration loaded: {matching_config.max_policy_dimensions} max dimensions")

    # Semantic matching is disabled in V2 — sentence-transformers/PyTorch
    # removed from requirements to reduce image size from ~7.8GB to ~800MB.
    # The LLM pipeline handles context-aware interpretation directly.
    print("ℹ️  Semantic matching: disabled in V2 (LLM handles context-aware matching)")

    # Check LLM service
    if settings.ENABLE_LLM_MATCHING:
        try:
            from app.services.llm_service import llm_service
            if llm_service.client:
                print(f"✅ LLM matching enabled with {settings.LLM_PROVIDER} ({settings.LLM_MODEL})")
            else:
                print("⚠️  LLM matching disabled - no valid API key")
        except Exception as e:
            print(f"❌ LLM service initialization failed: {e}")

    # Check caching service
    try:
        from app.services.caching_service import cache_service
        stats = cache_service.get_cache_stats()
        print(f"✅ Caching enabled: {stats['cache_type']}")
    except Exception as e:
        print(f"❌ Caching service initialization failed: {e}")

    # Initialize policy services
    try:
        from app.services.position_inference_service import position_inference_service
        from app.services.consistency_analyzer_service import consistency_analyzer_service
        from app.services.policy_dimension_discovery_service import policy_dimension_discovery_service
        print("✅ Policy analysis services initialized")
    except Exception as e:
        print(f"❌ Policy services initialization failed: {e}")

    print("🎯 Enhanced policy matching engine ready!")
    print(f"📡 API Documentation: {settings.AI_SERVICE_API_URL}/docs")
    print(f"📊 Health Check: {settings.AI_SERVICE_API_URL}/api/v2/matching_engine/health")
    print(f"⚙️  Max Policy Dimensions: {matching_config.max_policy_dimensions}")
    print(f"🔧 Consistency Analysis: {'Enabled' if matching_config.enable_consistency_analysis else 'Disabled'}")

    yield

def create_app() -> FastAPI:
    apps = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        lifespan=app_lifespan
    )
    apps.state.start_time = time.time()  # type: ignore
    apps.state.requests_processed = 0  # type: ignore

    create_all_tables()
    register_middlewares(apps)

    # Mount static files directory
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
    apps.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Add dashboard route
    @apps.get("/dashboard", include_in_schema=False)
    async def get_metrics_dashboard():
        dashboard_path = Path(__file__).parent / "static" / "metrics-dashboard.html"
        return FileResponse(str(dashboard_path))

    # Initialize routers
    server_metrics_router = ServerMetrics(apps).router
    matching_engine_router = MatchingEngineRouter().router_manager.router
    mock_candidates_response_router = MockCandidatesResponseRouter().router_manager.router

    # Register the routers
    apps.include_router(server_metrics_router)
    apps.include_router(matching_engine_router, prefix=settings.API_V1_STR)
    apps.include_router(mock_candidates_response_router, prefix=settings.API_V1_STR)

    return apps


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  # pragma: no cover
