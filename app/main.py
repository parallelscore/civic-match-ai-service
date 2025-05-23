import time
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.middleware import register_middlewares
from app.api.routes.server_metrics import ServerMetrics
from app.api.models.model_init import create_all_tables
from app.api.routes.matching_engine import MatchingEngineRouter
from app.api.routes.mock_candidates_response import MockCandidatesResponseRouter


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
    )
    app.state.start_time = time.time()
    app.state.requests_processed = 0

    # Initialize database
    create_all_tables()

    # Register middleware
    register_middlewares(app)

    # Mount static files directory
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Add dashboard route
    @app.get("/dashboard", include_in_schema=False)
    async def get_metrics_dashboard():
        dashboard_path = Path(__file__).parent / "static" / "metrics-dashboard.html"
        return FileResponse(str(dashboard_path))

    # Initialize routers
    server_metrics_router = ServerMetrics(app).router
    matching_engine_router = MatchingEngineRouter().router_manager.router
    mock_candidates_response_router = MockCandidatesResponseRouter().router_manager.router

    # Register the routers
    app.include_router(server_metrics_router)
    app.include_router(matching_engine_router, prefix=settings.API_V1_STR)
    app.include_router(mock_candidates_response_router, prefix=settings.API_V1_STR)

    # Add startup event for loading models
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        print("🚀 Starting CivicMatch Enhanced Matching Engine...")

        # Initialize semantic service (loads embedding model)
        if settings.ENABLE_SEMANTIC_MATCHING:
            try:
                from app.services.semantic_matching_service import semantic_service
                if semantic_service.model:
                    print(f"✅ Semantic matching enabled with model: {settings.EMBEDDING_MODEL}")
                else:
                    print("⚠️  Semantic matching disabled - model failed to load")
            except Exception as e:
                print(f"❌ Semantic service initialization failed: {e}")

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

        print("🎯 Enhanced matching engine ready!")
        print(f"📡 API Documentation: http://localhost:8000/docs")
        print(f"📊 Health Check: http://localhost:8000/api/v1/matching_engine/health")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  # pragma: no cover