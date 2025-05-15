import time

import uvicorn
from fastapi import FastAPI

from app.api.models.model_init import create_all_tables
from app.api.routes.matching_engine import MatchingEngineRouter
from app.api.routes.mock_candidates_response import MockCandidatesResponseRouter
from app.api.routes.server_metrics import ServerMetrics
from app.core.config import settings
from app.core.middleware import register_middlewares


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

    server_metrics_router = ServerMetrics(app).router

    matching_engine_router = MatchingEngineRouter().router_manager.router
    mock_candidates_response_router = MockCandidatesResponseRouter().router_manager.router

    # Register the routers
    app.include_router(server_metrics_router)

    app.include_router(matching_engine_router, prefix=settings.API_V1_STR)
    app.include_router(mock_candidates_response_router, prefix=settings.API_V1_STR)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)  # pragma: no cover
