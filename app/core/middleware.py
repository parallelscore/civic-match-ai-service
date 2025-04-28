from fastapi.middleware.cors import CORSMiddleware


def register_middlewares(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_credentials=True,
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def count_requests(request, call_next):
        response = await call_next(request)
        app.state.requests_processed += 1
        return response
