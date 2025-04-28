from typing import Callable, List, Optional

from fastapi import APIRouter, status

from app.utils.logging_util import setup_logger


class RouterManager:
    """
    A router manager class that encapsulates common functionalities for FastAPI routers.

    This class provides:
      - A generic method to add routes (GET, POST, etc.)
      - Unified exception handling
      - Logging for errors and standard operations

    Attributes
    ----------
    router : APIRouter
        The FastAPI router instance.
    logger : logging.Logger
        The logger instance for logging information and errors.
    """

    def __init__(self):
        self.router = APIRouter()
        self.logger = setup_logger(__name__)

    def add_route(
            self,
            path: str,
            handler_method: Callable,
            methods: List[str] = None,
            tags: Optional[List[str]] = None,
            status_code: Optional[int] = status.HTTP_200_OK,
    ):
        """
        Adds a route to the APIRouter with unified exception handling.

        Parameters
        ----------
        path : str
            The path for the API endpoint (e.g., "/items").
        handler_method : Callable
            The async function that will handle the request.
        methods : List[str], optional
            A list of HTTP methods (e.g. ["GET", "POST"]). Defaults to ["POST"].
        tags : List[str], optional
            Tags for categorizing the API route. Defaults to an empty list.
        status_code : int, optional
            The HTTP status code returned on success. Defaults to 200 (OK).
        """
        if methods is None:
            methods = ["POST"]  # Default method if none provided
        if tags is None:
            tags = []

        self.router.add_api_route(
            path=path,
            endpoint=handler_method,
            methods=methods,
            tags=tags,
            status_code=status_code
        )
