"""Shared test fixtures."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI test client with model loaded."""
    with TestClient(app) as c:
        yield c
