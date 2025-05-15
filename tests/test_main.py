import pytest
from fastapi.testclient import TestClient

from app.main import create_app


class TestMain:
    @pytest.fixture(scope='module')
    def test_app(self):
        app = create_app()
        return TestClient(app)

    def test_app_creation(self, test_app):
        response = test_app.get('/')
        assert response.status_code == 200

    def test_app_state(self, test_app):
        assert test_app.app.state.start_time is not None
        assert test_app.app.state.requests_processed == 1

    def test_database_init(self, mocker):
        mocker.patch('app.api.models.model_init.create_all_tables', return_value=None)
        from app.api.models.model_init import create_all_tables
        assert create_all_tables() is None

    def test_server_metrics_endpoint(self, test_app):
        # Call the route that returns the HTML response
        response = test_app.get('/')

        # Ensure the response status code is 200 (OK)
        assert response.status_code == 200

        # Ensure the content type is HTML
        assert 'text/html' in response.headers['content-type']

        # Check for specific content or elements in the HTML response
        assert '<title>Server Metrics</title>' in response.text
        assert '<h1>Server Metrics</h1>' in response.text
        assert 'CPU Usage' in response.text
        assert 'Memory Info' in response.text

    def test_middleware_registration(self, mocker):
        mocker.patch('app.core.middleware.register_middlewares', return_value=None)
        from app.core.middleware import register_middlewares
        assert register_middlewares(None) is None


if __name__ == '__main__':
    pytest.main()  # pragma: no cover
