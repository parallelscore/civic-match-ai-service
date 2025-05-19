import os
import pytest

from app.core.config import get_settings, BaseConfig, DevConfig, DemoConfig, ProdConfig


class TestConfig:

    @pytest.fixture(autouse=True)
    def clear_env(self):
        original_env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(original_env)

    def test_default_config(self):
        settings = get_settings()
        assert isinstance(settings, BaseConfig)
        assert settings.PROJECT_NAME == "Civic Match Matching Engine"
        assert settings.VERSION == "1.0.0"
        assert settings.CORS_ORIGINS == ["*"]

    def test_development_config(self, monkeypatch):
        monkeypatch.setenv('ENV', 'dev')
        settings = get_settings()
        assert isinstance(settings, DevConfig)
        assert settings.DEBUG is True

    def test_testing_config(self, monkeypatch):
        monkeypatch.setenv('ENV', 'demo')
        settings = get_settings()
        assert isinstance(settings, DemoConfig)
        assert settings.DEBUG is True

    def test_production_config(self, monkeypatch):
        monkeypatch.setenv('ENV', 'prod')
        settings = get_settings()
        assert isinstance(settings, ProdConfig)
        assert settings.DEBUG is False

    def test_env_file_does_not_exist(self, monkeypatch, capfd):
        monkeypatch.setenv('ENV', 'prod')
        monkeypatch.setattr(os.path, 'exists', lambda x: False)

        settings = get_settings()

        captured = capfd.readouterr()
        assert "Environment file .env.prod does not exist" in captured.out

        assert isinstance(settings, BaseConfig)
        assert settings.DEBUG is False  # Default value if not set


if __name__ == '__main__':
    pytest.main()  # pragma: no cover
