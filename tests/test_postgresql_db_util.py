from unittest.mock import patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.utils.postgresql_db_util import DBUtil


class TestDBUtil:
    """Test the database utility."""

    @patch('app.utils.postgresql_db_util.create_engine')
    @patch('app.utils.postgresql_db_util.declarative_base')
    @patch('app.utils.postgresql_db_util.sessionmaker')
    @patch('app.utils.postgresql_db_util.setup_logger')
    def test_init(self, mock_setup_logger, mock_sessionmaker, mock_declarative_base, mock_create_engine):
        """Test the initialization of DBUtil."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_base = MagicMock()
        mock_declarative_base.return_value = mock_base

        mock_session = MagicMock()
        mock_sessionmaker.return_value = mock_session

        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        # Create a DBUtil instance
        db_util = DBUtil()

        # Verify the initialization
        assert db_util.engine == mock_engine
        assert db_util.base == mock_base
        assert db_util.session == mock_session
        assert db_util.logger == mock_logger

        # Verify the mocks were called correctly
        mock_create_engine.assert_called_once()
        mock_declarative_base.assert_called_once()
        mock_sessionmaker.assert_called_once_with(bind=mock_engine)

    @patch('app.utils.postgresql_db_util.setup_logger')
    def test_create_all_tables_success(self, mock_setup_logger):
        """Test creating all tables successfully."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_engine = MagicMock()
        mock_base = MagicMock()
        mock_metadata = MagicMock()
        mock_base.metadata = mock_metadata
        mock_base.metadata.tables = {"table1": {}, "table2": {}}

        # Create a DBUtil instance with our mocks
        db_util = DBUtil()
        db_util.engine = mock_engine
        db_util.base = mock_base

        # Call the method
        db_util.create_all_tables()

        # Verify the method called the metadata create_all method
        mock_metadata.create_all.assert_called_once_with(bind=mock_engine)

        # Verify the logging
        mock_logger.info.assert_called_once_with('Tables created: %s', 'table1, table2')

    @patch('app.utils.postgresql_db_util.setup_logger')
    def test_create_all_tables_error(self, mock_setup_logger):
        """Test creating all tables with an error."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_engine = MagicMock()
        mock_base = MagicMock()
        mock_metadata = MagicMock()
        mock_base.metadata = mock_metadata
        mock_metadata.create_all.side_effect = SQLAlchemyError("Test error")

        # Create a DBUtil instance with our mocks
        db_util = DBUtil()
        db_util.engine = mock_engine
        db_util.base = mock_base

        # Call the method
        db_util.create_all_tables()

        # Verify the error logging
        mock_logger.error.assert_called_once_with("Error occurred while creating tables: %s", "Test error")

    @patch('app.utils.postgresql_db_util.setup_logger')
    def test_get_session_success(self, mock_setup_logger):
        """Test getting a session successfully."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_session_factory = MagicMock()
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        # Create a DBUtil instance with our mocks
        db_util = DBUtil()
        db_util.session = mock_session_factory

        # Call the method
        result = db_util.get_session()

        # Verify the result
        assert result == mock_session
        mock_session_factory.assert_called_once()

    @patch('app.utils.postgresql_db_util.setup_logger')
    def test_get_session_error(self, mock_setup_logger):
        """Test getting a session with an error."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_session_factory = MagicMock()
        mock_session_factory.side_effect = SQLAlchemyError("Test error")

        # Create a DBUtil instance with our mocks
        db_util = DBUtil()
        db_util.session = mock_session_factory

        # Call the method
        result = db_util.get_session()

        # Verify the result and logging
        assert result is None
        mock_logger.error.assert_called_once_with("Error occurred while getting session: %s", "Test error")
        