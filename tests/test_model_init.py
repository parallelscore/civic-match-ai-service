from unittest.mock import patch

from app.api.models.model_init import create_all_tables


class TestModelInit:
    """Test the model initialization."""

    @patch('app.api.models.model_init.db_util')
    def test_create_all_tables(self, mock_db_util):
        """Test the create_all_tables function."""
        # Call the function
        result = create_all_tables()

        # Verify the function called the database utility correctly
        mock_db_util.create_all_tables.assert_called_once()

        # Verify the return value
        assert result == 'Tables created successfully'