from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings
from app.utils.logging_util import setup_logger


class DBUtil:
    """
    A utility class for managing database connections, sessions, and table creation using SQLAlchemy.
    """

    def __init__(self):
        """
        Initializes the DBUtil with a database engine, a declarative base for defining tables, and a session factory.
        """
        self.logger = setup_logger(__name__)
        self.engine = create_engine(settings.POSTGRESQL_DATABASE_URL)
        self.base = declarative_base()
        self.session = sessionmaker(bind=self.engine)

    def create_all_tables(self):
        """
        Creates all tables defined in the metadata of the base declarative class.

        This method uses the metadata to create tables in the database if they don’t already exist,
        and logs the names of the tables created.

        Raises:
            SQLAlchemyError: If an error occurs during table creation.
        """
        try:
            self.base.metadata.create_all(bind=self.engine)

            # List the tables that were created
            tables = self.base.metadata.tables.keys()
            self.logger.info('Tables created: %s', ', '.join(tables))

        except SQLAlchemyError as e:
            self.logger.error("Error occurred while creating tables: %s", str(e))

    def get_session(self):
        """
        Provides a new session for database operations.

        Returns:
            session: A new SQLAlchemy session object for executing database queries.

        Raises:
            SQLAlchemyError: If an error occurs while creating a session.
        """

        try:
            return self.session()

        except SQLAlchemyError as e:
            self.logger.error("Error occurred while getting session: %s", str(e))
            return None


db_util = DBUtil()
