from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy import func, and_
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from app.utils.logging_util import setup_logger
from app.utils.postgresql_db_util import db_util


class DatabaseOperatorUtil:

    def __init__(self):

        self.logger = setup_logger(__name__)

    @staticmethod
    def _filter_valid_fields(model, data):

        model_columns = {column.key for column in inspect(model).columns}
        return {key: value for key, value in data.items() if key in model_columns}

    async def save_to_database(self, model, data: dict, filter_by: dict, update_fields: dict = None,
                               hash_value: str = None, hash_field: str = 'hash'):
        """
        Generic method to save or update an entry in the database.

        :param model: SQLAlchemy model class to operate on
        :param data: Data dictionary to create a new entry if none exists
        :param filter_by: Dictionary of fields to filter by, e.g., {'userId': 'some_id'}
        :param update_fields: Optional dictionary of fields to update if entry exists, e.g., {'status': 'new_status'}
        """
        session: Session = db_util.get_session()

        try:
            with session.begin():
                existing_entry = session.query(model).filter_by(**filter_by).first()

                if existing_entry:
                    if update_fields:
                        for key, value in update_fields.items():
                            setattr(existing_entry, key, value)
                else:
                    filtered_data = self._filter_valid_fields(model, data)
                    if hash_value:
                        filtered_data[hash_field] = hash_value
                    new_entry = model(**filtered_data)
                    session.add(new_entry)

            session.commit()

        except IntegrityError as db_err:
            session.rollback()
            self.logger.error(f'Duplicate entry detected with filters {filter_by}: {db_err}')
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail='Duplicate entry detected.')
        except SQLAlchemyError as db_err:
            session.rollback()
            self.logger.error(
                f'Failed to save to database for model {model.__name__} with filters {filter_by}: {db_err}')
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail='Database save error.')
        finally:
            session.close()
            self.logger.info(f'Session closed after saving to database for model {model.__name__}')

    async def find_all(self, model, filter_expr, limit=None, offset=None, order_by=None):
        """
        Retrieves all entries matching the filter expression with optional pagination.

        :param model: SQLAlchemy model class to query
        :param filter_expr: SQLAlchemy filter expression
        :param limit: Optional maximum number of results to return
        :param offset: Optional number of results to skip
        :param order_by: Optional column to order results by
        :return: List of model instances matching the query
        """
        session: Session = db_util.get_session()

        try:
            with session.begin():
                query = session.query(model).filter(filter_expr)

                # Apply ordering if specified
                if order_by is not None:
                    query = query.order_by(order_by)

                # Apply pagination if specified
                if offset is not None:
                    query = query.offset(offset)
                if limit is not None:
                    query = query.limit(limit)

                results = query.all()
                self.logger.info(f'Found {len(results)} entries matching filter expression.')
                return [instance.to_dict() for instance in results]

        except SQLAlchemyError as db_err:
            self.logger.error(f'Error finding entries with filter expression: {db_err}')
            return []

        finally:
            session.close()
            self.logger.info('Session closed after find_all operation.')

    async def find_one(self, model, filter_expr):
        """
        Retrieves a single entry matching the filter expression.

        :param model: SQLAlchemy model class to query
        :param filter_expr: SQLAlchemy filter expression
        :return: Single model instance matching the query or None
        """
        session: Session = db_util.get_session()

        try:
            with session.begin():
                result = session.query(model).filter(filter_expr).first()
                if result:
                    self.logger.info('Entry found for filter expression.')
                    return result.to_dict()
                else:
                    self.logger.info('No entry found for filter expression.')
                    return {}

        except SQLAlchemyError as db_err:
            self.logger.error(f'Error finding entry with filter expression: {db_err}')
            return {}

        finally:
            session.close()
            self.logger.info('Session closed after find_one operation.')

    async def update_database(self, model, filter_expr, update_data: dict):
        """
        Updates entries matching the filter expression with the provided data.

        :param model: SQLAlchemy model class to operate on
        :param filter_expr: SQLAlchemy filter expression
        :param update_data: Dictionary of fields to update
        """
        session: Session = db_util.get_session()

        try:
            with session.begin():
                updated_count = session.query(model).filter(filter_expr).update(update_data)
            session.commit()
            self.logger.debug(f'Updated {updated_count} entries matching filter expression with data {update_data}')

        except SQLAlchemyError as db_err:
            session.rollback()
            self.logger.error(f'Error updating entries with filter expression: {db_err}')
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail='Database update error.')
        finally:
            session.close()
            self.logger.info('Session closed after update_database operation.')

    async def count_entries(self, model, filter_expr):
        """
        Count entries matching the filter expression.

        :param model: SQLAlchemy model class to query
        :param filter_expr: SQLAlchemy filter expression
        :return: Count of matching entries
        """
        session: Session = db_util.get_session()

        try:
            with session.begin():
                count = session.query(func.count()).select_from(model).filter(filter_expr).scalar()
                return count or 0

        except SQLAlchemyError as db_err:
            self.logger.error(f'Error counting entries: {db_err}')
            return 0

        finally:
            session.close()
            self.logger.info('Session closed after count_entries operation.')

    async def update(self, model, filter_by, data):
        """
        Alias for update_database that accepts filter_by as a dictionary
        instead of a SQLAlchemy expression.

        :param model: SQLAlchemy model class to operate on
        :param filter_by: Dictionary of filter conditions
        :param data: Dictionary of fields to update
        """
        # Convert filter_by dictionary to SQLAlchemy expression
        conditions = []
        for key, value in filter_by.items():
            column = getattr(model, key)
            conditions.append(column == value)

        filter_expr = None
        if len(conditions) == 1:
            filter_expr = conditions[0]
        else:
            filter_expr = and_(*conditions)

        return await self.update_database(model, filter_expr, data)


database_operator_util = DatabaseOperatorUtil()
