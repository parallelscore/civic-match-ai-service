from app.utils.postgresql_db_util import db_util


def create_all_tables():
    db_util.create_all_tables()
    return 'Tables created successfully'


if __name__ == '__main__':
    create_all_tables()  # pragma: no cover
