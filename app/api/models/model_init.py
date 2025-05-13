# Import all models to ensure they're registered with SQLAlchemy
from app.utils.postgresql_db_util import db_util


def create_all_tables():
    """
    Create all tables in the correct order to handle foreign key dependencies.

    Order matters:
    1. Elections (no foreign keys)
    2. Questions, Candidates, Voters (depend on Elections)
    3. Candidate Responses, Voter Responses (depend on Questions, Candidates, Voters)
    4. Match Results (depend on Candidates and Voters)
    5. User Feedback (depends on Match Results)
    """
    db_util.create_all_tables()
    return 'Tables created successfully'


if __name__ == '__main__':
    create_all_tables()  # pragma: no cover
