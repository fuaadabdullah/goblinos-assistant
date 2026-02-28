"""add_avatar_url_to_users

Revision ID: add_avatar_url_001
Revises: 0ae54fa82ef0
Create Date: 2025-01-15

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_avatar_url_001"
down_revision = "0ae54fa82ef0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add avatar_url column to app_users table."""
    op.add_column("app_users", sa.Column("avatar_url", sa.String(), nullable=True))


def downgrade() -> None:
    """Remove avatar_url column from app_users table."""
    op.drop_column("app_users", "avatar_url")
