"""add_avatar_url_to_users

Revision ID: add_avatar_url_001
Revises: 0ae54fa82ef0
Create Date: 2025-01-15

H14 fix: this migration targets `app_users`, which the initial migration never
created (it created the legacy `users` table). Made idempotent: the column is
added only if the table exists and the column is missing. Databases that need
the full `app_users` table get it from the corrective migration
d292501fc04c, which also heals a missing avatar_url column.
"""

from alembic import op, context
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_avatar_url_001"
down_revision = "0ae54fa82ef0"
branch_labels = None
depends_on = None


def _column_missing(table: str, column: str) -> bool:
    if context.is_offline_mode():
        return True  # emit statements unconditionally for SQL review
    insp = sa.inspect(op.get_bind())
    if not insp.has_table(table):
        return False
    return column not in {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    """Add avatar_url column to app_users table."""
    if _column_missing("app_users", "avatar_url"):
        op.add_column("app_users", sa.Column("avatar_url", sa.String(), nullable=True))


def downgrade() -> None:
    """Remove avatar_url column from app_users table."""
    if context.is_offline_mode():
        op.drop_column("app_users", "avatar_url")
        return
    insp = sa.inspect(op.get_bind())
    if insp.has_table("app_users") and "avatar_url" in {
        c["name"] for c in insp.get_columns("app_users")
    }:
        op.drop_column("app_users", "avatar_url")
