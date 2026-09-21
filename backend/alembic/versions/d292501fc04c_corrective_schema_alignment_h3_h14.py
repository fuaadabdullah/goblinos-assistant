"""Corrective schema alignment: H14 migration drift + H3 plaintext api_key removal.

Revision ID: d292501fc04c
Revises: add_avatar_url_001
Create Date: 2026-09-21

H14 context
-----------
The migration chain (0ae54fa82ef0 -> add_avatar_url_001) only describes 6
tables (users, search_collections, search_documents, streams, tasks,
stream_chunks) with an outdated `users` table (String PK), while the canonical
models describe 18 tables. The canonical model homes are:

  * backend/models_base.py      -> app_users, user_sessions, audit_logs,
                                   user_roles, user_role_assignments, tasks,
                                   streams, stream_chunks, search_collections,
                                   search_documents, support_messages
  * backend/models/provider.py  -> providers, provider_metrics,
                                   provider_policies, provider_credentials,
                                   model_configs, routing_requests
  * backend/models/model.py     -> models

Duplicate shadow models (backend/models/user.py, task.py, search.py) were
deleted in the same change set; nothing imported them.

H3 context
----------
The plaintext `providers.api_key` column was removed from
backend/models/provider.py. Providers tables created by this migration never
get the column. The guarded drop below covers databases that were built via
`Base.metadata.create_all()` (which included the old column) and are now
being brought under Alembic management.

Safety properties
-----------------
* Purely additive except for the H3 column drop, which only fires when the
  column actually exists and holds no data the app can still read (all live
  code paths were migrated to `api_key_encrypted`).
* Never drops tables that might hold production data. The legacy `users`
  table is left untouched, as are the legacy `tasks`/`streams` tables whose
  `user_id` columns still reference `users.id`.
* Idempotent: every operation is guarded by inspector checks, so the
  migration can be re-run safely. In offline (--sql) mode the guards are
  bypassed and statements are emitted unconditionally for review.

Known divergences intentionally NOT fixed here (documented, operator decision)
-----------------------------------------------------------------------------
* No data migration `users` -> `app_users`: the PK types differ (String vs
  UUID) and column sets differ. If production `users` rows must move, do it
  as a separate, reviewed data migration.
* Legacy `tasks.user_id` / `streams.user_id` (String, FK -> users.id) are not
  altered to UUID FK -> app_users.id; altering column types on populated
  tables risks data loss and is backend-specific.
* `add_avatar_url_001` targets `app_users`, which the initial migration never
  created; the defensive add_column below heals databases in that state.
"""

from typing import Sequence, Union

from alembic import op, context
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d292501fc04c"
down_revision: Union[str, Sequence[str], None] = "add_avatar_url_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Idempotency helpers
# ---------------------------------------------------------------------------

# Tables this migration created in the current run (inspector snapshots go
# stale after DDL, and offline mode has no inspector at all).
_created_tables: set[str] = set()


def _inspector():
    """Return a SQLAlchemy inspector, or None in offline (--sql) mode."""
    if context.is_offline_mode():
        return None
    return sa.inspect(op.get_bind())


def _has_table(insp, name: str) -> bool:
    if insp is None:
        # Offline SQL generation: emit statements unconditionally for review.
        return False
    return insp.has_table(name)


def _column_names(insp, table: str) -> set[str]:
    if insp is None:
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def _index_names(insp, table: str) -> set[str]:
    if insp is None:
        return set()
    return {i["name"] for i in insp.get_indexes(table)}


def _ensure_table(name: str, *columns, indexes=()) -> None:
    """Create table + indexes only if missing (idempotent).

    `indexes` entries are (name, columns, unique) tuples. Unique columns are
    declared as UniqueConstraint objects in the column list (matching what the
    SQLAlchemy models render via ``unique=True``); pass unique=False here for
    plain secondary indexes.
    """
    insp = _inspector()
    if not _has_table(insp, name):
        op.create_table(name, *columns)
        _created_tables.add(name)
    existing_indexes = (
        set() if name in _created_tables else _index_names(_inspector(), name)
    )
    for ix_name, ix_cols, ix_unique in indexes:
        if ix_name not in existing_indexes:
            op.create_index(ix_name, name, ix_cols, unique=ix_unique)
            existing_indexes.add(ix_name)


def _ensure_column(table: str, column: sa.Column) -> None:
    """Add a column to an existing table only if missing (idempotent)."""
    if table in _created_tables:
        return  # just created above with the full column set
    if column.name not in _column_names(_inspector(), table):
        op.add_column(table, column)


# ---------------------------------------------------------------------------
# Canonical column sets (mirror the SQLAlchemy models)
# ---------------------------------------------------------------------------

_UUID = sa.UUID(as_uuid=True)


def _app_users_columns():
    return [
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("avatar_url", sa.String(), nullable=True),
        sa.Column("google_id", sa.String(), nullable=True),
        sa.Column("passkey_credential_id", sa.String(), nullable=True),
        sa.Column("passkey_public_key", sa.Text(), nullable=True),
        sa.Column("role", sa.String(), nullable=True),
        sa.Column("token_version", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    ]


def upgrade() -> None:
    # -- Canonical auth/user tables (missing from the old chain) -------------
    _ensure_table(
        "app_users",
        *_app_users_columns(),
        sa.UniqueConstraint("google_id"),
        indexes=[
            ("ix_app_users_id", ["id"], False),
            # unique=True + index=True on email renders as one unique index
            ("ix_app_users_email", ["email"], True),
        ],
    )
    # Heal databases where add_avatar_url_001 could not run (it targets
    # app_users, which the initial migration never created).
    _ensure_column("app_users", sa.Column("avatar_url", sa.String(), nullable=True))

    _ensure_table(
        "user_sessions",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("user_id", _UUID, sa.ForeignKey("app_users.id"), nullable=False),
        sa.Column("refresh_token_id", sa.String(), nullable=False),
        sa.Column("device_info", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_active", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("revoked", sa.Boolean(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_reason", sa.Text(), nullable=True),
        sa.UniqueConstraint("refresh_token_id"),
    )

    _ensure_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("actor_id", _UUID, sa.ForeignKey("app_users.id"), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("object_table", sa.String(), nullable=True),
        sa.Column("object_id", sa.String(), nullable=True),
        sa.Column("old_values", sa.JSON(), nullable=True),
        sa.Column("new_values", sa.JSON(), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    _ensure_table(
        "user_roles",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("permissions", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("name"),
    )

    _ensure_table(
        "user_role_assignments",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("user_id", _UUID, sa.ForeignKey("app_users.id"), nullable=False),
        sa.Column("role_id", _UUID, sa.ForeignKey("user_roles.id"), nullable=False),
        sa.Column("assigned_by", _UUID, sa.ForeignKey("app_users.id"), nullable=True),
        sa.Column("assigned_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
    )

    _ensure_table(
        "support_messages",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("user_id", _UUID, sa.ForeignKey("app_users.id"), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        indexes=[("ix_support_messages_id", ["id"], False)],
    )

    # -- Provider tables (missing from the old chain) ------------------------
    # H3: created WITHOUT the plaintext api_key column.
    _ensure_table(
        "providers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("display_name", sa.String(200), nullable=True),
        sa.Column("base_url", sa.String(500), nullable=True),
        sa.Column("api_key_encrypted", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=True),
        sa.Column("capabilities", sa.JSON(), nullable=True),
        sa.Column("models", sa.JSON(), nullable=True),
        sa.Column("rate_limits", sa.JSON(), nullable=True),
        sa.Column("cost_per_token", sa.Float(), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("name"),
        indexes=[
            ("ix_providers_id", ["id"], False),
        ],
    )

    # H3: drop the plaintext api_key column if it exists (databases built via
    # Base.metadata.create_all() before the model fix). Guarded + batched so
    # it is safe on SQLite and Postgres; never fires on fresh Alembic builds.
    if "providers" not in _created_tables and "api_key" in _column_names(
        _inspector(), "providers"
    ):
        with op.batch_alter_table("providers") as batch_op:
            batch_op.drop_column("api_key")

    _ensure_table(
        "provider_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "provider_id",
            sa.Integer(),
            sa.ForeignKey("providers.id"),
            nullable=False,
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("is_healthy", sa.Boolean(), nullable=False),
        sa.Column("response_time_ms", sa.Float(), nullable=True),
        sa.Column("error_rate", sa.Float(), nullable=True),
        sa.Column("throughput_rpm", sa.Float(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("cost_incurred", sa.Float(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        indexes=[
            ("ix_provider_metrics_id", ["id"], False),
            ("ix_provider_metrics_timestamp", ["timestamp"], False),
        ],
    )

    _ensure_table(
        "provider_policies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "provider_id",
            sa.Integer(),
            sa.ForeignKey("providers.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("policy_type", sa.String(50), nullable=False),
        sa.Column("conditions", sa.JSON(), nullable=False),
        sa.Column("actions", sa.JSON(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        indexes=[
            ("ix_provider_policies_id", ["id"], False),
        ],
    )

    _ensure_table(
        "provider_credentials",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "provider_id", sa.Integer(), sa.ForeignKey("providers.id"), nullable=True
        ),
        sa.Column("encrypted_key", sa.Text(), nullable=True),
        sa.Column("scopes", sa.JSON(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        indexes=[
            ("ix_provider_credentials_id", ["id"], False),
        ],
    )

    _ensure_table(
        "model_configs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "provider_id", sa.Integer(), sa.ForeignKey("providers.id"), nullable=True
        ),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        indexes=[
            ("ix_model_configs_id", ["id"], False),
        ],
    )

    _ensure_table(
        "routing_requests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("request_id", sa.String(100), nullable=False),
        sa.Column("capability", sa.String(100), nullable=False),
        sa.Column("requirements", sa.JSON(), nullable=True),
        sa.Column(
            "selected_provider_id",
            sa.Integer(),
            sa.ForeignKey("providers.id"),
            nullable=True,
        ),
        sa.Column("response_time_ms", sa.Float(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.UniqueConstraint("request_id"),
        indexes=[("ix_routing_requests_id", ["id"], False)],
    )

    # -- Model catalog table (missing from the old chain) --------------------
    _ensure_table(
        "models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model_id", sa.String(100), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=True),
        sa.UniqueConstraint("name"),
    )

    # NOTE: legacy `users`, `tasks`, `streams`, `stream_chunks`,
    # `search_collections`, `search_documents` tables from the old chain are
    # intentionally left untouched (may hold production data). See module
    # docstring for the documented divergences.


def downgrade() -> None:
    """Best-effort reversal of the additive changes above.

    Note: downgrades are operator-initiated and inherently destructive (tables
    are dropped). The H3 column restore runs before the table drops.
    """
    insp = _inspector()
    # H3 reversal: restore the (removed) plaintext column if providers exists.
    # Runs before the drops below, which include providers.
    if _has_table(insp, "providers") and "api_key" not in _column_names(insp, "providers"):
        op.add_column("providers", sa.Column("api_key", sa.String(500), nullable=True))
    # Drop in reverse dependency order; guarded so partial states are safe.
    for table in [
        "routing_requests",
        "model_configs",
        "provider_credentials",
        "provider_policies",
        "provider_metrics",
        "models",
        "support_messages",
        "user_role_assignments",
        "user_roles",
        "audit_logs",
        "user_sessions",
        "providers",
        "app_users",
    ]:
        if _has_table(insp, table):
            op.drop_table(table)
