"""
Authentication policies and role definitions for Goblin Assistant.

Defines scopes, roles, and permissions for JWT tokens and API keys.
"""

from enum import Enum
from typing import Dict, Set, List
from dataclasses import dataclass


class AuthScope(Enum):
    """Authentication scopes for fine-grained permissions."""

    # Read operations
    READ_USER = "read:user"
    READ_MODELS = "read:models"
    READ_CONVERSATIONS = "read:conversations"
    READ_SETTINGS = "read:settings"

    # Write operations
    WRITE_USER = "write:user"
    WRITE_CONVERSATIONS = "write:conversations"
    WRITE_SETTINGS = "write:settings"

    # Admin operations
    ADMIN_USERS = "admin:users"
    ADMIN_MODELS = "admin:models"
    ADMIN_SYSTEM = "admin:system"

    # Service operations (for API keys)
    SERVICE_INFERENCE = "service:inference"
    SERVICE_MONITORING = "service:monitoring"
    SERVICE_ADMIN = "service:admin"


class UserRole(Enum):
    """User roles with predefined permission sets."""

    USER = "user"
    PREMIUM_USER = "premium_user"
    ADMIN = "admin"
    SERVICE = "service"  # For API keys/bots


@dataclass
class RolePermissions:
    """Permissions associated with a role."""

    scopes: Set[AuthScope]
    description: str


# Role-based permissions
ROLE_PERMISSIONS: Dict[UserRole, RolePermissions] = {
    UserRole.USER: RolePermissions(
        scopes={
            AuthScope.READ_USER,
            AuthScope.READ_MODELS,
            AuthScope.READ_CONVERSATIONS,
            AuthScope.READ_SETTINGS,
            AuthScope.WRITE_CONVERSATIONS,
            AuthScope.WRITE_SETTINGS,
        },
        description="Basic user with read/write access to personal data",
    ),
    UserRole.PREMIUM_USER: RolePermissions(
        scopes={
            AuthScope.READ_USER,
            AuthScope.READ_MODELS,
            AuthScope.READ_CONVERSATIONS,
            AuthScope.READ_SETTINGS,
            AuthScope.WRITE_USER,
            AuthScope.WRITE_CONVERSATIONS,
            AuthScope.WRITE_SETTINGS,
        },
        description="Premium user with additional write permissions",
    ),
    UserRole.ADMIN: RolePermissions(
        scopes=set(AuthScope),  # All scopes
        description="Administrator with full system access",
    ),
    UserRole.SERVICE: RolePermissions(
        scopes={
            AuthScope.SERVICE_INFERENCE,
            AuthScope.SERVICE_MONITORING,
        },
        description="Service account for automated operations",
    ),
}


def get_role_permissions(role: UserRole) -> RolePermissions:
    """Get permissions for a role.

    Args:
        role: User role

    Returns:
        RolePermissions for the role

    Raises:
        ValueError: If role is not defined
    """
    if role not in ROLE_PERMISSIONS:
        raise ValueError(f"Unknown role: {role}")
    return ROLE_PERMISSIONS[role]


def get_scopes_for_role(role: UserRole) -> Set[AuthScope]:
    """Get scopes for a role.

    Args:
        role: User role

    Returns:
        Set of scopes for the role
    """
    return get_role_permissions(role).scopes


def validate_scopes(scopes: List[str]) -> Set[AuthScope]:
    """Validate and convert scope strings to AuthScope enums.

    Args:
        scopes: List of scope strings

    Returns:
        Set of valid AuthScope enums

    Raises:
        ValueError: If any scope is invalid
    """
    valid_scopes = set()
    for scope_str in scopes:
        try:
            scope = AuthScope(scope_str)
            valid_scopes.add(scope)
        except ValueError:
            raise ValueError(f"Invalid scope: {scope_str}")

    return valid_scopes


def has_scope(user_scopes: Set[AuthScope], required_scope: AuthScope) -> bool:
    """Check if user has required scope.

    Args:
        user_scopes: User's scopes
        required_scope: Required scope

    Returns:
        True if user has the scope
    """
    return required_scope in user_scopes


def has_any_scope(user_scopes: Set[AuthScope], required_scopes: Set[AuthScope]) -> bool:
    """Check if user has any of the required scopes.

    Args:
        user_scopes: User's scopes
        required_scopes: Required scopes

    Returns:
        True if user has at least one required scope
    """
    return bool(user_scopes & required_scopes)


def has_all_scopes(
    user_scopes: Set[AuthScope], required_scopes: Set[AuthScope]
) -> bool:
    """Check if user has all required scopes.

    Args:
        user_scopes: User's scopes
        required_scopes: Required scopes

    Returns:
        True if user has all required scopes
    """
    return required_scopes.issubset(user_scopes)
