"""
Encrypted API key storage for service authentication.

Provides secure storage and retrieval of API keys using Fernet encryption.
API keys are used for service-to-service authentication and automated operations.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .secrets_manager import get_secrets_manager


@dataclass
class APIKey:
    """API key with metadata."""

    key_id: str
    name: str
    hashed_key: str  # PBKDF2 hash of the actual key
    salt: str  # Base64 encoded salt
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    scopes: List[str]
    metadata: Dict[str, Any]
    active: bool = True

    def is_expired(self) -> bool:
        """Check if the key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if the key is valid (active and not expired)."""
        return self.active and not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["created_at"] = self.created_at.isoformat()
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        if self.last_used:
            data["last_used"] = self.last_used.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKey":
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        if data.get("last_used"):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


class APIKeyStore:
    """Encrypted storage for API keys."""

    def __init__(self, storage_path: str, encryption_key: Optional[str] = None):
        """Initialize the API key store.

        Args:
            storage_path: Path to store encrypted keys
            encryption_key: Base64 encoded encryption key (generated if None)
        """
        self.storage_path = storage_path

        # Use secrets manager for encryption key if not provided
        if encryption_key is None:
            secrets_manager = get_secrets_manager()
            encryption_key = secrets_manager.get_api_key_store_encryption_key()

        self.encryption_key = encryption_key or self._generate_key()
        self._ensure_storage_path()

        try:
            self.fernet = Fernet(self.encryption_key)
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _generate_key(self) -> str:
        """Generate a new Fernet key."""
        return Fernet.generate_key().decode()

    def _ensure_storage_path(self):
        """Ensure storage directory exists."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def _load_keys(self) -> Dict[str, APIKey]:
        """Load encrypted keys from storage."""
        if not os.path.exists(self.storage_path):
            return {}

        try:
            with open(self.storage_path, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self.fernet.decrypt(encrypted_data)
            keys_data = json.loads(decrypted_data.decode())

            keys = {}
            for key_data in keys_data:
                key = APIKey.from_dict(key_data)
                keys[key.key_id] = key

            return keys

        except (InvalidToken, json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to load API keys: {e}")

    def _save_keys(self, keys: Dict[str, APIKey]):
        """Save keys to encrypted storage."""
        keys_data = [key.to_dict() for key in keys.values()]
        json_data = json.dumps(keys_data, indent=2)

        encrypted_data = self.fernet.encrypt(json_data.encode())

        with open(self.storage_path, "wb") as f:
            f.write(encrypted_data)

    def _hash_key(self, key: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """Hash an API key using PBKDF2.

        Args:
            key: The API key to hash
            salt: Optional salt bytes

        Returns:
            Tuple of (hashed_key, salt_b64)
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        hashed = base64.urlsafe_b64encode(kdf.derive(key.encode())).decode()
        salt_b64 = base64.urlsafe_b64encode(salt).decode()

        return hashed, salt_b64

    def _verify_key(self, key: str, hashed_key: str, salt_b64: str) -> bool:
        """Verify an API key against its hash.

        Args:
            key: The API key to verify
            hashed_key: The stored hash
            salt_b64: The stored salt

        Returns:
            True if key matches hash
        """
        try:
            salt = base64.urlsafe_b64decode(salt_b64)
            expected_hash, _ = self._hash_key(key, salt)
            return expected_hash == hashed_key
        except Exception:
            return False

    def create_key(
        self,
        name: str,
        scopes: List[str],
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """Create a new API key.

        Args:
            name: Human-readable name for the key
            scopes: List of scopes for the key
            expires_in_days: Optional expiration in days
            metadata: Optional metadata

        Returns:
            Tuple of (key_id, actual_key)
        """
        import uuid

        key_id = str(uuid.uuid4())
        actual_key = str(uuid.uuid4())  # Generate random key

        # Hash the key
        hashed_key, salt = self._hash_key(actual_key)

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create key object
        key = APIKey(
            key_id=key_id,
            name=name,
            hashed_key=hashed_key,
            salt=salt,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            last_used=None,
            scopes=scopes,
            metadata=metadata or {},
            active=True,
        )

        # Save to storage
        keys = self._load_keys()
        keys[key_id] = key
        self._save_keys(keys)

        return key_id, actual_key

    def verify_key(self, provided_key: str) -> Optional[APIKey]:
        """Verify an API key and return the key object if valid.

        Args:
            provided_key: The API key to verify

        Returns:
            APIKey object if valid, None otherwise
        """
        keys = self._load_keys()

        for key in keys.values():
            if key.is_valid() and self._verify_key(
                provided_key, key.hashed_key, key.salt
            ):
                # Update last used
                key.last_used = datetime.utcnow()
                self._save_keys(keys)
                return key

        return None

    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get a key by ID.

        Args:
            key_id: The key ID

        Returns:
            APIKey object if found
        """
        keys = self._load_keys()
        return keys.get(key_id)

    def list_keys(self, include_inactive: bool = False) -> List[APIKey]:
        """List all keys.

        Args:
            include_inactive: Whether to include inactive keys

        Returns:
            List of APIKey objects
        """
        keys = self._load_keys()
        if include_inactive:
            return list(keys.values())
        return [key for key in keys.values() if key.active]

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: The key ID to revoke

        Returns:
            True if key was revoked
        """
        keys = self._load_keys()
        if key_id in keys:
            keys[key_id].active = False
            self._save_keys(keys)
            return True
        return False

    def delete_key(self, key_id: str) -> bool:
        """Delete an API key permanently.

        Args:
            key_id: The key ID to delete

        Returns:
            True if key was deleted
        """
        keys = self._load_keys()
        if key_id in keys:
            del keys[key_id]
            self._save_keys(keys)
            return True
        return False

    def rotate_key(self, key_id: str) -> Optional[tuple[str, str]]:
        """Rotate an API key (generate new key with same permissions).

        Args:
            key_id: The key ID to rotate

        Returns:
            Tuple of (key_id, new_actual_key) if successful
        """
        keys = self._load_keys()
        if key_id not in keys:
            return None

        old_key = keys[key_id]

        # Generate new key
        new_actual_key = str(uuid.uuid4())
        hashed_key, salt = self._hash_key(new_actual_key)

        # Update key object
        old_key.hashed_key = hashed_key
        old_key.salt = salt
        old_key.created_at = datetime.utcnow()
        old_key.last_used = None

        self._save_keys(keys)

        return key_id, new_actual_key

    def cleanup_expired_keys(self) -> int:
        """Remove expired keys.

        Returns:
            Number of keys removed
        """
        keys = self._load_keys()
        expired_count = 0

        for key_id, key in list(keys.items()):
            if key.is_expired():
                del keys[key_id]
                expired_count += 1

        if expired_count > 0:
            self._save_keys(keys)

        return expired_count
