"""
License management and validation module for Goblin Assistant.

Handles API key validation, license tier enforcement, and rate limiting based on license.
"""

import os
import hashlib
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from pydantic import BaseModel


class LicenseTier(str, Enum):
    """License tier levels"""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class LicenseInfo(BaseModel):
    """License information"""

    key: str
    tier: LicenseTier
    created_at: datetime
    expires_at: Optional[datetime] = None
    requests_per_minute: int
    concurrent_requests: int
    max_response_tokens: int
    features: Dict[str, bool]

    class Config:
        use_enum_values = False


class LicenseValidator:
    """Validates and manages API licenses"""

    # Default tier configurations
    TIER_CONFIGS = {
        LicenseTier.FREE: {
            "requests_per_minute": 10,
            "concurrent_requests": 2,
            "max_response_tokens": 1000,
            "features": {
                "rag": False,
                "streaming": False,
                "advanced_routing": False,
                "batch_processing": False,
            },
        },
        LicenseTier.PRO: {
            "requests_per_minute": 100,
            "concurrent_requests": 10,
            "max_response_tokens": 8000,
            "features": {
                "rag": True,
                "streaming": True,
                "advanced_routing": True,
                "batch_processing": False,
            },
        },
        LicenseTier.ENTERPRISE: {
            "requests_per_minute": 1000,
            "concurrent_requests": 100,
            "max_response_tokens": 32000,
            "features": {
                "rag": True,
                "streaming": True,
                "advanced_routing": True,
                "batch_processing": True,
            },
        },
    }

    def __init__(self):
        """Initialize license validator with environment settings"""
        # License keys: comma-separated "key:tier" pairs
        self.licenses_str = os.getenv("LICENSE_KEYS", "")
        # Default tier for requests without license
        self.default_tier = LicenseTier(os.getenv("DEFAULT_LICENSE_TIER", "free"))
        # Enable/disable license enforcement
        self.enforce_licensing = os.getenv("ENFORCE_LICENSING", "true").lower() == "true"
        # Build license registry
        self._licenses = self._build_license_registry()

    def _build_license_registry(self) -> Dict[str, Tuple[LicenseTier, Optional[datetime]]]:
        """
        Build registry of valid licenses from environment.

        Format: "key1:pro,key2:enterprise,key3:free:2026-12-31"
        """
        registry = {}
        if not self.licenses_str:
            return registry

        for license_entry in self.licenses_str.split(","):
            parts = license_entry.strip().split(":")
            if len(parts) < 2:
                continue

            key = parts[0].strip()
            tier_str = parts[1].strip().lower()
            expires_at = None

            # Parse expiration date if provided
            if len(parts) >= 3:
                try:
                    expires_at = datetime.fromisoformat(parts[2].strip())
                except (ValueError, IndexError):
                    pass

            try:
                tier = LicenseTier(tier_str)
                registry[key] = (tier, expires_at)
            except ValueError:
                # Invalid tier, skip
                pass

        return registry

    def validate_license(self, api_key: Optional[str]) -> Tuple[bool, LicenseTier, Optional[str]]:
        """
        Validate an API key and return (is_valid, tier, error_message).

        Returns:
            Tuple of (is_valid, tier, error_message)
            - is_valid: True if key is valid
            - tier: LicenseTier for the key (or default tier if no key provided)
            - error_message: None if valid, error description if invalid
        """
        if not self.enforce_licensing:
            return True, self.default_tier, None

        # No key provided - use default tier
        if not api_key:
            return True, self.default_tier, None

        # Check if key exists in registry
        if api_key not in self._licenses:
            return False, self.default_tier, f"Invalid API key: {api_key[:8]}..."

        tier, expires_at = self._licenses[api_key]

        # Check expiration
        if expires_at and datetime.utcnow() > expires_at:
            return False, tier, f"License expired on {expires_at.isoformat()}"

        return True, tier, None

    def get_tier_config(self, tier: LicenseTier) -> Dict:
        """Get configuration for a license tier"""
        return self.TIER_CONFIGS.get(tier, self.TIER_CONFIGS[LicenseTier.FREE])

    def get_license_info(self, api_key: Optional[str]) -> Optional[LicenseInfo]:
        """Get license information for an API key"""
        is_valid, tier, error = self.validate_license(api_key)

        if not is_valid:
            return None

        config = self.get_tier_config(tier)
        expires_at = None

        if api_key and api_key in self._licenses:
            _, expires_at = self._licenses[api_key]

        return LicenseInfo(
            key=api_key or "default",
            tier=tier,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            requests_per_minute=config["requests_per_minute"],
            concurrent_requests=config["concurrent_requests"],
            max_response_tokens=config["max_response_tokens"],
            features=config["features"],
        )

    def has_feature(self, tier: LicenseTier, feature: str) -> bool:
        """Check if a tier has access to a feature"""
        config = self.get_tier_config(tier)
        return config["features"].get(feature, False)


# Global validator instance
_validator: Optional[LicenseValidator] = None


def get_license_validator() -> LicenseValidator:
    """Get or create the global license validator instance"""
    global _validator
    if _validator is None:
        _validator = LicenseValidator()
    return _validator


def reset_validator():
    """Reset validator (for testing)"""
    global _validator
    _validator = None
