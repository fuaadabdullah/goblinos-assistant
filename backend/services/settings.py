import os
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from models.settings import Provider, ProviderCredential, ModelConfig, GlobalSetting

# Import Vault client for secrets management
try:
    from ..vault_client import get_vault_manager

    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

# Fallback to Fernet encryption if Vault is unavailable
if not VAULT_AVAILABLE or not os.getenv("USE_VAULT", "").lower() in (
    "true",
    "1",
    "yes",
):
    # Defer cipher creation until needed to avoid import-time failures
    _encryption_key = None
    _cipher = None

    def _get_cipher():
        global _cipher
        if _cipher is None:
            key = os.getenv("SETTINGS_ENCRYPTION_KEY")
            if not key:
                raise ValueError("SETTINGS_ENCRYPTION_KEY environment variable must be set")
            from cryptography.fernet import Fernet
            _cipher = Fernet(key.encode())
        return _cipher

    cipher = _get_cipher  # Make it callable for backward compatibility
else:
    cipher = None  # Using Vault instead


class SettingsService:
    def __init__(self, db: Session):
        self.db = db

    def get_all_settings(self) -> Dict[str, Any]:
        """Assemble full settings response from DB"""
        providers = self.db.query(Provider).filter(Provider.is_active).all()
        global_settings = {s.key: s.value for s in self.db.query(GlobalSetting).all()}

        result = {"providers": {}, "global": global_settings}

        for provider in providers:
            creds = (
                self.db.query(ProviderCredential)
                .filter(ProviderCredential.provider_id == provider.id)
                .first()
            )
            models = (
                self.db.query(ModelConfig)
                .filter(ModelConfig.provider_id == provider.id)
                .all()
            )

            result["providers"][provider.name] = {
                "display_name": provider.display_name,
                "capabilities": provider.capabilities,
                "default_model": provider.default_model,
                "metadata": provider.metadata_,
                "has_credentials": creds is not None,
                "models": {m.name: m.params for m in models},
            }

        return result

    def get_provider(self, provider_name: str) -> Optional[Provider]:
        """Get a single provider by name"""
        return self.db.query(Provider).filter(Provider.name == provider_name).first()

    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        """Get a single model by name"""
        return self.db.query(ModelConfig).filter(ModelConfig.name == model_name).first()

    def set_provider_credential(self, provider_name: str, key_type: str, api_key: str):
        """Set encrypted credential for provider"""
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")

        # Try Vault first, fallback to database encryption
        if VAULT_AVAILABLE and os.getenv("USE_VAULT", "").lower() in (
            "true",
            "1",
            "yes",
        ):
            vault_manager = get_vault_manager()
            success = vault_manager.set_api_key(provider_name, api_key, key_type)
            if success:
                # Store a reference in DB that Vault is being used
                self._update_or_create_credential_record(provider.id, "vault", key_type)
                return

        # Fallback to database encryption
        if cipher is None:
            raise ValueError("Neither Vault nor Fernet encryption is available")

        # Remove existing credential if any
        existing = (
            self.db.query(ProviderCredential)
            .filter(ProviderCredential.provider_id == provider.id)
            .first()
        )
        if existing:
            self.db.delete(existing)

        # Add new encrypted credential
        encrypted_key = self.encrypt_key(api_key)
        cred = ProviderCredential(
            provider_id=provider.id,
            encrypted_key=encrypted_key,
            created_by="system",
        )
        self.db.add(cred)
        self.db.commit()

    def get_provider_credential(
        self, provider_name: str, key_type: str
    ) -> Optional[str]:
        """Get decrypted credential for provider"""
        provider = self.get_provider(provider_name)
        if not provider:
            return None

        # Try Vault first
        if VAULT_AVAILABLE and os.getenv("USE_VAULT", "").lower() in (
            "true",
            "1",
            "yes",
        ):
            vault_manager = get_vault_manager()
            credential = vault_manager.get_api_key(provider_name, key_type)
            if credential:
                return credential

        # Fallback to database encryption
        if cipher is None:
            return None

        cred = (
            self.db.query(ProviderCredential)
            .filter(ProviderCredential.provider_id == provider.id)
            .first()
        )
        if not cred:
            return None

        return self.decrypt_key(cred.encrypted_key)

    def update_provider(self, provider_name: str, data: Dict[str, Any]) -> Provider:
        """Update or create provider"""
        provider = (
            self.db.query(Provider).filter(Provider.name == provider_name).first()
        )
        if not provider:
            provider = Provider(
                name=provider_name,
                display_name=data.get("display_name", provider_name.title()),
                capabilities=data.get("capabilities", ["chat"]),
                metadata_=data.get("metadata", {}),
            )
            self.db.add(provider)

        for key, value in data.items():
            if key == "default_model":
                provider.default_model = value
            elif key == "capabilities":
                provider.capabilities = value
            elif key == "metadata":
                setattr(provider, "metadata_", value)
            elif key == "display_name":
                provider.display_name = value

        self.db.commit()
        self.db.refresh(provider)
        return provider

    def update_model(self, model_name: str, data: Dict[str, Any]) -> ModelConfig:
        """Update or create model config"""
        # Extract params from data, defaulting to the remaining data if no params key
        params = data.get("params", data.copy())
        # Remove non-param keys
        for key in ["model_name", "provider_name"]:
            params.pop(key, None)

        provider_name = data.get("provider_name", model_name.split("-")[0])
        provider = (
            self.db.query(Provider).filter(Provider.name == provider_name).first()
        )
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")

        model = (
            self.db.query(ModelConfig)
            .filter(
                ModelConfig.name == model_name, ModelConfig.provider_id == provider.id
            )
            .first()
        )

        if not model:
            model = ModelConfig(name=model_name, provider_id=provider.id, params=params)
            self.db.add(model)
        else:
            model.params = params

        self.db.commit()
        self.db.refresh(model)
        return model

    def encrypt_key(self, key: str) -> str:
        """Encrypt API key using Fernet (fallback method)"""
        if cipher is None:
            raise ValueError("Fernet encryption not available (using Vault)")
        return cipher().encrypt(key.encode()).decode()

    def decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt API key using Fernet (fallback method)"""
        if cipher is None:
            raise ValueError("Fernet encryption not available (using Vault)")
        return cipher().decrypt(encrypted_key.encode()).decode()

    def test_connection(
        self, provider_name: str, api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test connection to provider"""
        provider = (
            self.db.query(Provider).filter(Provider.name == provider_name).first()
        )
        if not provider:
            return {"success": False, "error": "Provider not found"}

        if not api_key:
            cred = (
                self.db.query(ProviderCredential)
                .filter(ProviderCredential.provider_id == provider.id)
                .first()
            )
            if not cred:
                return {"success": False, "error": "No credentials found"}
            api_key = self.decrypt_key(cred.encrypted_key)

        # Import provider-specific tester
        try:
            module = __import__(
                f"providers.{provider_name}", fromlist=["test_connection"]
            )
            return module.test_connection(api_key)
        except ImportError:
            return {"success": False, "error": f"No tester for {provider_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
