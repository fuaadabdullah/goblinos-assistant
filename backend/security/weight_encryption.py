"""
Weight Encryption
AES-256-GCM encryption for model weights on untrusted hosts (Vast.ai)
"""

import base64
import os
import secrets
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import structlog

from ..config import settings

logger = structlog.get_logger()


class WeightEncryption:
    """
    AES-256-GCM encryption for model weights.
    
    Used for:
    - Encrypting weights before uploading to GCS
    - Decrypting weights at container start on untrusted hosts
    - Protecting sensitive/proprietary models
    """
    
    NONCE_SIZE = 12  # 96 bits for GCM
    KEY_SIZE = 32    # 256 bits
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryptor with key.
        
        Args:
            key: 32-byte encryption key. If None, loads from settings.
        """
        if key:
            self.key = key
        else:
            key_b64 = settings.weight_encryption_key.get_secret_value()
            if key_b64:
                self.key = base64.b64decode(key_b64)
            else:
                raise ValueError("No encryption key configured")
        
        if len(self.key) != self.KEY_SIZE:
            raise ValueError(f"Key must be {self.KEY_SIZE} bytes")
        
        self.cipher = AESGCM(self.key)
    
    def encrypt(self, data: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt data with AES-256-GCM.
        
        Args:
            data: Plaintext bytes
            
        Returns:
            Tuple of (nonce, ciphertext)
        """
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        ciphertext = self.cipher.encrypt(nonce, data, None)
        return nonce, ciphertext
    
    def decrypt(self, nonce: bytes, ciphertext: bytes) -> bytes:
        """
        Decrypt data with AES-256-GCM.
        
        Args:
            nonce: 12-byte nonce
            ciphertext: Encrypted data
            
        Returns:
            Decrypted plaintext
        """
        return self.cipher.decrypt(nonce, ciphertext, None)
    
    def encrypt_file(self, input_path: Path, output_path: Path) -> None:
        """
        Encrypt a file.
        
        Output format: [12-byte nonce][ciphertext]
        """
        logger.info("Encrypting file", input=str(input_path))
        
        with open(input_path, "rb") as f:
            plaintext = f.read()
        
        nonce, ciphertext = self.encrypt(plaintext)
        
        with open(output_path, "wb") as f:
            f.write(nonce)
            f.write(ciphertext)
        
        logger.info(
            "File encrypted",
            input=str(input_path),
            output=str(output_path),
            original_size=len(plaintext),
            encrypted_size=len(nonce) + len(ciphertext),
        )
    
    def decrypt_file(self, input_path: Path, output_path: Path) -> None:
        """
        Decrypt a file.
        
        Input format: [12-byte nonce][ciphertext]
        """
        logger.info("Decrypting file", input=str(input_path))
        
        with open(input_path, "rb") as f:
            nonce = f.read(self.NONCE_SIZE)
            ciphertext = f.read()
        
        plaintext = self.decrypt(nonce, ciphertext)
        
        with open(output_path, "wb") as f:
            f.write(plaintext)
        
        logger.info(
            "File decrypted",
            input=str(input_path),
            output=str(output_path),
        )
    
    @classmethod
    def generate_key(cls) -> str:
        """Generate a new encryption key (base64 encoded)."""
        key = secrets.token_bytes(cls.KEY_SIZE)
        return base64.b64encode(key).decode()


def encrypt_file(input_path: str, output_path: str, key: Optional[str] = None) -> None:
    """Convenience function to encrypt a file."""
    key_bytes = base64.b64decode(key) if key else None
    encryptor = WeightEncryption(key=key_bytes)
    encryptor.encrypt_file(Path(input_path), Path(output_path))


def decrypt_file(input_path: str, output_path: str, key: Optional[str] = None) -> None:
    """Convenience function to decrypt a file."""
    key_bytes = base64.b64decode(key) if key else None
    encryptor = WeightEncryption(key=key_bytes)
    encryptor.decrypt_file(Path(input_path), Path(output_path))
