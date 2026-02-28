import base64
import hashlib
import secrets
from typing import Dict, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature
import json


class WebAuthnPasskey:
    @staticmethod
    def decode_base64url(data: str) -> bytes:
        """Decode base64url encoded string to bytes"""
        # Add padding if needed
        missing_padding = len(data) % 4
        if missing_padding:
            data += "=" * (4 - missing_padding)
        return base64.urlsafe_b64decode(data)

    @staticmethod
    def encode_base64url(data: bytes) -> str:
        """Encode bytes to base64url string"""
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    @staticmethod
    def parse_authenticator_data(auth_data: bytes) -> Dict[str, Any]:
        """Parse authenticator data from WebAuthn response"""
        if len(auth_data) < 37:
            raise ValueError("Authenticator data too short")

        # Parse RP ID hash (32 bytes)
        rp_id_hash = auth_data[:32]

        # Parse flags (1 byte)
        flags = auth_data[32]

        # Parse sign count (4 bytes)
        sign_count = int.from_bytes(auth_data[33:37], byteorder="big")

        # Parse attested credential data if present
        attested_credential_data = None
        remaining_data = auth_data[37:]

        if remaining_data:
            # AAGUID (16 bytes)
            aaguid = remaining_data[:16]
            # Credential ID length (2 bytes)
            cred_id_len = int.from_bytes(remaining_data[16:18], byteorder="big")
            # Credential ID
            credential_id = remaining_data[18 : 18 + cred_id_len]
            # Public key (COSE format)
            public_key_cose = remaining_data[18 + cred_id_len :]

            attested_credential_data = {
                "aaguid": aaguid.hex(),
                "credential_id": credential_id,
                "public_key_cose": public_key_cose,
            }

        return {
            "rp_id_hash": rp_id_hash,
            "flags": flags,
            "sign_count": sign_count,
            "attested_credential_data": attested_credential_data,
        }

    @staticmethod
    def parse_cose_public_key(cose_key: bytes) -> ec.EllipticCurvePublicKey:
        """Parse COSE formatted public key"""
        # This is a simplified implementation
        # In production, use a proper COSE library
        try:
            # For ES256, the key should be in uncompressed format
            if len(cose_key) == 65 and cose_key[0] == 0x04:
                # Uncompressed EC point
                x = int.from_bytes(cose_key[1:33], byteorder="big")
                y = int.from_bytes(cose_key[33:65], byteorder="big")

                # Create EC public key
                from cryptography.hazmat.primitives.asymmetric import ec
                from cryptography.hazmat.backends import default_backend

                public_numbers = ec.EllipticCurvePublicNumbers(
                    x=x, y=y, curve=ec.SECP256R1()
                )
                return public_numbers.public_key(default_backend())
        except Exception:
            pass

        raise ValueError("Unsupported or invalid COSE public key format")

    @staticmethod
    def verify_signature(
        public_key: ec.EllipticCurvePublicKey,
        signature: bytes,
        authenticator_data: bytes,
        client_data_json: bytes,
    ) -> bool:
        """Verify WebAuthn signature"""
        try:
            # Create the data to verify (authenticator_data + SHA256(client_data_json))
            client_data_hash = hashlib.sha256(client_data_json).digest()
            signed_data = authenticator_data + client_data_hash

            # Verify signature
            public_key.verify(signature, signed_data, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False
        except Exception:
            return False

    @staticmethod
    async def verify_passkey_authentication(
        credential_id: str,
        stored_public_key: str,
        authenticator_data_b64: str,
        client_data_json_b64: str,
        signature_b64: str,
        challenge: str,
        origin: str,
    ) -> bool:
        """
        Verify a passkey authentication response
        """
        try:
            # Decode base64url data
            authenticator_data = WebAuthnPasskey.decode_base64url(
                authenticator_data_b64
            )
            client_data_json = WebAuthnPasskey.decode_base64url(client_data_json_b64)
            signature = WebAuthnPasskey.decode_base64url(signature_b64)

            # Parse client data JSON
            client_data = json.loads(client_data_json.decode("utf-8"))

            # Verify challenge
            if client_data.get("challenge") != WebAuthnPasskey.encode_base64url(
                challenge.encode()
            ):
                return False

            # Verify origin
            if client_data.get("origin") != origin:
                return False

            # Verify type
            if client_data.get("type") != "webauthn.get":
                return False

            # Parse authenticator data
            # Parse authenticator data to ensure it is valid (not used directly here)
            WebAuthnPasskey.parse_authenticator_data(authenticator_data)

            # Parse stored public key
            stored_key_bytes = WebAuthnPasskey.decode_base64url(stored_public_key)
            public_key = WebAuthnPasskey.parse_cose_public_key(stored_key_bytes)

            # Verify signature
            return WebAuthnPasskey.verify_signature(
                public_key, signature, authenticator_data, client_data_json
            )

        except Exception:
            return False

    @staticmethod
    def generate_challenge() -> str:
        """Generate a random challenge for WebAuthn"""
        return secrets.token_urlsafe(32)
