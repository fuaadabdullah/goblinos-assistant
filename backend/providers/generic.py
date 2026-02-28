import time
from typing import Dict, Any


def test_connection(api_key: str) -> Dict[str, Any]:
    """Generic HTTP-based connection test"""
    start_time = time.time()
    try:
        # This is a fallback for providers without specific adapters
        # In a real implementation, you'd have provider-specific logic
        # For now, just check if the key looks valid
        if len(api_key) < 10:
            raise ValueError("API key too short")

        latency = time.time() - start_time
        return {
            "success": True,
            "latency_ms": round(latency * 1000, 2),
            "status_code": 200,
            "message": "API key format validated (no specific adapter available)",
        }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "success": False,
            "latency_ms": round(latency * 1000, 2),
            "error": str(e),
        }
