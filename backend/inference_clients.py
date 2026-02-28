# goblin/inference_clients.py
import os
import httpx
import time
import logging

# Use existing Kamatera infrastructure
KAMATERA_HOST = os.getenv("KAMATERA_HOST", "66.55.77.147")
KAMATERA_RAPTOR_URL = (
    f"http://{KAMATERA_HOST}:8080/v1/generate"  # Raptor Mini on port 8080
)
KAMATERA_API_KEY = os.getenv("KAMATERA_LLM_API_KEY", "")  # Reuse existing API key
LOCAL_RAPTOR_API_KEY = os.getenv("LOCAL_RAPTOR_API_KEY", "")
# Support LOCAL_LLM_API_KEY (proxy) as alternate env var for local LLM authentication
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")
# Prefer LOCAL_LLM_PROXY_URL (more generic), but fall back to LOCAL_RAPTOR_URL
LOCAL_URL = os.getenv(
    "LOCAL_LLM_PROXY_URL",
    os.getenv("LOCAL_RAPTOR_URL", "http://127.0.0.1:8080/v1/generate"),
)
FALLBACK_TTL = int(os.getenv("RAPTOR_FALLBACK_TTL", "30"))  # seconds

# simple circuit-breaker state
_last_fail = 0.0
_backoff_duration = 10.0


def _kamatera_available():
    return (time.time() - _last_fail) > _backoff_duration


async def call_raptor(prompt: str, max_tokens: int = 128, timeout: float = 6.0):
    global _last_fail
    # try Kamatera first (existing infrastructure)
    headers = {}
    if KAMATERA_API_KEY:
        headers["X-API-Key"] = KAMATERA_API_KEY

    payload = {"prompt": prompt, "max_tokens": max_tokens}
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Kamatera attempt
        if _kamatera_available():
            try:
                r = await client.post(
                    KAMATERA_RAPTOR_URL, json=payload, headers=headers
                )
                if r.status_code == 200:
                    return r.json()
                else:
                    # transient failure: mark fail and fallthrough
                    _last_fail = time.time()
            except Exception as e:
                logging.exception("Kamatera raptor call failed: %s", e)
                _last_fail = time.time()

        # fallback local
        try:
            local_headers = {}
            if LOCAL_RAPTOR_API_KEY:
                local_headers["X-API-Key"] = LOCAL_RAPTOR_API_KEY
                # Also include Authorization Bearer for proxies that prefer it
                local_headers["Authorization"] = f"Bearer {LOCAL_RAPTOR_API_KEY}"
            elif LOCAL_LLM_API_KEY:
                # Allow proxy to use the more generic LOCAL_LLM_API_KEY env var
                local_headers["X-API-Key"] = LOCAL_LLM_API_KEY
                local_headers["Authorization"] = f"Bearer {LOCAL_LLM_API_KEY}"
            # allow the same headers to be reused as a convenience
            if not local_headers and headers:
                # reuse kamatera key for local if not separately provided
                local_headers = headers
            r = await client.post(
                LOCAL_URL, json=payload, headers=local_headers, timeout=timeout
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logging.exception("Local raptor call failed: %s", e)

    # both failed
    return {"ok": False, "error": "raptor-unavailable"}
