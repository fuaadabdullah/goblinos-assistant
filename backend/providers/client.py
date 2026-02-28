from typing import Any, Dict, Optional
import os
import httpx
import logging
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential, RetryError

logger = logging.getLogger("goblin-assistant.providers")


class ProviderClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def post(self, path: str, payload: Dict[str, Any], retries: int = 3, backoff: int = 1) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/') if path else ''}".rstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    # Some providers may return raw text
                    return {"text": resp.text}

        retryer = AsyncRetrying(
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            stop=stop_after_attempt(retries),
            wait=wait_exponential(multiplier=backoff, min=backoff, max=30),
            reraise=True,
        )

        try:
            async for attempt in retryer:
                with attempt:
                    result = await _call()
                    return result
        except RetryError as e:
            # If retries are exhausted, log and raise a more helpful error with context
            last = e.last_attempt if hasattr(e, 'last_attempt') else None
            err = e
            logger.exception("Provider call failed after retries", extra={"url": url, "error": str(err)})
            raise

    async def get(self, path: str, retries: int = 2, backoff: int = 1) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/') if path else ''}".rstrip("/")
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    return {"text": resp.text}

        retryer = AsyncRetrying(
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            stop=stop_after_attempt(retries),
            wait=wait_exponential(multiplier=backoff, min=backoff, max=20),
            reraise=True,
        )

        try:
            async for attempt in retryer:
                with attempt:
                    result = await _call()
                    return result
        except RetryError as e:
            logger.exception("Provider GET failed after retries", extra={"url": url, "error": str(e)})
            raise
