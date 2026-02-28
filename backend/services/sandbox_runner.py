import os
import time
import json
import base64
from dataclasses import dataclass

import httpx

try:
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token
except Exception:  # pragma: no cover - optional import for local dev
    Request = None
    id_token = None

try:
    import vertexai
except Exception:  # pragma: no cover - optional import for local dev
    vertexai = None

try:
    from vertexai import types as vertex_types
except Exception:  # pragma: no cover - optional import for local dev
    vertex_types = None


class SandboxRunnerError(RuntimeError):
    pass


@dataclass
class SandboxExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int


def _resolve_runner_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/run"):
        return base
    return f"{base}/run"


def _ensure_google_credentials() -> None:
    existing_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if existing_path and os.path.exists(existing_path):
        return

    json_payload = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
    b64_payload = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64", "").strip()

    if not json_payload and not b64_payload:
        return

    if not json_payload and b64_payload:
        try:
            json_payload = base64.b64decode(b64_payload).decode("utf-8")
        except Exception as exc:
            raise SandboxRunnerError(
                "Failed to decode GOOGLE_APPLICATION_CREDENTIALS_B64"
            ) from exc

    target_path = existing_path or "/tmp/gcp-credentials.json"
    try:
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write(json_payload)
        os.chmod(target_path, 0o600)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = target_path
    except Exception as exc:
        raise SandboxRunnerError("Failed to write Google credentials file") from exc


def _get_id_token(audience: str) -> str:
    _ensure_google_credentials()
    if Request is None or id_token is None:
        raise SandboxRunnerError(
            "google-auth is not available; install google-auth to use ID token auth"
        )
    request = Request()
    return id_token.fetch_id_token(request, audience)


async def run_in_cloud_run(
    code: str,
    language: str,
    timeout_seconds: int,
    max_output_chars: int,
) -> SandboxExecutionResult:
    base_url = os.getenv("SANDBOX_RUNNER_URL", "").strip()
    if not base_url:
        raise SandboxRunnerError("SANDBOX_RUNNER_URL is not set")

    auth_mode = os.getenv("SANDBOX_RUNNER_AUTH_MODE", "idtoken").strip().lower()
    audience = os.getenv("SANDBOX_RUNNER_AUDIENCE", "").strip() or base_url
    api_key = os.getenv("SANDBOX_RUNNER_API_KEY", "").strip()

    headers: dict[str, str] = {"Content-Type": "application/json"}

    if auth_mode == "idtoken":
        headers["Authorization"] = f"Bearer {_get_id_token(audience)}"
    elif auth_mode == "apikey":
        if not api_key:
            raise SandboxRunnerError(
                "SANDBOX_RUNNER_API_KEY is required for apikey auth"
            )
        headers["x-api-key"] = api_key
    elif auth_mode == "none":
        pass
    else:
        raise SandboxRunnerError(f"Unsupported SANDBOX_RUNNER_AUTH_MODE: {auth_mode}")

    payload = {
        "code": code,
        "language": language,
        "timeout_seconds": timeout_seconds,
        "max_output_chars": max_output_chars,
    }

    url = _resolve_runner_url(base_url)
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds + 5) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise SandboxRunnerError(f"Sandbox runner error: {detail}") from exc
    except Exception as exc:
        raise SandboxRunnerError(f"Sandbox runner request failed: {exc}") from exc

    stdout = data.get("stdout", "")
    stderr = data.get("stderr", "")
    exit_code = int(data.get("exit_code", 1))
    timed_out = bool(data.get("timed_out", False))
    duration_ms = int(data.get("duration_ms", (time.time() - start) * 1000))

    return SandboxExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=duration_ms,
    )


def _resolve_vertex_config() -> tuple[str, str]:
    project = (
        os.getenv("SANDBOX_VERTEX_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or ""
    )
    location = (
        os.getenv("SANDBOX_VERTEX_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or os.getenv("GCP_REGION")
        or ""
    )
    return project, location


async def run_in_vertex_code_execution(
    code: str,
    timeout_seconds: int,
) -> SandboxExecutionResult:
    _ensure_google_credentials()
    if vertexai is None:
        raise SandboxRunnerError(
            "vertexai SDK not available; install google-cloud-aiplatform to use Vertex sandbox"
        )

    project, location = _resolve_vertex_config()
    if not project or not location:
        raise SandboxRunnerError(
            "SANDBOX_VERTEX_PROJECT and SANDBOX_VERTEX_LOCATION must be set for Vertex sandbox"
        )
    if location != "us-central1":
        raise SandboxRunnerError(
            "Vertex sandbox execution is only supported in us-central1"
        )

    start = time.time()
    try:
        client = vertexai.Client(project=project, location=location)
        agent_engine_name = os.getenv("SANDBOX_VERTEX_AGENT_ENGINE_NAME", "").strip()

        if not agent_engine_name:
            agent_engine = client.agent_engines.create()
            agent_engine_name = agent_engine.api_resource.name

        sandbox_config = None
        if vertex_types is not None:
            try:
                sandbox_config = vertex_types.CreateAgentEngineSandboxConfig(
                    display_name="goblin-sandbox"
                )
            except Exception:
                sandbox_config = None

        operation = client.agent_engines.sandboxes.create(
            name=agent_engine_name,
            spec={"code_execution_environment": {}},
            config=sandbox_config,
        )
        sandbox_name = operation.response.name

        response = client.agent_engines.sandboxes.execute_code(
            name=sandbox_name,
            input_data={"code": code},
        )

        stdout = ""
        stderr = ""
        outputs = getattr(response, "outputs", []) or []
        for chunk in outputs:
            data = getattr(chunk, "data", b"")
            mime_type = getattr(chunk, "mime_type", "")
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = ""
            if mime_type == "application/json" and text:
                try:
                    payload = json.loads(text)
                    stdout += payload.get("msg_out", "")
                    stderr += payload.get("msg_err", "")
                except Exception:
                    stdout += text
            else:
                stdout += text

        duration_ms = int((time.time() - start) * 1000)
        exit_code = 0 if not stderr else 1

        return SandboxExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=False,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        raise SandboxRunnerError(f"Vertex sandbox execution failed: {exc}") from exc


async def execute_sandbox_code(code: str, language: str) -> SandboxExecutionResult:
    timeout_seconds = int(os.getenv("SANDBOX_RUNNER_TIMEOUT_SECONDS", "20"))
    max_output_chars = int(os.getenv("SANDBOX_RUNNER_OUTPUT_LIMIT", "12000"))
    provider = os.getenv("SANDBOX_RUNNER_PROVIDER", "").strip().lower()

    if language.lower() not in {"python", "py"}:
        raise SandboxRunnerError("Sandbox execution currently supports Python only")

    if not provider:
        project, location = _resolve_vertex_config()
        provider = "vertex_code_execution" if project and location else "cloud_run"

    if provider == "cloud_run":
        return await run_in_cloud_run(code, language, timeout_seconds, max_output_chars)
    if provider == "vertex_code_execution":
        return await run_in_vertex_code_execution(code, timeout_seconds)

    raise SandboxRunnerError(f"Unsupported SANDBOX_RUNNER_PROVIDER: {provider}")
