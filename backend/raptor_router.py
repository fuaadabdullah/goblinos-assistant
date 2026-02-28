from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import os
import sys
from pathlib import Path

# Add GoblinOS to path for raptor import (optional in production containers).
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "GoblinOS"))
try:
    from raptor_mini import raptor  # type: ignore
except Exception:  # pragma: no cover - GoblinOS isn't shipped in Fly image

    class _RaptorStub:
        running = False
        ini_path = "config/raptor.ini"

        class _Cfg:
            def get(self, *_args, fallback: str = "logs/raptor.log", **_kwargs):
                return fallback

        cfg = _Cfg()

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

        def trace(self, fn):  # decorator
            return fn

    raptor = _RaptorStub()

router = APIRouter(prefix="/raptor", tags=["raptor"])


class LogsRequest(BaseModel):
    max_chars: int = 1000


@router.post("/start")
async def raptor_start():
    """Start raptor monitoring with real RaptorMini system"""
    try:
        raptor.start()
        return {"running": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start raptor: {str(e)}")


@router.post("/stop")
async def raptor_stop():
    """Stop raptor monitoring"""
    try:
        raptor.stop()
        return {"running": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop raptor: {str(e)}")


@router.get("/status")
async def raptor_status():
    """Get raptor status from real monitoring system"""
    try:
        return {
            "running": bool(raptor.running),
            "config_file": getattr(raptor, "ini_path", "config/raptor.ini"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get raptor status: {str(e)}"
        )


@router.post("/logs")
async def raptor_logs(request: LogsRequest):
    """Get raptor logs from configured log file"""
    try:
        logfile = raptor.cfg.get("logging", "file", fallback="logs/raptor.log")

        if not os.path.exists(logfile):
            return {"log_tail": "Log file not found. Raptor may not be running yet."}

        with open(logfile, "rb") as f:
            f.seek(0, os.SEEK_END)
            length = f.tell()
            # Read last request.max_chars bytes (approx), ensure not negative
            read_from = max(0, length - request.max_chars)
            f.seek(read_from)
            data = f.read().decode("utf-8", errors="replace")
            return {"log_tail": data}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read raptor logs: {str(e)}"
        )


@router.get("/logs")
async def raptor_logs_get(tail: int = Query(1000, ge=1, le=50000)):
    """GET wrapper for frontend compatibility (`/raptor/logs?tail=N`)."""
    return await raptor_logs(LogsRequest(max_chars=tail))


@router.get("/demo/{value}")
async def raptor_demo(value: str):
    """Demo endpoint for testing raptor exception tracing"""
    try:
        # Use the @raptor.trace decorator to test exception logging
        if value.lower() == "boom":

            @raptor.trace
            def raise_demo_error():
                raise RuntimeError("Demo error triggered by /demo/boom")

            try:
                raise_demo_error()
            except RuntimeError:
                # Expected - we just test trace logging
                pass

        return {
            "result": f"Demo executed with value: {value}",
            "traced": value.lower() == "boom",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


@router.post("/demo/{value}")
async def raptor_demo_post(value: str):
    """POST wrapper for frontend compatibility."""
    return await raptor_demo(value)
