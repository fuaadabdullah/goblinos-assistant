"""
Real Goblin Task Execution Service

Integrates with GoblinOS automation system to execute real tasks.
Supports both API-based and shell-based integration.
"""

import os
import sys
import subprocess
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)


class GoblinExecutor:
    """Executes tasks using the GoblinOS system"""

    def __init__(self):
        # Try to use API-based integration first (production-ready)
        self.api_url = os.getenv("GOBLINOS_API_URL")
        self.api_key = os.getenv("GOBLINOS_API_KEY")

        # Fallback to shell-based integration for local development
        self.cli_path = os.getenv("GOBLINOS_CLI_PATH")

        # Legacy fallback: relative path (for backward compatibility)
        if not self.cli_path and not self.api_url:
            legacy_path = Path(__file__).parent.parent.parent.parent.parent / "GoblinOS"
            if legacy_path.exists():
                self.cli_path = str(legacy_path / "goblin-cli.sh")
                self.goblin_os_path = legacy_path
                logger.warning(
                    "Using legacy relative path for GoblinOS. "
                    "Please set GOBLINOS_API_URL or GOBLINOS_CLI_PATH environment variable."
                )
        elif self.cli_path:
            cli_path_obj = Path(self.cli_path)
            if cli_path_obj.is_file():
                self.goblin_os_path = cli_path_obj.parent
            else:
                # CLI path might be to the goblin-cli.sh directly
                self.goblin_os_path = cli_path_obj
                if not (self.goblin_os_path / "goblin-cli.sh").exists():
                    raise FileNotFoundError(
                        f"goblin-cli.sh not found at {self.goblin_os_path}"
                    )
                self.cli_path = str(self.goblin_os_path / "goblin-cli.sh")

        # Determine integration mode
        self.use_api = bool(self.api_url)
        self.use_shell = bool(self.cli_path)

        if not self.use_api and not self.use_shell:
            raise FileNotFoundError(
                "GoblinOS not configured. Please set either:\n"
                "  - GOBLINOS_API_URL for API-based integration (recommended), or\n"
                "  - GOBLINOS_CLI_PATH for shell-based integration (local dev)"
            )

    async def list_available_goblins(self) -> Dict[str, Any]:
        """List all available goblins from goblins.yaml or API"""
        try:
            if self.use_api:
                return await self._list_goblins_via_api()
            else:
                return await self._list_goblins_via_shell()
        except Exception as e:
            logger.error(f"Failed to list goblins: {e}")
            return {"success": False, "error": str(e)}

    async def _list_goblins_via_api(self) -> Dict[str, Any]:
        """List goblins via HTTP API"""
        import aiohttp

        url = f"{self.api_url}/api/goblins/list"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "goblins": data.get("goblins", []),
                            "mode": "api",
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"API returned {response.status}: {error_text}",
                        }
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {"success": False, "error": f"API call failed: {str(e)}"}

    async def _list_goblins_via_shell(self) -> Dict[str, Any]:
        """List goblins via shell execution"""
        try:
            result = await self._run_command(
                ["bash", str(self.cli_path), "list"], timeout=10
            )
            return {
                "success": True,
                "goblins": self._parse_goblin_list(result["stdout"]),
                "mode": "shell",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_goblin_list(self, output: str) -> list:
        """Parse goblin list output"""
        # Expected format from goblin-cli.sh list command
        goblins = []
        for line in output.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("Available"):
                # Parse format: "goblin-id - Description"
                if " - " in line:
                    goblin_id, description = line.split(" - ", 1)
                    goblins.append(
                        {"id": goblin_id.strip(), "description": description.strip()}
                    )
        return goblins

    async def execute_goblin(
        self,
        goblin_id: str,
        task_description: str,
        code: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a goblin task

        Args:
            goblin_id: The goblin tool ID to execute
            task_description: Human-readable description of the task
            code: Optional code to execute (for code-based goblins)
            dry_run: If True, simulate execution without making changes

        Returns:
            Dict with execution results
        """
        try:
            if self.use_api:
                return await self._execute_goblin_via_api(
                    goblin_id, task_description, code, dry_run
                )
            else:
                return await self._execute_goblin_via_shell(
                    goblin_id, task_description, code, dry_run
                )
        except Exception as e:
            return {
                "success": False,
                "goblin_id": goblin_id,
                "error": str(e),
                "task_description": task_description,
            }

    async def _execute_goblin_via_api(
        self,
        goblin_id: str,
        task_description: str,
        code: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute goblin via HTTP API"""
        import aiohttp

        url = f"{self.api_url}/api/goblins/execute"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "goblinId": goblin_id,
            "command": task_description,
            "parameters": {"code": code} if code else {},
            "context": {"dryRun": dry_run},
        }

        try:
            start_time = datetime.utcnow()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers, timeout=300
                ) as response:
                    end_time = datetime.utcnow()
                    execution_time = (end_time - start_time).total_seconds()

                    if response.status == 200:
                        data = await response.json()
                        return {
                            **data,
                            "execution_time_seconds": execution_time,
                            "mode": "api",
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "goblin_id": goblin_id,
                            "error": f"API returned {response.status}: {error_text}",
                            "task_description": task_description,
                        }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "goblin_id": goblin_id,
                "error": "Task execution timeout (exceeded 5 minutes)",
                "task_description": task_description,
            }
        except Exception as e:
            return {
                "success": False,
                "goblin_id": goblin_id,
                "error": f"API call failed: {str(e)}",
                "task_description": task_description,
            }

    async def _execute_goblin_via_shell(
        self,
        goblin_id: str,
        task_description: str,
        code: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute goblin via shell"""
        try:
            # Prepare command
            cmd = ["bash", str(self.cli_path), "run"]

            if dry_run:
                cmd.append("--dry")

            cmd.append(goblin_id)

            # Execute command
            start_time = datetime.utcnow()
            result = await self._run_command(cmd, timeout=300)  # 5 minute timeout
            end_time = datetime.utcnow()

            execution_time = (end_time - start_time).total_seconds()

            return {
                "success": result["returncode"] == 0,
                "goblin_id": goblin_id,
                "task_description": task_description,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
                "execution_time_seconds": execution_time,
                "dry_run": dry_run,
                "timestamp": start_time.isoformat(),
                "mode": "shell",
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "goblin_id": goblin_id,
                "error": "Task execution timeout (exceeded 5 minutes)",
                "task_description": task_description,
            }
        except Exception as e:
            return {
                "success": False,
                "goblin_id": goblin_id,
                "error": str(e),
                "task_description": task_description,
            }

    async def execute_custom_script(
        self, script_content: str, working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a custom bash script

        Args:
            script_content: The bash script content to execute
            working_dir: Optional working directory

        Returns:
            Dict with execution results
        """
        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write("#!/bin/bash\n")
                f.write("set -e\n")  # Exit on error
                f.write(script_content)
                script_path = f.name

            # Make executable
            os.chmod(script_path, 0o755)

            try:
                # Execute script
                start_time = datetime.utcnow()
                result = await self._run_command(
                    ["bash", script_path], timeout=300, cwd=working_dir
                )
                end_time = datetime.utcnow()

                execution_time = (end_time - start_time).total_seconds()

                return {
                    "success": result["returncode"] == 0,
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "returncode": result["returncode"],
                    "execution_time_seconds": execution_time,
                    "timestamp": start_time.isoformat(),
                }
            finally:
                # Clean up temp file
                try:
                    os.unlink(script_path)
                except:
                    pass

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_command(
        self, cmd: list, timeout: int = 60, cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a shell command asynchronously

        Args:
            cmd: Command and arguments as list
            timeout: Timeout in seconds
            cwd: Working directory

        Returns:
            Dict with stdout, stderr, and returncode
        """
        # Determine working directory
        work_dir = cwd
        if not work_dir and hasattr(self, "goblin_os_path"):
            work_dir = str(self.goblin_os_path)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "returncode": process.returncode,
            }
        except asyncio.TimeoutError:
            # Kill the process if it times out
            process.kill()
            await process.wait()
            raise

    async def validate_goblin(self, goblin_id: str) -> Dict[str, Any]:
        """
        Validate that a goblin exists and can be executed

        Args:
            goblin_id: The goblin tool ID to validate

        Returns:
            Dict with validation result
        """
        goblins_result = await self.list_available_goblins()

        if not goblins_result["success"]:
            return {"valid": False, "error": "Failed to list goblins"}

        goblin_ids = [g["id"] for g in goblins_result["goblins"]]

        if goblin_id in goblin_ids:
            return {"valid": True, "goblin_id": goblin_id}
        else:
            return {
                "valid": False,
                "error": f"Goblin '{goblin_id}' not found",
                "available_goblins": goblin_ids,
            }


# Singleton instance
_executor: Optional[GoblinExecutor] = None


def get_goblin_executor() -> GoblinExecutor:
    """Get or create singleton GoblinExecutor instance"""
    global _executor
    if _executor is None:
        try:
            _executor = GoblinExecutor()
            mode = "API" if _executor.use_api else "shell"
            logger.info(f"GoblinExecutor initialized in {mode} mode")
        except FileNotFoundError as exc:
            # When GoblinOS isn't configured or available, fall back to a stub
            # so the frontend can still exercise the execution flows without hard 500s.
            logger.warning("GoblinOS not available; using stub executor: %s", exc)
            _executor = _StubGoblinExecutor(str(exc))  # type: ignore[assignment]
    return _executor


class _StubGoblinExecutor:
    """Best-effort executor used when GoblinOS isn't available in the image."""

    def __init__(self, reason: str):
        self._reason = reason

    async def list_available_goblins(self) -> Dict[str, Any]:
        goblins = [
            {"id": "docs-writer", "description": "Documentation Writer"},
            {"id": "code-writer", "description": "Code Writer"},
            {"id": "search-goblin", "description": "Search Specialist"},
            {"id": "analyze-goblin", "description": "Data Analyst"},
            {"id": "general-goblin", "description": "General Assistant"},
        ]
        return {
            "success": True,
            "goblins": goblins,
            "stub": True,
            "reason": self._reason,
        }

    async def validate_goblin(self, goblin_id: str) -> Dict[str, Any]:
        goblins_result = await self.list_available_goblins()
        goblin_ids = [g["id"] for g in goblins_result.get("goblins", [])]
        if goblin_id in goblin_ids:
            return {"valid": True, "goblin_id": goblin_id, "stub": True}
        return {
            "valid": False,
            "error": f"Goblin '{goblin_id}' not found",
            "available_goblins": goblin_ids,
            "stub": True,
        }

    async def execute_goblin(
        self,
        goblin_id: str,
        task_description: str,
        code: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        # Never execute user-provided code in this stub.
        now = datetime.utcnow()
        stdout_lines = [
            f"[stub] GoblinOS is not available in this deployment.",
            f"[stub] Reason: {self._reason}",
            f"[stub] goblin_id={goblin_id}",
            f"[stub] dry_run={dry_run}",
            f"[stub] task={task_description}",
        ]
        if code:
            stdout_lines.append("[stub] NOTE: code execution is disabled in stub mode.")

        return {
            "success": True,
            "goblin_id": goblin_id,
            "task_description": task_description,
            "stdout": "\n".join(stdout_lines) + "\n",
            "stderr": "",
            "returncode": 0,
            "execution_time_seconds": 0.0,
            "dry_run": dry_run,
            "timestamp": now.isoformat(),
            "stub": True,
        }

    async def execute_custom_script(
        self, script_content: str, working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "Custom script execution is disabled (GoblinOS not available).",
            "stub": True,
        }
