#!/usr/bin/env python3
"""Check RunPod deployments and endpoints."""

import asyncio
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load env
load_dotenv(Path(__file__).parent.parent / ".env")


async def check_runpod_deployments():
    """Check for deployed endpoints and pods on RunPod."""
    api_key = os.getenv("RUNPOD_API_KEY", "")

    if not api_key:
        print("RUNPOD_API_KEY not set")
        return False

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.runpod.io/graphql",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": """
                    query {
                        myself {
                            id
                            email
                            endpoints {
                                id
                                name
                                templateId
                                gpuIds
                                workersMin
                                workersMax
                                idleTimeout
                            }
                            pods {
                                id
                                name
                                gpuCount
                            }
                        }
                    }
                """
            },
        )

        if response.status_code == 200:
            data = response.json()
            myself = data.get("data", {}).get("myself", {})

            endpoints = myself.get("endpoints", [])
            pods = myself.get("pods", [])
            email = myself.get("email", "")

            print("\n📡 RUNPOD DEPLOYMENTS STATUS")
            print("=" * 50)
            print(f"Account: {email}")

            print(f"\n🔌 Serverless Endpoints: {len(endpoints)}")
            if endpoints:
                for ep in endpoints:
                    print(f"   - {ep.get('name', 'unnamed')}")
                    print(f"     ID: {ep.get('id')}")
                    print(f"     GPUs: {ep.get('gpuIds', [])}")
                    print(
                        f"     Workers: {ep.get('workersMin', 0)}-{ep.get('workersMax', 0)}"
                    )
                    print(f"     Idle Timeout: {ep.get('idleTimeout', 0)}s")
            else:
                print("   No serverless endpoints deployed.")

            print(f"\n🖥️  Active Pods: {len(pods)}")
            if pods:
                for pod in pods:
                    print(f"   - {pod.get('name', 'unnamed')}")
                    print(f"     ID: {pod.get('id')}")
                    print(f"     GPU Count: {pod.get('gpuCount', 0)}")
            else:
                print("   No active pods running.")

            if not endpoints and not pods:
                print("\n⚠️  No models deployed on RunPod yet.")
                print("   To deploy a serverless endpoint:")
                print("   1. Go to https://www.runpod.io/console/serverless")
                print("   2. Create a new endpoint with vLLM or TGI template")
                print(
                    "   3. Configure your model (e.g., mistralai/Mistral-7B-Instruct)"
                )

            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False


if __name__ == "__main__":
    asyncio.run(check_runpod_deployments())
