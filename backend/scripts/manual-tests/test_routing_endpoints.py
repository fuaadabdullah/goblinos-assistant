#!/usr/bin/env python
"""
Test script for routing endpoints.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests

# Add the backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))


def test_routing_endpoints():
    base_url = "http://localhost:8001"

    print("Testing routing endpoints...")

    try:
        # Test health endpoint
        print("\n1. Testing /health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

        # Test providers endpoint
        print("\n2. Testing /routing/providers endpoint...")
        response = requests.get(f"{base_url}/routing/providers")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            providers = response.json()
            print(f"Found {len(providers)} providers:")
            for provider in providers:
                print(
                    f"  - {provider['name']}: {provider['display_name']} (active: {provider['is_active']})"
                )
        else:
            print(f"Error: {response.text}")

        # Test providers by capability
        print("\n3. Testing /routing/providers/chat endpoint...")
        response = requests.get(f"{base_url}/routing/providers/chat")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            providers = response.json()
            print(f"Found {len(providers)} providers for 'chat' capability:")
            for provider in providers:
                print(f"  - {provider['name']}: {provider['display_name']}")
        else:
            print(f"Error: {response.text}")

        # Test routing endpoint
        print("\n4. Testing /routing/route endpoint...")
        route_request = {
            "capability": "chat",
            "requirements": {"model": "gpt-4", "max_tokens": 1000},
        }
        response = requests.post(f"{base_url}/routing/route", json=route_request)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Routing result: {result}")
        else:
            print(f"Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {e}")


def start_server():
    """Start the test server in background"""
    script_dir = Path(__file__).resolve().parent
    server_script = script_dir / "test_routing_server.py"
    cmd = [sys.executable, str(server_script)]
    return subprocess.Popen(cmd, cwd=str(BACKEND_DIR))


def stop_server(process):
    """Stop the server"""
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    server_process = None
    try:
        print("Starting test server...")
        server_process = start_server()
        time.sleep(3)  # Wait for server to start

        test_routing_endpoints()

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        print("\nStopping server...")
        stop_server(server_process)
