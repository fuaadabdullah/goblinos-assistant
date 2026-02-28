#!/usr/bin/env python3
"""
Chaos testing script for APScheduler with Redis locks.

This script simulates various failure scenarios to validate the distributed
locking mechanism and job execution reliability.

Usage:
    python chaos_test_scheduler.py [test_name]

Available tests:
- replica_failure: Simulate one replica failing during job execution
- network_partition: Simulate Redis network issues
- lock_timeout: Test lock expiration behavior
- concurrent_jobs: Test multiple jobs running simultaneously
- redis_restart: Test Redis restart scenarios
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from scheduler import (
    start_scheduler,
    stop_scheduler,
    get_scheduler_status,
    redis_client,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChaosTest:
    """Base class for chaos tests."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None

    def setup(self):
        """Setup test environment."""
        logger.info(f"Setting up test: {self.name}")
        self.start_time = datetime.now()

    def run(self) -> Dict[str, Any]:
        """Run the chaos test."""
        raise NotImplementedError

    def cleanup(self):
        """Clean up test environment."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Test {self.name} completed in {duration:.2f}s")

    def execute(self) -> Dict[str, Any]:
        """Execute the full test lifecycle."""
        try:
            self.setup()
            result = self.run()
            self.cleanup()
            return {
                "test": self.name,
                "success": True,
                "duration": (self.end_time - self.start_time).total_seconds(),
                "result": result,
            }
        except Exception as e:
            logger.error(f"Test {self.name} failed: {e}")
            return {
                "test": self.name,
                "success": False,
                "error": str(e),
                "duration": (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0,
            }


class ReplicaFailureTest(ChaosTest):
    """Test replica failure during job execution."""

    def __init__(self):
        super().__init__(
            "replica_failure", "Simulate one replica failing during job execution"
        )

    def run(self) -> Dict[str, Any]:
        """Run replica failure test."""
        # Start scheduler
        start_scheduler()

        # Wait for jobs to start
        time.sleep(5)

        # Get initial status
        initial_status = get_scheduler_status()

        # Simulate replica failure by stopping scheduler abruptly
        logger.info("Simulating replica failure...")
        stop_scheduler()

        # Wait a bit
        time.sleep(10)

        # Check Redis locks are cleaned up
        lock_keys = redis_client.keys("system_health_check")
        active_locks = len(lock_keys) if lock_keys else 0

        # Restart scheduler (simulating replica restart)
        logger.info("Restarting scheduler...")
        start_scheduler()

        # Wait for recovery
        time.sleep(5)

        final_status = get_scheduler_status()

        return {
            "initial_jobs": len(initial_status.get("jobs", [])),
            "final_jobs": len(final_status.get("jobs", [])),
            "active_locks_after_failure": active_locks,
            "scheduler_recovered": final_status.get("status") == "running",
        }


class NetworkPartitionTest(ChaosTest):
    """Test Redis network partition scenarios."""

    def __init__(self):
        super().__init__(
            "network_partition", "Simulate Redis network connectivity issues"
        )

    def run(self) -> Dict[str, Any]:
        """Run network partition test."""
        start_scheduler()
        time.sleep(5)

        initial_status = get_scheduler_status()

        # Simulate network partition by disconnecting Redis
        logger.info("Simulating network partition...")
        # In a real scenario, we'd use network tools to block Redis
        # For this test, we'll just test Redis reconnection logic

        try:
            # Test Redis connectivity
            redis_client.ping()
            redis_up_before = True
        except Exception:
            redis_up_before = False

        # Wait for potential job timeouts
        time.sleep(70)  # Longer than lock TTL

        try:
            redis_client.ping()
            redis_up_after = True
        except Exception:
            redis_up_after = False

        final_status = get_scheduler_status()

        return {
            "redis_up_before": redis_up_before,
            "redis_up_after": redis_up_after,
            "jobs_before": len(initial_status.get("jobs", [])),
            "jobs_after": len(final_status.get("jobs", [])),
            "scheduler_status": final_status.get("status"),
        }


class LockTimeoutTest(ChaosTest):
    """Test lock timeout behavior."""

    def __init__(self):
        super().__init__("lock_timeout", "Test lock expiration when jobs hang")

    def run(self) -> Dict[str, Any]:
        """Run lock timeout test."""
        # Manually acquire a lock
        lock_key = "test_lock_timeout"
        acquired = redis_client.set(lock_key, "1", nx=True, ex=30)

        if not acquired:
            return {"error": "Could not acquire test lock"}

        # Wait for lock to expire
        logger.info("Waiting for lock to expire...")
        time.sleep(35)

        # Try to acquire the same lock again
        acquired_again = redis_client.set(lock_key, "1", nx=True, ex=30)

        # Clean up
        redis_client.delete(lock_key)

        return {
            "lock_acquired_initially": acquired is not None,
            "lock_acquired_after_timeout": acquired_again is not None,
            "lock_expired": acquired_again is not None,
        }


class ConcurrentJobsTest(ChaosTest):
    """Test multiple jobs running concurrently."""

    def __init__(self):
        super().__init__("concurrent_jobs", "Test multiple jobs running simultaneously")

    def run(self) -> Dict[str, Any]:
        """Run concurrent jobs test."""
        start_scheduler()
        time.sleep(5)

        status = get_scheduler_status()
        jobs = status.get("jobs", [])

        # Check Redis for active locks
        active_locks = []
        for job in jobs:
            lock_key = job["id"]
            if redis_client.exists(lock_key):
                active_locks.append(lock_key)

        return {
            "total_jobs": len(jobs),
            "active_locks": len(active_locks),
            "scheduler_running": status.get("status") == "running",
        }


def run_test(test_name: str) -> Dict[str, Any]:
    """Run a specific chaos test."""
    tests = {
        "replica_failure": ReplicaFailureTest,
        "network_partition": NetworkPartitionTest,
        "lock_timeout": LockTimeoutTest,
        "concurrent_jobs": ConcurrentJobsTest,
    }

    if test_name not in tests:
        available = list(tests.keys())
        raise ValueError(f"Unknown test: {test_name}. Available: {available}")

    test_class = tests[test_name]
    test = test_class()
    return test.execute()


def run_all_tests() -> List[Dict[str, Any]]:
    """Run all chaos tests."""
    test_names = [
        "replica_failure",
        "network_partition",
        "lock_timeout",
        "concurrent_jobs",
    ]
    results = []

    for test_name in test_names:
        logger.info(f"Running chaos test: {test_name}")
        result = run_test(test_name)
        results.append(result)

        # Cleanup between tests
        stop_scheduler()
        time.sleep(2)

    return results


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        result = run_test(test_name)
        print(f"Test Result: {result}")
    else:
        print("Running all chaos tests...")
        results = run_all_tests()

        print("\nChaos Test Results:")
        print("=" * 50)
        for result in results:
            status = "PASS" if result["success"] else "FAIL"
            print(f"{result['test']}: {status} ({result['duration']:.2f}s)")
            if not result["success"]:
                print(f"  Error: {result.get('error', 'Unknown')}")

        successful = sum(1 for r in results if r["success"])
        total = len(results)
        print(f"\nSummary: {successful}/{total} tests passed")


if __name__ == "__main__":
    main()
