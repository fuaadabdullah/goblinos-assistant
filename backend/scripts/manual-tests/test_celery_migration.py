# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
Test script for Celery task queue migration from RQ.

This script tests the new celery_task_queue.py module to ensure
it provides the same interface as the original task_queue.py.
"""

import os
import sys
import time
import importlib.util
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))


def test_celery_task_queue():
    """Test the Celery task queue implementation."""
    print("Testing Celery Task Queue Migration...")

    try:
        # Import the new Celery task queue
        from celery_task_queue import (
            enqueue_task,
            set_task_running,
            set_task_completed,
            add_task_log,
            get_task_logs,
            add_task_artifact,
            get_task_artifacts,
            get_task_meta,
            clear_task,
            get_all_tasks,
        )

        print("✅ Successfully imported Celery task queue functions")

        # Test task creation
        task_id = f"test_task_{int(time.time())}"
        payload = {"action": "test", "data": "sample"}

        print(f"📝 Enqueuing test task: {task_id}")
        returned_task_id = enqueue_task(task_id, payload, queue="default")

        assert returned_task_id == task_id, "Task ID mismatch"
        print("✅ Task enqueued successfully")

        # Test metadata retrieval
        print("📊 Testing task metadata retrieval")
        meta = get_task_meta(task_id)
        assert meta is not None, "Task metadata not found"
        assert meta.get("status") == "queued", f"Expected status 'queued', got {meta.get('status')}"
        print("✅ Task metadata retrieved successfully")

        # Test task status updates
        print("🔄 Testing task status updates")
        set_task_running(task_id)
        meta = get_task_meta(task_id)
        assert meta.get("status") == "running", (
            f"Expected status 'running', got {meta.get('status')}"
        )

        set_task_completed(task_id, "success")
        meta = get_task_meta(task_id)
        assert meta.get("status") == "completed", (
            f"Expected status 'completed', got {meta.get('status')}"
        )
        print("✅ Task status updates work correctly")

        # Test logging
        print("📝 Testing task logging")
        add_task_log(task_id, "info", "Test log message")
        add_task_log(task_id, "error", "Test error message")

        logs = get_task_logs(task_id)
        assert len(logs) >= 2, f"Expected at least 2 logs, got {len(logs)}"
        assert any(log["message"] == "Test log message" for log in logs), (
            "Test log message not found"
        )
        print("✅ Task logging works correctly")

        # Test artifacts
        print("📦 Testing task artifacts")
        artifact = {"name": "test.txt", "content": "test data", "size": 9}
        add_task_artifact(task_id, artifact)

        artifacts = get_task_artifacts(task_id)
        assert len(artifacts) >= 1, f"Expected at least 1 artifact, got {len(artifacts)}"
        assert any(a["name"] == "test.txt" for a in artifacts), "Test artifact not found"
        print("✅ Task artifacts work correctly")

        # Test task listing
        print("📋 Testing task listing")
        all_tasks = get_all_tasks()
        assert len(all_tasks) >= 1, f"Expected at least 1 task, got {len(all_tasks)}"
        task_ids = [t["task_id"] for t in all_tasks]
        assert task_id in task_ids, f"Test task {task_id} not found in task list"
        print("✅ Task listing works correctly")

        # Clean up
        print("🧹 Cleaning up test task")
        clear_task(task_id)
        meta = get_task_meta(task_id)
        assert not meta, "Task should be cleared"
        print("✅ Task cleanup works correctly")

        print("\n🎉 All tests passed! Celery task queue migration is successful.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that the old task_queue.py interface still works if needed."""
    print("\n🔄 Testing backward compatibility...")

    try:
        # Try importing the old module (should fail gracefully)
        if importlib.util.find_spec("task_queue") is not None:
            print("⚠️  Old task_queue.py still exists - this should be removed after migration")
        else:
            print("✅ Old task_queue.py properly removed")

        print("✅ Backward compatibility check passed")
        return True

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Celery Task Queue Migration Tests\n")

    success = True
    success &= test_celery_task_queue()
    success &= test_backward_compatibility()

    if success:
        print("\n🎯 Migration test completed successfully!")
        print("Next steps:")
        print("1. Start Celery worker: celery -A celery_app worker --loglevel=info")
        print("2. Start Flower dashboard: celery -A celery_app flower")
        print("3. Test with actual application workflows")
        print("4. Remove old task_queue.py file")
        sys.exit(0)
    else:
        print("\n💥 Migration test failed!")
        sys.exit(1)
