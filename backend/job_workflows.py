"""
Advanced Job Features for Goblin Assistant

Provides job dependencies, chains, groups, chords, and workflow orchestration.
"""

from celery import chain, group, chord, signature
from .celery_app import app
import logging

logger = logging.getLogger(__name__)


class JobWorkflow:
    """Manages complex job workflows with dependencies."""

    @staticmethod
    def create_provider_health_workflow():
        """Create a workflow that probes all providers and aggregates results."""
        from tasks.provider_probe_worker import probe_provider, aggregate_health_results

        # Create a group of parallel provider probes
        provider_tasks = [
            probe_provider.s("openai"),
            probe_provider.s("anthropic"),
            probe_provider.s("groq"),
            probe_provider.s("ollama"),
            probe_provider.s("huggingface"),
        ]

        # Create a chord: run all probes in parallel, then aggregate results
        workflow = chord(provider_tasks, aggregate_health_results.s())

        return workflow

    @staticmethod
    def create_model_training_workflow(model_config: dict):
        """Create a workflow for model training with dependencies."""
        from tasks.model_training_worker import (
            prepare_training_data,
            train_model,
            validate_model,
            deploy_model,
            cleanup_training_artifacts,
        )

        # Chain: prepare data -> train -> validate -> deploy -> cleanup
        workflow = chain(
            prepare_training_data.s(model_config),
            train_model.s(),
            validate_model.s(),
            deploy_model.s(),
            cleanup_training_artifacts.s(),
        )

        return workflow

    @staticmethod
    def create_data_processing_pipeline(data_source: str, processing_config: dict):
        """Create a data processing pipeline with parallel processing."""
        from tasks.data_processing_worker import (
            extract_data,
            validate_data,
            process_chunks,
            merge_results,
            store_processed_data,
        )

        # Extract and validate first
        extract_task = extract_data.s(data_source)
        validate_task = validate_data.s()

        # Process in parallel chunks
        process_tasks = [
            process_chunks.s(chunk_id=i, config=processing_config)
            for i in range(processing_config.get("num_chunks", 4))
        ]

        # Chain: extract -> validate -> parallel processing -> merge -> store
        workflow = chain(
            extract_task,
            validate_task,
            chord(process_tasks, merge_results.s()),
            store_processed_data.s(),
        )

        return workflow

    @staticmethod
    def create_notification_workflow(event_type: str, recipients: list, context: dict):
        """Create a notification workflow with retry logic."""
        from tasks.notification_worker import (
            prepare_notification,
            send_email_notification,
            send_webhook_notification,
            log_notification_result,
        )

        # Prepare notification content
        prepare_task = prepare_notification.s(event_type, context)

        # Send notifications in parallel (email and webhook)
        notification_tasks = [
            send_email_notification.s(recipients),
            send_webhook_notification.s(context.get("webhook_url")),
        ]

        # Chain: prepare -> send notifications -> log results
        workflow = chain(
            prepare_task, group(notification_tasks), log_notification_result.s()
        )

        return workflow


class JobScheduler:
    """Handles scheduled and delayed job execution."""

    @staticmethod
    def schedule_delayed_task(task_name: str, delay_seconds: int, *args, **kwargs):
        """Schedule a task to run after a delay."""
        task_sig = signature(task_name, args=args, kwargs=kwargs)
        return task_sig.apply_async(countdown=delay_seconds)

    @staticmethod
    def schedule_eta_task(task_name: str, eta_datetime, *args, **kwargs):
        """Schedule a task to run at a specific datetime."""
        task_sig = signature(task_name, args=args, kwargs=kwargs)
        return task_sig.apply_async(eta=eta_datetime)

    @staticmethod
    def create_recurring_task(task_name: str, cron_expression: dict, *args, **kwargs):
        """Create a recurring task with cron-like scheduling."""
        from celery.schedules import crontab

        # Convert cron expression to crontab
        schedule = crontab(**cron_expression)

        # Register with beat schedule
        schedule_name = f"recurring_{task_name}_{hash(str(args) + str(kwargs))}"
        app.conf.beat_schedule[schedule_name] = {
            "task": task_name,
            "schedule": schedule,
            "args": args,
            "kwargs": kwargs,
        }

        return schedule_name


class JobMonitor:
    """Monitors job execution and provides observability."""

    @staticmethod
    def get_job_status(job_id: str):
        """Get the status of a job."""
        result = app.AsyncResult(job_id)
        return {
            "job_id": job_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "error": str(result.info) if result.failed() else None,
            "date_done": result.date_done,
            "runtime": (result.date_done - result.date_queued).total_seconds()
            if result.date_done and result.date_queued
            else None,
        }

    @staticmethod
    def get_active_jobs():
        """Get all currently active jobs."""
        inspect = app.control.inspect()

        active_tasks = []
        for worker, tasks in (inspect.active() or {}).items():
            for task in tasks:
                active_tasks.append(
                    {
                        "worker": worker,
                        "task_id": task.get("id"),
                        "task_name": task.get("name"),
                        "args": task.get("args"),
                        "kwargs": task.get("kwargs"),
                        "time_start": task.get("time_start"),
                    }
                )

        return active_tasks

    @staticmethod
    def cancel_job(job_id: str):
        """Cancel a running job."""
        result = app.AsyncResult(job_id)
        result.revoke(terminate=True)
        return True

    @staticmethod
    def get_job_stats():
        """Get job queue statistics."""
        inspect = app.control.inspect()

        stats = {
            "active": inspect.active(),
            "scheduled": inspect.scheduled(),
            "reserved": inspect.reserved(),
            "registered": inspect.registered(),
            "stats": inspect.stats(),
        }

        return stats


# Convenience functions for common workflows
def run_provider_health_check():
    """Run the complete provider health check workflow."""
    workflow = JobWorkflow.create_provider_health_workflow()
    result = workflow.apply_async()
    logger.info(f"Started provider health check workflow: {result.id}")
    return result.id


def run_model_training(model_config: dict):
    """Run the complete model training workflow."""
    workflow = JobWorkflow.create_model_training_workflow(model_config)
    result = workflow.apply_async()
    logger.info(f"Started model training workflow: {result.id}")
    return result.id


def run_data_processing(data_source: str, config: dict):
    """Run the complete data processing pipeline."""
    workflow = JobWorkflow.create_data_processing_pipeline(data_source, config)
    result = workflow.apply_async()
    logger.info(f"Started data processing pipeline: {result.id}")
    return result.id


def send_notifications(event_type: str, recipients: list, context: dict):
    """Send notifications through the complete workflow."""
    workflow = JobWorkflow.create_notification_workflow(event_type, recipients, context)
    result = workflow.apply_async()
    logger.info(f"Started notification workflow: {result.id}")
    return result.id
