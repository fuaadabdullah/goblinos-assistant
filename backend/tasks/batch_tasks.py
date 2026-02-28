"""
Batch Tasks
Celery tasks for batch processing and maintenance
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

from celery import shared_task
import structlog

from ..orchestrator import ProviderRouter
from ..orchestrator.providers import JobType, ProviderType
from ..config import settings

logger = structlog.get_logger()


def run_async(coro):
    """Helper to run async code in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@shared_task(
    bind=True,
    name="src.tasks.batch_tasks.batch_inference",
    queue="batch",
    time_limit=7200,  # 2 hours
)
def batch_inference(
    self,
    prompts: list[str],
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 10,
) -> dict[str, Any]:
    """
    Batch inference for multiple prompts.
    
    Uses Vast.ai for cost optimization.
    """
    logger.info(
        "Starting batch inference",
        task_id=self.request.id,
        num_prompts=len(prompts),
    )
    
    async def _run():
        router = ProviderRouter()
        
        # Use Vast.ai for cost-sensitive batch
        decision = await router.route(
            job_type=JobType.BATCH_INFERENCE,
            prefer_provider=ProviderType.VASTAI,
        )
        
        provider = router.providers[decision.provider]
        results = []
        total_cost = 0.0
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                result = await provider.submit_inference(
                    model=model or "llama3:8b",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                batch_results.append({
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                })
                total_cost += result.cost
            
            results.extend(batch_results)
            
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": len(results),
                    "total": len(prompts),
                    "cost": total_cost,
                }
            )
        
        router.record_cost(decision.provider, total_cost)
        
        return {
            "success": True,
            "results": results,
            "total_prompts": len(prompts),
            "successful": sum(1 for r in results if r["success"]),
            "total_cost": total_cost,
            "provider": decision.provider.value,
        }
    
    try:
        return run_async(_run())
    except Exception as e:
        logger.error("Batch inference failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@shared_task(
    bind=True,
    name="src.tasks.batch_tasks.hyperparameter_sweep",
    queue="batch",
    time_limit=86400,  # 24 hours
)
def hyperparameter_sweep(
    self,
    base_config: dict[str, Any],
    sweep_params: dict[str, list[Any]],
    num_trials: int = 10,
    metric: str = "eval_loss",
) -> dict[str, Any]:
    """
    Hyperparameter sweep using Vast.ai spot instances.
    
    Uses cheap spots for parallel exploration.
    """
    import itertools
    
    logger.info(
        "Starting hyperparameter sweep",
        task_id=self.request.id,
        num_params=len(sweep_params),
        num_trials=num_trials,
    )
    
    # Generate parameter combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    all_combinations = list(itertools.product(*param_values))
    
    # Select subset for trials
    if len(all_combinations) > num_trials:
        import random
        combinations = random.sample(all_combinations, num_trials)
    else:
        combinations = all_combinations
    
    async def _run():
        router = ProviderRouter()
        results = []
        
        for i, combo in enumerate(combinations):
            # Build config for this trial
            trial_config = {**base_config}
            for name, value in zip(param_names, combo):
                trial_config[name] = value
            
            trial_config["name"] = f"sweep-{self.request.id[:8]}-trial-{i}"
            
            # Submit training job
            decision = await router.route(
                job_type=JobType.SWEEP,
                prefer_provider=ProviderType.VASTAI,
            )
            
            provider = router.providers[decision.provider]
            result = await provider.submit_training(config=trial_config)
            
            results.append({
                "trial": i,
                "params": dict(zip(param_names, combo)),
                "job_id": result.job_id,
                "provider": result.provider.value,
                "success": result.success,
            })
            
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": i + 1,
                    "total": len(combinations),
                    "completed_trials": results,
                }
            )
        
        return {
            "success": True,
            "num_trials": len(combinations),
            "results": results,
            "sweep_params": sweep_params,
        }
    
    try:
        return run_async(_run())
    except Exception as e:
        logger.error("Hyperparameter sweep failed", error=str(e))
        return {"success": False, "error": str(e)}


@shared_task(name="src.tasks.batch_tasks.provider_health_check")
def provider_health_check() -> dict[str, Any]:
    """
    Periodic health check for all providers.
    Runs every 5 minutes via Celery Beat.
    """
    async def _run():
        router = ProviderRouter()
        health = await router.get_all_health()
        
        logger.info("Provider health check", health=health)
        
        # Alert if any provider is down
        unhealthy = [p for p, h in health.items() if not h]
        if unhealthy:
            logger.warning("Unhealthy providers detected", providers=unhealthy)
            # TODO: Send alert to webhook
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": health,
            "unhealthy": unhealthy,
        }
    
    return run_async(_run())


@shared_task(name="src.tasks.batch_tasks.daily_cost_report")
def daily_cost_report() -> dict[str, Any]:
    """
    Generate daily cost report.
    Runs daily via Celery Beat.
    """
    logger.info("Generating daily cost report")
    
    # TODO: Aggregate costs from router._cost_tracker
    # TODO: Query provider APIs for actual spend
    
    return {
        "date": datetime.utcnow().date().isoformat(),
        "total_cost": 0.0,  # Placeholder
        "by_provider": {},
        "budget_remaining": settings.daily_budget_limit_usd,
    }


@shared_task(name="src.tasks.batch_tasks.cleanup_old_checkpoints")
def cleanup_old_checkpoints(
    max_age_days: int = 30,
) -> dict[str, Any]:
    """
    Clean up old checkpoints from GCS.
    Runs weekly via Celery Beat.
    """
    from google.cloud import storage
    
    logger.info("Cleaning up old checkpoints", max_age_days=max_age_days)
    
    try:
        client = storage.Client()
        bucket = client.bucket(settings.gcs_checkpoints_bucket)
        
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        deleted = 0
        
        for blob in bucket.list_blobs():
            if blob.updated < cutoff:
                blob.delete()
                deleted += 1
        
        logger.info("Checkpoint cleanup complete", deleted=deleted)
        
        return {
            "success": True,
            "deleted": deleted,
            "cutoff_date": cutoff.isoformat(),
        }
        
    except Exception as e:
        logger.error("Checkpoint cleanup failed", error=str(e))
        return {"success": False, "error": str(e)}


@shared_task(
    bind=True,
    name="src.tasks.batch_tasks.upload_to_vector_store",
    queue="batch",
)
def upload_to_vector_store(
    self,
    documents: list[dict[str, Any]],
    collection_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Batch upload documents to Qdrant vector store.
    
    Documents should have:
    - content: str
    - metadata: dict (optional)
    """
    from ..rag import QdrantVectorStore, EmbeddingService, DocumentChunker
    from ..rag.vector_store import Document
    
    logger.info(
        "Uploading to vector store",
        task_id=self.request.id,
        num_documents=len(documents),
    )
    
    async def _run():
        # Initialize services
        vector_store = QdrantVectorStore(collection_name=collection_name)
        embeddings = EmbeddingService()
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        
        all_docs = []
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Chunk document
            chunks = chunker.chunk_text(content, metadata)
            
            # Generate embeddings
            texts = [c.content for c in chunks]
            chunk_embeddings = embeddings.embed_batch(texts)
            
            # Create documents
            for chunk, embedding in zip(chunks, chunk_embeddings):
                all_docs.append(Document.create(
                    content=chunk.content,
                    embedding=embedding,
                    metadata={**chunk.metadata, "source_doc_index": i},
                ))
            
            self.update_state(
                state="PROGRESS",
                meta={"processed": i + 1, "total": len(documents)}
            )
        
        # Upload to vector store
        ids = await vector_store.add_documents(all_docs)
        
        return {
            "success": True,
            "documents_processed": len(documents),
            "chunks_created": len(all_docs),
            "ids": ids[:10],  # Return first 10 IDs
        }
    
    try:
        return run_async(_run())
    except Exception as e:
        logger.error("Upload to vector store failed", error=str(e))
        return {"success": False, "error": str(e)}
