"""
Inference Tasks
Celery tasks for AI inference across providers
"""

import asyncio
from typing import Any, Optional

from celery import shared_task
import structlog

from ..orchestrator import ProviderRouter, RoutingDecision
from ..orchestrator.providers import JobType, ProviderType
from ..rag import RAGRetriever

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
    name="src.tasks.inference_tasks.inference_default",
    max_retries=3,
    default_retry_delay=5,
)
def inference_default(
    self,
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs,
) -> dict[str, Any]:
    """
    Default inference task using GCP/Ollama.
    
    Use for development and low-priority requests.
    """
    logger.info(
        "Starting default inference",
        task_id=self.request.id,
        model=model,
    )
    
    async def _run():
        router = ProviderRouter()
        
        # Route to GCP for default
        decision = await router.route(
            job_type=JobType.INFERENCE,
            prefer_provider=ProviderType.GCP,
        )
        
        provider = router.providers[decision.provider]
        result = await provider.submit_inference(
            model=model or "phi3:mini",  # Fallback model
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        # Record cost
        router.record_cost(decision.provider, result.cost)
        
        return {
            "success": result.success,
            "output": result.output,
            "provider": result.provider.value,
            "job_id": result.job_id,
            "cost": result.cost,
            "duration": result.duration_seconds,
            "error": result.error,
        }
    
    try:
        return run_async(_run())
    except Exception as e:
        logger.error("Inference failed", error=str(e), task_id=self.request.id)
        raise self.retry(exc=e)


@shared_task(
    bind=True,
    name="src.tasks.inference_tasks.inference_high_priority",
    max_retries=3,
    default_retry_delay=2,
    queue="high_priority",
)
def inference_high_priority(
    self,
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs,
) -> dict[str, Any]:
    """
    High-priority inference using RunPod serverless.
    
    Use for production, low-latency requirements.
    """
    logger.info(
        "Starting high-priority inference",
        task_id=self.request.id,
        model=model,
    )
    
    async def _run():
        router = ProviderRouter()
        
        # Prefer RunPod for production
        decision = await router.route(
            job_type=JobType.INFERENCE,
            prefer_provider=ProviderType.RUNPOD,
            max_latency_ms=1000,  # <1s requirement
        )
        
        provider = router.providers[decision.provider]
        result = await provider.submit_inference(
            model=model or "llama3:70b",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        # Fallback to GCP if RunPod fails
        if not result.success and decision.fallback_provider:
            logger.warning(
                "Primary provider failed, using fallback",
                primary=decision.provider.value,
                fallback=decision.fallback_provider.value,
            )
            fallback = router.providers[decision.fallback_provider]
            result = await fallback.submit_inference(
                model="phi3:mini",  # Use smaller model for fallback
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        router.record_cost(result.provider, result.cost)
        
        return {
            "success": result.success,
            "output": result.output,
            "provider": result.provider.value,
            "job_id": result.job_id,
            "cost": result.cost,
            "duration": result.duration_seconds,
            "error": result.error,
        }
    
    try:
        return run_async(_run())
    except Exception as e:
        logger.error(
            "High-priority inference failed",
            error=str(e),
            task_id=self.request.id,
        )
        raise self.retry(exc=e)


@shared_task(
    bind=True,
    name="src.tasks.inference_tasks.inference_with_rag",
    max_retries=2,
    default_retry_delay=5,
)
def inference_with_rag(
    self,
    query: str,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_k: int = 5,
    filters: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    RAG-enhanced inference task.
    
    1. Retrieves relevant context from vector store
    2. Constructs augmented prompt
    3. Runs inference with context
    """
    logger.info(
        "Starting RAG inference",
        task_id=self.request.id,
        query_length=len(query),
    )
    
    async def _run():
        # Initialize RAG retriever
        retriever = RAGRetriever(top_k=top_k)
        
        # Retrieve relevant context
        retrieval_result = await retriever.retrieve(
            query=query,
            filters=filters,
        )
        
        # Build augmented prompt
        augmented_prompt = retriever.build_rag_prompt(
            query=query,
            context=retrieval_result.formatted_context,
        )
        
        # Run inference
        router = ProviderRouter()
        decision = await router.route(job_type=JobType.INFERENCE)
        
        provider = router.providers[decision.provider]
        result = await provider.submit_inference(
            model=model or "llama3:70b",
            prompt=augmented_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        router.record_cost(decision.provider, result.cost)
        
        return {
            "success": result.success,
            "output": result.output,
            "provider": result.provider.value,
            "job_id": result.job_id,
            "cost": result.cost,
            "duration": result.duration_seconds,
            "error": result.error,
            "rag_metadata": {
                "contexts_retrieved": len(retrieval_result.contexts),
                "top_score": retrieval_result.metadata.get("top_score", 0),
            },
        }
    
    try:
        return run_async(_run())
    except Exception as e:
        logger.error("RAG inference failed", error=str(e), task_id=self.request.id)
        raise self.retry(exc=e)
