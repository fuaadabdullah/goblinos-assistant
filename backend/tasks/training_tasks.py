"""
Training Tasks
Celery tasks for distributed training on RunPod/Vast.ai
"""

import asyncio
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
    name="src.tasks.training_tasks.submit_training_job",
    max_retries=2,
    default_retry_delay=60,
    queue="training",
    time_limit=86400,  # 24 hours
)
def submit_training_job(
    self,
    config: dict[str, Any],
    checkpoint_path: Optional[str] = None,
    prefer_provider: Optional[str] = None,
) -> dict[str, Any]:
    """
    Submit a full training job.
    
    Config should include:
    - model_name: Base model to train
    - dataset: Dataset path/identifier
    - hyperparameters: Training hyperparameters
    - gpu_type: Required GPU type
    - gpu_count: Number of GPUs
    """
    logger.info(
        "Submitting training job",
        task_id=self.request.id,
        model=config.get("model_name"),
        gpus=config.get("gpu_count", 1),
    )
    
    async def _run():
        router = ProviderRouter()
        
        # Determine provider
        preferred = None
        if prefer_provider:
            preferred = ProviderType(prefer_provider)
        
        # Training goes to RunPod (reliable) or Vast.ai (cost-sensitive)
        decision = await router.route(
            job_type=JobType.TRAINING,
            required_gpu=config.get("gpu_type", "RTX_4090"),
            prefer_provider=preferred,
        )
        
        provider = router.providers[decision.provider]
        
        # Prepare training config
        training_config = {
            "name": f"goblin-train-{self.request.id[:8]}",
            "gpu_type": config.get("gpu_type", "RTX_4090"),
            "gpu_count": config.get("gpu_count", 1),
            "image": config.get("image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"),
            "disk_gb": config.get("disk_gb", 100),
            "env": {
                "MODEL_NAME": config.get("model_name", ""),
                "DATASET": config.get("dataset", ""),
                "WANDB_PROJECT": config.get("wandb_project", "goblin-training"),
                "GCS_CHECKPOINT_BUCKET": settings.gcs_checkpoints_bucket,
                **config.get("env", {}),
            },
            "script": config.get("training_script", ""),
        }
        
        result = await provider.submit_training(
            config=training_config,
            checkpoint_path=checkpoint_path,
        )
        
        return {
            "success": result.success,
            "job_id": result.job_id,
            "provider": result.provider.value,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
        }
    
    try:
        return run_async(_run())
    except Exception as e:
        logger.error("Training job failed", error=str(e), task_id=self.request.id)
        raise self.retry(exc=e)


@shared_task(
    bind=True,
    name="src.tasks.training_tasks.submit_finetuning_job",
    max_retries=2,
    default_retry_delay=30,
    queue="training",
    time_limit=43200,  # 12 hours
)
def submit_finetuning_job(
    self,
    model_name: str,
    dataset_path: str,
    output_path: str,
    lora_config: Optional[dict[str, Any]] = None,
    training_args: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Submit a LoRA/QLoRA fine-tuning job.
    
    Optimized for RTX 4090 spot instances on Vast.ai.
    """
    logger.info(
        "Submitting fine-tuning job",
        task_id=self.request.id,
        model=model_name,
    )
    
    # Default LoRA config for 4090 (24GB VRAM)
    default_lora = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    
    # Default training args
    default_training = {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "fp16": True,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
    }
    
    lora = {**default_lora, **(lora_config or {})}
    training = {**default_training, **(training_args or {})}
    
    # Build fine-tuning script
    finetuning_script = f"""
#!/bin/bash
set -e

# Install dependencies
pip install -q transformers peft bitsandbytes accelerate datasets wandb

# Pull model weights
python /scripts/pull_weights.py --model {model_name} --output /models/

# Run fine-tuning
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    '/models/{model_name}',
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map='auto',
)
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(**{lora})
model = get_peft_model(model, lora_config)

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained('/models/{model_name}')
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset('json', data_files='{dataset_path}')

# Training arguments
training_args = TrainingArguments(
    output_dir='{output_path}',
    **{training}
)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=2048,
)
trainer.train(resume_from_checkpoint={repr(checkpoint_path)})
trainer.save_model('{output_path}/final')
"

# Upload to GCS
gsutil -m cp -r {output_path}/* gs://{settings.gcs_checkpoints_bucket}/finetune/{self.request.id}/
"""

    config = {
        "name": f"goblin-finetune-{self.request.id[:8]}",
        "gpu_type": "RTX_4090",
        "gpu_count": 1,
        "image": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
        "disk_gb": 50,
        "script": finetuning_script,
        "env": {
            "WANDB_PROJECT": "goblin-finetuning",
            "HF_HOME": "/models/cache",
        },
    }
    
    async def _run():
        router = ProviderRouter()
        
        # Prefer Vast.ai for cost-sensitive fine-tuning
        decision = await router.route(
            job_type=JobType.FINE_TUNING,
            required_gpu="RTX_4090",
            prefer_provider=ProviderType.VASTAI,
        )
        
        provider = router.providers[decision.provider]
        result = await provider.submit_training(
            config=config,
            checkpoint_path=checkpoint_path,
        )
        
        return {
            "success": result.success,
            "job_id": result.job_id,
            "provider": result.provider.value,
            "output": result.output,
            "error": result.error,
            "config": {
                "model": model_name,
                "lora": lora,
                "training_args": training,
            },
        }
    
    try:
        return run_async(_run())
    except Exception as e:
        logger.error("Fine-tuning job failed", error=str(e), task_id=self.request.id)
        raise self.retry(exc=e)


@shared_task(
    bind=True,
    name="src.tasks.training_tasks.check_training_status",
)
def check_training_status(
    self,
    job_id: str,
    provider: str,
) -> dict[str, Any]:
    """Check status of a training job."""
    
    async def _run():
        router = ProviderRouter()
        provider_type = ProviderType(provider)
        
        if provider_type not in router.providers:
            return {"error": f"Provider {provider} not available"}
        
        status = await router.providers[provider_type].get_job_status(job_id)
        return status
    
    return run_async(_run())
