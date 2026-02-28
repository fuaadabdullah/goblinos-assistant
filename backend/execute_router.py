from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uuid
import asyncio
from sqlalchemy.orm import Session
from .database import get_db
from .models import Task
from .services.goblin_executor import get_goblin_executor
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/execute", tags=["execute"])


class ExecuteRequest(BaseModel):
    goblin: str
    task: str
    code: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    dry_run: Optional[bool] = False


class ExecuteResponse(BaseModel):
    taskId: str
    status: str = "queued"
    message: Optional[str] = None


@router.post("", response_model=ExecuteResponse)
@router.post("/", response_model=ExecuteResponse, include_in_schema=False)
async def execute_task(
    request: ExecuteRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Execute a task using the specified goblin with real GoblinOS integration"""
    try:
        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Validate goblin exists
        executor = get_goblin_executor()
        validation = await executor.validate_goblin(request.goblin)

        if not validation.get("valid"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid goblin: {validation.get('error', 'Unknown error')}",
            )

        # Create task in database
        task = Task(
            id=task_id,
            goblin=request.goblin,
            task=request.task,
            code=request.code,
            provider=request.provider,
            model=request.model,
            status="queued",
        )

        db.add(task)
        db.commit()

        # Execute task in background
        background_tasks.add_task(
            execute_goblin_task,
            task_id=task_id,
            goblin_id=request.goblin,
            task_description=request.task,
            code=request.code,
            dry_run=request.dry_run,
        )

        message = "Task queued for execution"
        if request.dry_run:
            message = "Task queued for dry-run (no changes will be made)"

        return ExecuteResponse(taskId=task_id, status="queued", message=message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute task: {str(e)}")


async def execute_goblin_task(
    task_id: str,
    goblin_id: str,
    task_description: str,
    code: Optional[str] = None,
    dry_run: bool = False,
):
    """Execute a goblin task with real GoblinOS integration"""
    from .database import SessionLocal

    db = SessionLocal()

    try:
        # Update status to running
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            logger.error(f"Task {task_id} not found in database")
            return

        task.status = "running"
        db.commit()

        # Get executor
        executor = get_goblin_executor()

        # Execute the goblin
        logger.info(f"Executing goblin '{goblin_id}' for task {task_id}")

        if code:
            # Execute custom code
            result = await executor.execute_custom_script(code)
        else:
            # Execute goblin command
            result = await executor.execute_goblin(
                goblin_id=goblin_id,
                task_description=task_description,
                code=code,
                dry_run=dry_run,
            )

        # Update task with results
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            if result.get("success"):
                task.status = "completed"
                task.result = result.get("stdout", "")
                if result.get("stderr"):
                    task.result += f"\n\nStderr:\n{result['stderr']}"
            else:
                task.status = "failed"
                task.result = f"Error: {result.get('error', 'Unknown error')}"
                if result.get("stderr"):
                    task.result += f"\n\nStderr:\n{result['stderr']}"

            db.commit()
            logger.info(f"Task {task_id} completed with status: {task.status}")

    except Exception as e:
        logger.error(f"Error executing task {task_id}: {str(e)}", exc_info=True)

        # Update task as failed
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "failed"
            task.result = f"Execution error: {str(e)}"
            db.commit()

    finally:
        db.close()


@router.get("/status/{task_id}")
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """Get the status of a task"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "taskId": task.id,
        "status": task.status,
        "result": task.result,
        "goblin": task.goblin,
        "task": task.task,
    }
