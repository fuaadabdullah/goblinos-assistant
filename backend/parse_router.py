from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/parse", tags=["parse"])


class ParseRequest(BaseModel):
    text: str
    default_goblin: Optional[str] = None


class OrchestrationStep(BaseModel):
    goblin: str
    task: str
    dependencies: List[str] = []


class OrchestrationPlan(BaseModel):
    steps: List[OrchestrationStep]
    estimated_duration: int = 0
    complexity: str = "medium"


@router.post("", response_model=OrchestrationPlan)
@router.post("/", response_model=OrchestrationPlan, include_in_schema=False)
async def parse_orchestration(request: ParseRequest):
    """Parse natural language text into an orchestration plan"""
    try:
        text = request.text.lower()

        # Simple keyword-based parsing (in production, use NLP/AI)
        steps = []

        # Detect common patterns
        if "search" in text or "find" in text or "query" in text:
            steps.append(
                OrchestrationStep(
                    goblin="search-goblin",
                    task="Search for information",
                    dependencies=[],
                )
            )

        if "analyze" in text or "review" in text or "examine" in text:
            steps.append(
                OrchestrationStep(
                    goblin="analyze-goblin",
                    task="Analyze the results",
                    dependencies=["search-goblin"] if steps else [],
                )
            )

        if "create" in text or "build" in text or "generate" in text:
            steps.append(
                OrchestrationStep(
                    goblin="create-goblin",
                    task="Create or generate content",
                    dependencies=["analyze-goblin"] if len(steps) > 1 else [],
                )
            )

        # If no specific patterns detected, use default goblin
        if not steps:
            default_goblin = request.default_goblin or "general-goblin"
            steps.append(
                OrchestrationStep(
                    goblin=default_goblin,
                    task=request.text[:100] + "..."
                    if len(request.text) > 100
                    else request.text,
                    dependencies=[],
                )
            )

        # Estimate complexity and duration
        complexity = (
            "low" if len(steps) <= 1 else "medium" if len(steps) <= 3 else "high"
        )
        duration = len(steps) * 30  # 30 seconds per step estimate

        return OrchestrationPlan(
            steps=steps, estimated_duration=duration, complexity=complexity
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse orchestration: {str(e)}"
        )
