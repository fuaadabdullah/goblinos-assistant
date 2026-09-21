from __future__ import annotations

import os
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/essay", tags=["essay"])


class EssayRequest(BaseModel):
    prompt: str
    length: Optional[int] = 500
    style: Optional[str] = "academic"


class EssayResponse(BaseModel):
    essay: str
    status: str


async def _research_topic(prompt: str) -> str:
    """Research the essay topic using Brave Search."""
    from ..services.brave_search import BraveSearchNotConfigured, web_search

    try:
        results = await web_search(prompt, count=5)
    except BraveSearchNotConfigured:
        return "No research available - BRAVE_API_KEY not configured."
    except Exception as e:  # pragma: no cover - external dependency
        return f"Research failed: {str(e)}"

    if not results:
        return "No relevant research found."

    research_info = []
    for result in results[:3]:
        snippet = result.snippet[:300]
        if snippet:
            research_info.append(f"{result.title}: {snippet}...")
        else:
            research_info.append(result.title)

    return "\n\n".join(research_info)


async def _generate_essay_with_llm(
    prompt: str, length: int, style: str, research_data: str
) -> str:
    """Generate essay using OpenAI directly."""
    try:
        from openai import OpenAI

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return (
                "Essay generation failed: OpenAI API key not configured. "
                f"Research data: {research_data[:500]}..."
            )

        client = OpenAI(api_key=openai_api_key)
        system_prompt = (
            "You are an expert essay writer. Write a {style} essay on the following topic. "
            "Use the provided research information to support your arguments. "
            "Aim for approximately {length} words. "
            "Structure the essay with an introduction, body paragraphs, and conclusion. "
            "Use proper {style} language and formatting."
        ).format(style=style, length=length)

        user_prompt = f"""Topic: {prompt}

Research Information:
{research_data}

Please write a complete {style} essay on this topic using the research provided."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=length * 4,
            temperature=0.7,
        )

        essay_content = response.choices[0].message.content.strip()
        word_count = len(essay_content.split())
        return f"**{style.title()} Essay** ({word_count} words)\n\n{essay_content}"

    except Exception as e:  # pragma: no cover - external dependency
        return f"Essay generation failed: {str(e)}. Research data: {research_data[:500]}..."


@router.post("/", response_model=EssayResponse)
async def generate_essay(request: Request, essay_req: EssayRequest):
    """Generate an essay based on the provided prompt."""
    try:
        prompt = essay_req.prompt
        length = essay_req.length or 500
        style = essay_req.style or "academic"

        research_data = await _research_topic(prompt)
        essay = await _generate_essay_with_llm(prompt, length, style, research_data)

        return EssayResponse(essay=essay, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Essay generation failed: {str(e)}")


__all__ = ["router", "EssayRequest", "EssayResponse"]
