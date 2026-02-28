from __future__ import annotations

import os
import httpx
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
    """Research the essay topic using Tavily API."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "No research available - Tavily API key not configured."

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": prompt,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": False,
                    "max_results": 5,
                },
            )
            response.raise_for_status()
            data = response.json()

            research_info = []
            if "answer" in data and data["answer"]:
                research_info.append(f"Summary: {data['answer']}")

            if "results" in data:
                for result in data["results"][:3]:
                    title = result.get("title", "")
                    content = result.get("content", "")
                    if content:
                        research_info.append(f"{title}: {content[:300]}...")

            return "\n\n".join(research_info) if research_info else "No relevant research found."

    except Exception as e:  # pragma: no cover - external dependency
        return f"Research failed: {str(e)}"


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
