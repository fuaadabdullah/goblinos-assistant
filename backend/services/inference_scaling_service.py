"""
Inference-Time Scaling Service
Generates multiple candidate reasoning chains and uses PRM to select the best path.

Features:
- Multiple path attempts for complex queries
- PRM-based chain scoring and selection
- Reduces hallucinations on hard questions
- Configurable number of candidates and scoring thresholds
"""

import asyncio
import random
from typing import List, Dict, Any, Optional
import logging

from .prm_service import PRMService, ReasoningChain
from ..providers import OllamaAdapter

logger = logging.getLogger(__name__)


class InferenceScalingService:
    """Service for inference-time scaling with multiple candidate generation and PRM scoring."""

    def __init__(self, prm_service: Optional[PRMService] = None):
        """Initialize inference scaling service."""
        self.prm_service = prm_service or PRMService()

        # Configuration
        self.default_candidates = 3  # Number of candidate chains to generate
        self.min_score_threshold = 0.6  # Minimum score to accept a chain
        self.max_attempts = 5  # Maximum attempts to find good chain

        # Prompt variations for diversity
        self.prompt_templates = [
            "Solve this step by step: {query}",
            "Think through this carefully: {query}",
            "Break this down systematically: {query}",
            "Analyze this problem: {query}",
            "Work through this methodically: {query}",
        ]

    async def scale_inference(
        self,
        query: str,
        adapter: OllamaAdapter,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        num_candidates: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform inference-time scaling: generate multiple candidates and select best.

        Args:
            query: The user query
            adapter: LLM adapter to use
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            num_candidates: Number of candidates to generate (default: self.default_candidates)

        Returns:
            Dict with best chain, all candidates, and scoring info
        """
        num_candidates = num_candidates or self.default_candidates

        logger.info(
            f"Starting inference scaling for query: {query[:50]}... with {num_candidates} candidates"
        )

        # Generate candidate chains
        candidates = await self._generate_candidates(
            query, adapter, model, temperature, max_tokens, num_candidates
        )

        if not candidates:
            logger.warning("Failed to generate any candidate chains")
            return {
                "success": False,
                "error": "Failed to generate candidate chains",
                "candidates": [],
                "best_chain": None,
            }

        # Score candidates using PRM
        scored_chains = await self.prm_service.score_candidates(candidates, query)

        # Select best chain
        best_chain = self.prm_service.select_best_chain(scored_chains)

        # Check if best chain meets minimum threshold
        if best_chain.total_score < self.min_score_threshold:
            logger.warning(
                f"Best chain score {best_chain.total_score:.2f} below threshold {self.min_score_threshold}"
            )

            # Try one more attempt with different prompt
            fallback_candidates = await self._generate_candidates(
                query, adapter, model, temperature * 1.2, max_tokens, 2
            )

            if fallback_candidates:
                fallback_scored = await self.prm_service.score_candidates(
                    fallback_candidates, query
                )
                fallback_best = self.prm_service.select_best_chain(fallback_scored)

                if fallback_best.total_score > best_chain.total_score:
                    best_chain = fallback_best
                    scored_chains.extend(fallback_scored)

        # Detailed evaluation of best chain
        evaluation = await self.prm_service.evaluate_chain_quality(best_chain)

        result = {
            "success": True,
            "best_chain": {
                "response": self._format_chain_response(best_chain),
                "score": best_chain.total_score,
                "evaluation": evaluation,
            },
            "candidates_count": len(scored_chains),
            "all_scores": [chain.total_score for chain in scored_chains],
            "scaling_info": {
                "candidates_generated": len(candidates),
                "chains_scored": len(scored_chains),
                "best_score": best_chain.total_score,
                "score_distribution": self._analyze_score_distribution(scored_chains),
            },
        }

        logger.info(
            f"Inference scaling complete. Best score: {best_chain.total_score:.2f}"
        )
        return result

    async def _generate_candidates(
        self,
        query: str,
        adapter: OllamaAdapter,
        model: str,
        temperature: float,
        max_tokens: int,
        num_candidates: int,
    ) -> List[str]:
        """Generate multiple candidate reasoning chains."""
        candidates = []
        attempt = 0

        while len(candidates) < num_candidates and attempt < self.max_attempts:
            attempt += 1

            # Generate diverse prompts
            prompts = self._generate_diverse_prompts(
                query, min(num_candidates - len(candidates), 3)
            )

            # Generate responses concurrently
            tasks = []
            for prompt in prompts:
                task = adapter.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                    + random.uniform(-0.1, 0.1),  # Add some variation
                    max_tokens=max_tokens,
                )
                tasks.append(task)

            # Wait for all responses
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                for response in responses:
                    if isinstance(response, Exception):
                        logger.warning(f"Chain generation failed: {response}")
                        continue

                    if response and len(response.strip()) > 50:  # Minimum length check
                        candidates.append(response)

            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                continue

        logger.info(
            f"Generated {len(candidates)} candidate chains in {attempt} attempts"
        )
        return candidates

    def _generate_diverse_prompts(self, query: str, count: int) -> List[str]:
        """Generate diverse prompts to encourage different reasoning paths."""
        prompts = []

        # Use different templates
        templates_used = 0
        for i in range(count):
            template = self.prompt_templates[
                templates_used % len(self.prompt_templates)
            ]
            prompts.append(template.format(query=query))
            templates_used += 1

            # Add reasoning instructions for diversity
            if i > 0:
                reasoning_instructions = [
                    " Focus on step-by-step logical reasoning.",
                    " Emphasize careful analysis of the problem.",
                    " Consider multiple perspectives.",
                    " Pay special attention to edge cases.",
                ]
                instruction = reasoning_instructions[i % len(reasoning_instructions)]
                prompts[-1] += instruction

        return prompts

    def _format_chain_response(self, chain: ReasoningChain) -> str:
        """Format a reasoning chain for final response."""
        if len(chain.steps) <= 1:
            # Simple response, return as-is
            return chain.final_answer

        # Format multi-step response
        formatted_steps = []
        for step in chain.steps:
            formatted_steps.append(f"{step.step_number}. {step.content}")

        steps_text = "\n".join(formatted_steps)

        # Add final answer if different from last step
        if chain.final_answer and chain.final_answer != chain.steps[-1].content:
            return f"{steps_text}\n\nFinal Answer: {chain.final_answer}"
        else:
            return steps_text

    def _analyze_score_distribution(
        self, chains: List[ReasoningChain]
    ) -> Dict[str, Any]:
        """Analyze the distribution of scores across candidates."""
        if not chains:
            return {}

        scores = [chain.total_score for chain in chains]
        scores.sort(reverse=True)

        return {
            "best_score": scores[0],
            "worst_score": scores[-1],
            "average_score": sum(scores) / len(scores),
            "score_variance": sum((s - sum(scores) / len(scores)) ** 2 for s in scores)
            / len(scores),
            "top_3_scores": scores[:3],
        }

    def should_use_scaling(self, query: str, intent: str, context_length: int) -> bool:
        """Determine if inference-time scaling should be used for this query."""
        # Use scaling for complex queries
        if intent == "complex":
            return True

        # Use scaling for long queries that might benefit from multiple attempts
        if len(query) > 200:
            return True

        # Use scaling for queries with complex keywords
        complex_keywords = [
            "analyze",
            "compare",
            "evaluate",
            "explain",
            "solve",
            "calculate",
            "determine",
            "prove",
            "design",
            "optimize",
        ]

        query_lower = query.lower()
        if any(keyword in query_lower for keyword in complex_keywords):
            return True

        # Use scaling for queries with multiple parts (questions with multiple ?)
        if query.count("?") > 1:
            return True

        return False

    async def get_scaling_config(self, query_complexity: str) -> Dict[str, Any]:
        """Get scaling configuration based on query complexity."""
        configs = {
            "low": {
                "num_candidates": 2,
                "temperature": 0.3,
                "min_score_threshold": 0.5,
            },
            "medium": {
                "num_candidates": 3,
                "temperature": 0.5,
                "min_score_threshold": 0.6,
            },
            "high": {
                "num_candidates": 5,
                "temperature": 0.7,
                "min_score_threshold": 0.7,
            },
        }

        return configs.get(query_complexity, configs["medium"])
