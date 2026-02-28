"""
Process Reward Model (PRM) Service
Scores reasoning chains to select the best path and reduce hallucinations.

Features:
- Lightweight PRM for step-by-step reasoning evaluation
- Chain scoring with confidence metrics
- Multiple scoring criteria (correctness, coherence, completeness)
- Integration with inference-time scaling
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """Represents a single step in a reasoning chain."""

    step_number: int
    content: str
    reasoning_type: str  # "analysis", "calculation", "inference", "conclusion"
    confidence_score: float = 0.0


@dataclass
class ReasoningChain:
    """Complete reasoning chain with scoring."""

    chain_id: str
    steps: List[ChainStep]
    final_answer: str
    total_score: float = 0.0
    coherence_score: float = 0.0
    correctness_score: float = 0.0
    completeness_score: float = 0.0


class PRMService:
    """Process Reward Model for scoring reasoning chains."""

    def __init__(self):
        """Initialize PRM service with scoring criteria."""
        self.step_keywords = {
            "analysis": ["analyze", "examine", "consider", "evaluate", "assess"],
            "calculation": ["calculate", "compute", "determine", "solve", "find"],
            "inference": ["therefore", "thus", "consequently", "implies", "follows"],
            "conclusion": ["conclude", "final", "answer", "result", "summary"],
        }

        # Scoring weights
        self.weights = {"coherence": 0.3, "correctness": 0.4, "completeness": 0.3}

    def parse_reasoning_chain(
        self, response_text: str, chain_id: str
    ) -> ReasoningChain:
        """Parse a model response into a structured reasoning chain."""
        try:
            # Split response into steps (look for numbered steps or clear delimiters)
            steps = self._extract_steps(response_text)

            if not steps:
                # Fallback: treat entire response as single step
                steps = [
                    ChainStep(
                        step_number=1,
                        content=response_text.strip(),
                        reasoning_type="analysis",
                    )
                ]

            # Extract final answer
            final_answer = self._extract_final_answer(response_text)

            # Create chain
            chain = ReasoningChain(
                chain_id=chain_id, steps=steps, final_answer=final_answer
            )

            # Score the chain
            self._score_chain(chain)

            return chain

        except Exception as e:
            logger.error(f"Failed to parse reasoning chain: {e}")
            # Return minimal chain on error
            return ReasoningChain(
                chain_id=chain_id,
                steps=[ChainStep(1, response_text, "analysis")],
                final_answer=response_text,
            )

    def _extract_steps(self, text: str) -> List[ChainStep]:
        """Extract individual reasoning steps from response text."""
        steps = []

        # Pattern 1: Numbered steps (1., 2., etc.)
        numbered_pattern = r"(\d+)\.\s*([^(\n|(?=\d+\.))]+)"
        matches = re.findall(numbered_pattern, text, re.MULTILINE | re.DOTALL)

        if matches:
            for num, content in matches:
                step_type = self._classify_step(content.strip())
                steps.append(
                    ChainStep(
                        step_number=int(num),
                        content=content.strip(),
                        reasoning_type=step_type,
                    )
                )
        else:
            # Pattern 2: Look for step indicators
            step_indicators = [
                r"Step\s*\d+:?\s*([^(\n|(?=Step))]+)",
                r"First,?([^(\n|(?=(Second|Then|Next|Finally)))]+)",
                r"Second,?([^(\n|(?=(Third|Then|Next|Finally)))]+)",
                r"Third,?([^(\n|(?=(Then|Next|Finally)))]+)",
                r"(Then|Next|After that),?([^(\n|(?=(Finally|So|Therefore)))]+)",
                r"(Finally|So|Therefore),?(.+)",
            ]

            step_num = 1
            for pattern in step_indicators:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    content = match[1] if isinstance(match, tuple) else match
                    if content.strip():
                        step_type = self._classify_step(content.strip())
                        steps.append(
                            ChainStep(
                                step_number=step_num,
                                content=content.strip(),
                                reasoning_type=step_type,
                            )
                        )
                        step_num += 1
                        break  # Only take first match per pattern

        return steps

    def _classify_step(self, content: str) -> str:
        """Classify the type of reasoning step."""
        content_lower = content.lower()

        for step_type, keywords in self.step_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return step_type

        # Default classification based on content patterns
        if any(
            word in content_lower
            for word in ["because", "since", "due to", "as a result"]
        ):
            return "inference"
        elif any(
            word in content_lower for word in ["total", "sum", "equals", "result"]
        ):
            return "calculation"
        elif len(content.split()) < 20:  # Short content often indicates conclusion
            return "conclusion"
        else:
            return "analysis"

    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from the response."""
        # Look for common answer patterns
        answer_patterns = [
            r"(?:Final Answer|Answer|Result):\s*([^(\n|(?=Explanation))]+)",
            r"(?:Therefore|Thus|So),\s*([^(\n|(?=Let me))]+)",
            r"(.{50,})$",  # Last substantial sentence
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                answer = matches[-1].strip()  # Take the last match
                if len(answer) > 10:  # Ensure it's substantial
                    return answer

        # Fallback: last non-empty line
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines[-1] if lines else text.strip()

    def _score_chain(self, chain: ReasoningChain) -> None:
        """Score the reasoning chain using multiple criteria."""
        if not chain.steps:
            chain.total_score = 0.0
            return

        # Score coherence (logical flow between steps)
        chain.coherence_score = self._score_coherence(chain.steps)

        # Score correctness (mathematical/logical validity)
        chain.correctness_score = self._score_correctness(chain.steps)

        # Score completeness (covers all aspects of the query)
        chain.completeness_score = self._score_completeness(chain.steps)

        # Calculate weighted total score
        chain.total_score = (
            self.weights["coherence"] * chain.coherence_score
            + self.weights["correctness"] * chain.correctness_score
            + self.weights["completeness"] * chain.completeness_score
        )

    def _score_coherence(self, steps: List[ChainStep]) -> float:
        """Score logical coherence between steps."""
        if len(steps) < 2:
            return 0.8  # Single step is coherent by default

        coherence_score = 0.0
        transitions = 0

        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            # Check for logical transitions
            transition_words = [
                "therefore",
                "thus",
                "so",
                "then",
                "next",
                "after",
                "because",
            ]
            has_transition = any(
                word in next_step.content.lower() for word in transition_words
            )

            # Check for topic continuity
            current_words = set(current_step.content.lower().split())
            next_words = set(next_step.content.lower().split())
            overlap = len(current_words.intersection(next_words)) / len(
                current_words.union(next_words)
            )

            step_coherence = (0.6 if has_transition else 0.3) + (0.4 * overlap)
            coherence_score += step_coherence
            transitions += 1

        return min(1.0, coherence_score / transitions) if transitions > 0 else 0.5

    def _score_correctness(self, steps: List[ChainStep]) -> float:
        """Score logical/mathematical correctness."""
        correctness_indicators = 0
        total_checks = 0

        for step in steps:
            content = step.content.lower()

            # Check for mathematical operations
            if any(op in content for op in ["+", "-", "*", "/", "=", "equals"]):
                # Basic math validation (simplified)
                has_numbers = bool(re.search(r"\d", content))
                has_operation = bool(re.search(r"[+\-*/=]", content))
                if has_numbers and has_operation:
                    correctness_indicators += 1
                total_checks += 1

            # Check for logical connectors
            logical_indicators = ["if", "then", "and", "or", "not", "implies"]
            if any(indicator in content for indicator in logical_indicators):
                # Basic logical structure check
                has_premise = bool(re.search(r"\b(if|when|given)\b", content))
                has_conclusion = bool(re.search(r"\b(then|therefore|thus)\b", content))
                if has_premise or has_conclusion:
                    correctness_indicators += 0.5
                total_checks += 1

            # Check for citations or references (indicates grounded reasoning)
            if (
                "according to" in content
                or "based on" in content
                or "source" in content
            ):
                correctness_indicators += 1
                total_checks += 1

        return (correctness_indicators / total_checks) if total_checks > 0 else 0.7

    def _score_completeness(self, steps: List[ChainStep]) -> float:
        """Score whether the reasoning is complete and comprehensive."""
        if not steps:
            return 0.0

        completeness_indicators = 0
        total_possible = 4  # Maximum completeness score

        # Check for problem analysis
        has_analysis = any(step.reasoning_type == "analysis" for step in steps)
        if has_analysis:
            completeness_indicators += 1

        # Check for step-by-step approach
        has_multiple_steps = len(steps) >= 3
        if has_multiple_steps:
            completeness_indicators += 1

        # Check for conclusion
        has_conclusion = any(step.reasoning_type == "conclusion" for step in steps)
        if has_conclusion:
            completeness_indicators += 1

        # Check for comprehensive coverage (length and detail)
        total_content_length = sum(len(step.content) for step in steps)
        has_sufficient_detail = total_content_length > 200  # Arbitrary threshold
        if has_sufficient_detail:
            completeness_indicators += 1

        return completeness_indicators / total_possible

    async def score_candidates(
        self, candidates: List[str], query: str
    ) -> List[ReasoningChain]:
        """Score multiple candidate reasoning chains and return ranked results."""
        scored_chains = []

        for i, candidate in enumerate(candidates):
            chain_id = f"chain_{i + 1}"
            chain = self.parse_reasoning_chain(candidate, chain_id)
            scored_chains.append(chain)

        # Sort by total score (highest first)
        scored_chains.sort(key=lambda x: x.total_score, reverse=True)

        logger.info(
            f"Scored {len(scored_chains)} candidate chains for query: {query[:50]}..."
        )
        for chain in scored_chains[:3]:  # Log top 3
            logger.info(f"Chain {chain.chain_id}: Score {chain.total_score:.2f}")

        return scored_chains

    def select_best_chain(self, chains: List[ReasoningChain]) -> ReasoningChain:
        """Select the best reasoning chain from candidates."""
        if not chains:
            raise ValueError("No chains provided")

        # Return the highest-scoring chain
        return max(chains, key=lambda x: x.total_score)

    async def evaluate_chain_quality(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Provide detailed evaluation of a reasoning chain."""
        return {
            "chain_id": chain.chain_id,
            "total_score": chain.total_score,
            "coherence_score": chain.coherence_score,
            "correctness_score": chain.correctness_score,
            "completeness_score": chain.completeness_score,
            "step_count": len(chain.steps),
            "final_answer": chain.final_answer,
            "step_types": [step.reasoning_type for step in chain.steps],
            "recommendation": "high_confidence"
            if chain.total_score > 0.7
            else "medium_confidence"
            if chain.total_score > 0.5
            else "low_confidence",
        }
