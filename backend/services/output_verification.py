"""
Output Verification and Confidence Scoring Service

This module provides:
1. Safety verification using a 3B instruction-tuned model (llama3.2:3b-instruct)
2. Confidence scoring using phi3 to evaluate output quality
3. Escalation logic based on confidence thresholds
"""

import re
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of safety verification."""

    is_safe: bool
    safety_score: float  # 0.0 to 1.0
    issues: list
    explanation: str


@dataclass
class ConfidenceResult:
    """Result of confidence scoring."""

    confidence_score: float  # 0.0 to 1.0
    should_escalate: bool
    reasoning: str
    recommended_action: str


class OutputVerifier:
    """Verifies outputs for safety and alignment using a specialized model."""

    def __init__(self, adapter, verifier_model: str = "gemma:2b"):
        self.adapter = adapter
        self.verifier_model = verifier_model
        self.safety_threshold = 0.7  # Minimum safety score to pass

    async def verify_output(
        self,
        original_prompt: str,
        model_output: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Verify an output for safety and alignment issues.

        Args:
            original_prompt: The user's original prompt
            model_output: The model's response to verify
            context: Optional context about the request

        Returns:
            VerificationResult with safety assessment
        """
        # Build verification prompt
        verification_prompt = self._build_verification_prompt(
            original_prompt, model_output, context
        )

        try:
            # Call verifier model
            response = await self.adapter.chat(
                model=self.verifier_model,
                messages=[{"role": "user", "content": verification_prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            # Parse verification response
            result = self._parse_verification_response(response)
            return result

        except Exception as e:
            # If verification fails, err on the side of caution
            return VerificationResult(
                is_safe=False,
                safety_score=0.0,
                issues=["verification_error"],
                explanation=f"Verification failed: {str(e)}",
            )

    def _build_verification_prompt(
        self, prompt: str, output: str, context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for safety verification."""

        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"

        return f"""You are a safety verification assistant. Analyze the following AI model output for potential issues.

USER PROMPT:
{prompt}

MODEL OUTPUT:
{output}
{context_str}

Check for these safety concerns:
1. Hallucination - Does the output contain fabricated information?
2. Harmful content - Is there dangerous, unethical, or harmful advice?
3. Bias - Does it show unfair bias or discrimination?
4. Off-topic - Does it fail to address the user's question?
5. Overconfidence - Does it claim certainty about uncertain things?

Respond ONLY in this exact JSON format:
{{
  "is_safe": true/false,
  "safety_score": 0.0-1.0,
  "issues": ["list", "of", "issues"],
  "explanation": "brief explanation"
}}"""

    def _parse_verification_response(self, response: str) -> VerificationResult:
        """Parse JSON response from verifier model."""

        # Try to extract JSON
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return VerificationResult(
                    is_safe=data.get("is_safe", False),
                    safety_score=float(data.get("safety_score", 0.0)),
                    issues=data.get("issues", []),
                    explanation=data.get("explanation", ""),
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: heuristic parsing
        response_lower = response.lower()

        is_safe = 'is_safe": true' in response_lower or "safe" in response_lower

        issues = []
        if "hallucination" in response_lower:
            issues.append("hallucination")
        if "harmful" in response_lower or "dangerous" in response_lower:
            issues.append("harmful_content")
        if "bias" in response_lower:
            issues.append("bias")
        if "off-topic" in response_lower or "irrelevant" in response_lower:
            issues.append("off_topic")
        if "overconfident" in response_lower:
            issues.append("overconfidence")

        safety_score = 0.8 if is_safe and not issues else 0.3

        return VerificationResult(
            is_safe=is_safe and len(issues) == 0,
            safety_score=safety_score,
            issues=issues,
            explanation=response[:200],
        )


class ConfidenceScorer:
    """Scores output confidence using phi3 and determines escalation needs."""

    def __init__(self, adapter, scoring_model: str = "phi3:3.8b"):
        self.adapter = adapter
        self.scoring_model = scoring_model
        self.confidence_threshold = 0.65  # Below this, escalate
        self.critical_threshold = 0.4  # Below this, reject immediately

    async def score_confidence(
        self,
        original_prompt: str,
        model_output: str,
        model_used: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceResult:
        """
        Score the confidence/quality of a model output.

        Args:
            original_prompt: The user's original prompt
            model_output: The model's response to score
            model_used: Which model generated the output
            context: Optional context about the request

        Returns:
            ConfidenceResult with scoring and escalation decision
        """
        # Build scoring prompt
        scoring_prompt = self._build_scoring_prompt(
            original_prompt, model_output, model_used, context
        )

        try:
            # Call scoring model
            response = await self.adapter.chat(
                model=self.scoring_model,
                messages=[{"role": "user", "content": scoring_prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            # Parse scoring response
            result = self._parse_scoring_response(response)

            # Determine escalation
            result.should_escalate = result.confidence_score < self.confidence_threshold

            if result.confidence_score < self.critical_threshold:
                result.recommended_action = "reject"
            elif result.confidence_score < self.confidence_threshold:
                result.recommended_action = "escalate_to_better_model"
            else:
                result.recommended_action = "accept"

            return result

        except Exception as e:
            # If scoring fails, escalate out of caution
            return ConfidenceResult(
                confidence_score=0.0,
                should_escalate=True,
                reasoning=f"Scoring failed: {str(e)}",
                recommended_action="escalate_to_better_model",
            )

    def _build_scoring_prompt(
        self,
        prompt: str,
        output: str,
        model_used: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt for confidence scoring."""

        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"

        return f"""You are evaluating the quality and confidence of an AI model's output.

USER PROMPT:
{prompt}

MODEL OUTPUT (from {model_used}):
{output}
{context_str}

Rate the output on these criteria (0.0 to 1.0):
1. Relevance - Does it answer the question?
2. Completeness - Is the answer sufficient?
3. Accuracy - Does it seem factually correct?
4. Clarity - Is it well-explained?
5. Confidence - Does the model seem certain?

Respond ONLY in this exact JSON format:
{{
  "confidence_score": 0.0-1.0,
  "reasoning": "brief explanation of score"
}}"""

    def _parse_scoring_response(self, response: str) -> ConfidenceResult:
        """Parse JSON response from scoring model."""

        # Try to extract JSON
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                confidence_score = float(data.get("confidence_score", 0.5))
                reasoning = data.get("reasoning", "")

                return ConfidenceResult(
                    confidence_score=confidence_score,
                    should_escalate=False,  # Will be set by caller
                    reasoning=reasoning,
                    recommended_action="pending",
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: heuristic parsing
        response_lower = response.lower()

        # Look for confidence indicators
        score = 0.5  # Default moderate confidence

        if any(
            word in response_lower
            for word in ["excellent", "very good", "strong", "high confidence"]
        ):
            score = 0.85
        elif any(word in response_lower for word in ["good", "adequate", "reasonable"]):
            score = 0.7
        elif any(
            word in response_lower for word in ["uncertain", "incomplete", "lacking"]
        ):
            score = 0.4
        elif any(word in response_lower for word in ["poor", "inadequate", "failed"]):
            score = 0.2

        return ConfidenceResult(
            confidence_score=score,
            should_escalate=False,
            reasoning=response[:200],
            recommended_action="pending",
        )


class VerificationPipeline:
    """Orchestrates verification and confidence scoring with escalation."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.verifier = OutputVerifier(adapter)
        self.scorer = ConfidenceScorer(adapter)
        self.escalation_map = {
            "gemma:2b": "phi3:3.8b",
            "phi3:3.8b": "qwen2.5:3b",
            "qwen2.5:3b": "mistral:7b",
            "mistral:7b": None,  # No further escalation
        }

    async def verify_and_score(
        self,
        original_prompt: str,
        model_output: str,
        model_used: str,
        context: Optional[Dict[str, Any]] = None,
        skip_verification: bool = False,
    ) -> Tuple[VerificationResult, ConfidenceResult]:
        """
        Run both verification and confidence scoring.

        Args:
            original_prompt: User's prompt
            model_output: Model's response
            model_used: Which model generated the response
            context: Optional context
            skip_verification: Skip safety verification (for speed)

        Returns:
            Tuple of (VerificationResult, ConfidenceResult)
        """
        # Run verification and scoring in parallel if not skipping
        if skip_verification:
            verification = VerificationResult(
                is_safe=True,
                safety_score=1.0,
                issues=[],
                explanation="Verification skipped",
            )
        else:
            verification = await self.verifier.verify_output(
                original_prompt, model_output, context
            )

        confidence = await self.scorer.score_confidence(
            original_prompt, model_output, model_used, context
        )

        return verification, confidence

    def get_escalation_target(self, current_model: str) -> Optional[str]:
        """Get the next model to escalate to."""
        return self.escalation_map.get(current_model)

    def should_reject_output(
        self, verification: VerificationResult, confidence: ConfidenceResult
    ) -> bool:
        """Determine if output should be rejected entirely."""

        # Reject if unsafe
        if not verification.is_safe:
            return True

        # Reject if critically low confidence
        if confidence.confidence_score < self.scorer.critical_threshold:
            return True

        # Reject if critical safety issues
        critical_issues = ["harmful_content", "hallucination"]
        if any(issue in verification.issues for issue in critical_issues):
            return True

        return False

    def should_escalate(
        self,
        verification: VerificationResult,
        confidence: ConfidenceResult,
        current_model: str,
    ) -> bool:
        """Determine if output should be escalated to better model."""

        # Don't escalate if already at top model
        if self.get_escalation_target(current_model) is None:
            return False

        # Escalate if confidence says so
        if confidence.should_escalate:
            return True

        # Escalate if safety concerns (but not critical)
        if verification.safety_score < 0.8 and verification.is_safe:
            return True

        return False
