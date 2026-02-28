"""
Enhanced Routing Service with multi-factor decision algorithm.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging

from sqlalchemy.orm import Session
from .routing import RoutingService

logger = logging.getLogger(__name__)


class EnhancedRoutingService(RoutingService):
    """Enhanced routing service with advanced decision factors."""

    def __init__(self, db: Session, encryption_key: str):
        super().__init__(db, encryption_key)

        # Time-based weights
        self.time_weights = {
            "business_hours": 1.0,  # 9 AM - 5 PM UTC
            "peak": 0.8,  # 6 PM - 10 PM UTC
            "off_peak": 1.2,  # 11 PM - 8 AM UTC
        }

        # User tier weights
        self.user_tier_weights = {
            "free": 0.7,
            "pro": 1.0,
            "enterprise": 1.3,
        }

    def get_time_weight(self) -> float:
        """Get current time-based weight factor."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        if 9 <= hour < 17:  # Business hours
            return self.time_weights["business_hours"]
        elif 18 <= hour < 22:  # Peak hours
            return self.time_weights["peak"]
        else:  # Off-peak
            return self.time_weights["off_peak"]

    def get_user_tier_weight(self, user_id: str) -> float:
        """Get user tier weight factor."""
        # Mock tier lookup - in real implementation, query database
        if "enterprise" in user_id.lower():
            return self.user_tier_weights["enterprise"]
        elif "pro" in user_id.lower():
            return self.user_tier_weights["pro"]
        else:
            return self.user_tier_weights["free"]

    def analyze_content_complexity(self, content: str) -> float:
        """Analyze content complexity (0.0-1.0)."""
        if not content:
            return 0.0

        # Simple heuristics
        word_count = len(content.split())
        sentence_count = len(content.split("."))

        # Complexity based on length and structure
        complexity = min(1.0, (word_count / 100) + (sentence_count / 10))

        return complexity

    def calculate_routing_score(
        self, provider: str, model: str, user_id: str, content: str
    ) -> float:
        """Calculate comprehensive routing score."""
        base_score = 1.0

        # Time factor
        time_weight = self.get_time_weight()

        # User tier factor
        tier_weight = self.get_user_tier_weight(user_id)

        # Content complexity factor
        complexity = self.analyze_content_complexity(content)
        complexity_weight = 1.0 + (complexity * 0.5)  # Boost for complex content

        # Combine factors
        final_score = base_score * time_weight * tier_weight * complexity_weight

        logger.info(
            f"Routing score for {provider}/{model}: {final_score:.3f} "
            f"(time: {time_weight}, tier: {tier_weight}, complexity: {complexity_weight})"
        )

        return final_score

    async def route_enhanced(
        self,
        user_id: str,
        content: str,
        conversation_context: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Enhanced routing with multi-factor decision algorithm."""
        try:
            # Get available providers from base service
            available_providers = await self.get_available_providers()

            if not available_providers:
                return {"error": "No providers available"}

            # Calculate scores for each provider/model combination
            scored_options = []
            for provider_data in available_providers:
                provider = provider_data["provider"]
                models = provider_data.get("models", [])

                for model in models:
                    score = self.calculate_routing_score(
                        provider, model, user_id, content
                    )
                    scored_options.append(
                        {
                            "provider": provider,
                            "model": model,
                            "score": score,
                            "data": provider_data,
                        }
                    )

            # Sort by score (highest first)
            scored_options.sort(key=lambda x: x["score"], reverse=True)

            # Return top choice
            best_option = scored_options[0]
            return {
                "provider": best_option["provider"],
                "model": best_option["model"],
                "score": best_option["score"],
                "reasoning": f"Selected based on time ({self.get_time_weight():.1f}), "
                f"tier ({self.get_user_tier_weight(user_id):.1f}), "
                f"complexity ({self.analyze_content_complexity(content):.2f})",
            }

        except Exception as e:
            logger.error(f"Enhanced routing failed: {e}")
            return {"error": str(e)}
