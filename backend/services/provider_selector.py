"""
Provider selector service.

Selects the highest-scoring candidate from a pre-scored list produced by
ProviderScorer.  When scores are tied the first candidate in the list wins
(preserves discovery order as a tiebreaker).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProviderSelector:
    """Select the best provider candidate from a scored list."""

    async def select_best_provider(
        self,
        scored_providers: List[Dict[str, Any]],
        requirements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return the candidate with the highest "score" value.

        Args:
            scored_providers: Candidates annotated with a "score" key by ProviderScorer.
            requirements: Optional additional requirements (reserved for future filtering).

        Returns:
            The best candidate dict.

        Raises:
            ValueError: When the input list is empty.
        """
        if not scored_providers:
            raise ValueError("No providers to select from — discovery returned an empty list.")

        # Stable sort: highest score first, original order as tiebreaker
        sorted_providers = sorted(
            scored_providers,
            key=lambda c: c.get("score", 0.0),
            reverse=True,
        )

        selection = sorted_providers[0]
        provider_name = selection.get("provider_id") or selection.get("name", str(selection))
        logger.info(
            "ProviderSelector: selected %s (score=%.4f) from %d candidates",
            provider_name,
            selection.get("score", 0.0),
            len(sorted_providers),
        )

        # Attach runner-up info for debugging / observability
        if len(sorted_providers) > 1:
            runner_up = sorted_providers[1]
            logger.debug(
                "Runner-up: %s (score=%.4f)",
                runner_up.get("provider_id") or runner_up.get("name", ""),
                runner_up.get("score", 0.0),
            )

        return selection
