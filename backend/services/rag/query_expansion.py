"""
Query Expansion for Enhanced RAG
Expands queries using synonyms, related terms, and acronym expansion.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import nltk
    from nltk.corpus import wordnet

    # Download required NLTK data
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available for query expansion")


class QueryExpansion:
    """Query expansion using various strategies."""

    def __init__(self, llm_backend=None):
        self.llm = llm_backend

    async def expand_query(self, query: str, strategies: List[str] = None) -> List[str]:
        """Expand query using multiple strategies."""
        if strategies is None:
            strategies = ["synonyms", "related_terms"]

        expanded_queries = [query]  # Always include original

        for strategy in strategies:
            if strategy == "synonyms":
                expanded_queries.extend(self._expand_synonyms(query))
            elif strategy == "related_terms":
                expanded_queries.extend(await self._expand_related_terms(query))
            elif strategy == "acronyms":
                expanded_queries.extend(self._expand_acronyms(query))

        # Remove duplicates and limit
        expanded_queries = list(set(expanded_queries))[:5]  # Max 5 expanded queries
        return expanded_queries

    def _expand_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms using WordNet."""
        if not NLTK_AVAILABLE:
            logger.warning("WordNet not available for synonym expansion")
            return []

        try:
            words = query.split()
            expanded = []

            for word in words:
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace("_", " ")
                        if synonym != word and synonym not in synonyms:
                            synonyms.append(synonym)

                # Create variations with synonyms
                for synonym in synonyms[:3]:  # Limit synonyms per word
                    new_query = query.replace(word, synonym)
                    if new_query != query:
                        expanded.append(new_query)

            return expanded
        except Exception as e:
            logger.warning(f"Failed to expand synonyms: {e}")
            return []

    async def _expand_related_terms(self, query: str) -> List[str]:
        """Expand query with related terms using LLM if available."""
        if not self.llm:
            return []

        try:
            prompt = f"""Given the query: "{query}"

Generate 2-3 related search terms or phrases that could help find relevant information.
Focus on technical terms, related concepts, or alternative phrasings.

Return only the terms/phrases, one per line, no explanations."""

            response = await self.llm.generate(prompt, max_tokens=100)
            terms = [line.strip() for line in response.split("\n") if line.strip()]
            return terms[:3]
        except Exception as e:
            logger.warning(f"Failed to expand query with LLM: {e}")
            return []

    def _expand_acronyms(self, query: str) -> List[str]:
        """Expand common acronyms in the query."""
        acronym_map = {
            "api": "application programming interface",
            "llm": "large language model",
            "rag": "retrieval augmented generation",
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "db": "database",
            "sql": "structured query language",
            "http": "hypertext transfer protocol",
            "json": "javascript object notation",
            "xml": "extensible markup language",
        }

        words = query.lower().split()
        expanded = []

        for acronym, expansion in acronym_map.items():
            if acronym in words:
                new_query = query.replace(acronym, expansion)
                expanded.append(new_query)

        return expanded
