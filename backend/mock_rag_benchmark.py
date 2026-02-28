#!/usr/bin/env python3
"""
Simplified RAG Benchmarking for Algorithm Testing
Tests enhanced RAG algorithms without heavy ML dependencies.
"""

import asyncio
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""

    query: str
    baseline_latency: float
    enhanced_latency: float
    baseline_tokens: int
    enhanced_tokens: int
    baseline_chunks: int
    enhanced_chunks: int
    baseline_score: float
    enhanced_score: float
    improvement_ratio: float


@dataclass
class TestQuery:
    """Test query with expected results."""

    query: str
    expected_keywords: List[str]
    content_type: str = "general"


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self):
        self.dimension = 384

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        import hashlib
        import numpy as np

        embeddings = []
        for text in texts:
            # Create deterministic embeddings based on text hash
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            np.random.seed(hash_int % 2**32)
            embedding = np.random.normal(0, 1, self.dimension).tolist()
            embeddings.append(embedding)
        return embeddings


class MockBM25Retriever:
    """Mock BM25 retriever for testing."""

    def __init__(self):
        self.documents = []

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retriever."""
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using mock BM25 scoring."""
        if not self.documents:
            return []

        # Simple mock scoring based on keyword matching
        results = []
        query_words = set(query.lower().split())

        for doc in self.documents:
            content = doc.get("content", "").lower()
            score = 0

            # Count matching words
            for word in query_words:
                if word in content:
                    score += 1

            if score > 0:
                results.append(
                    {
                        "id": doc.get("id", f"doc_{len(results)}"),
                        "content": doc.get("content", ""),
                        "score": score,
                        "metadata": doc.get("metadata", {}),
                    }
                )

        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class MockDenseRetriever:
    """Mock dense retriever for testing."""

    def __init__(self):
        self.documents = []
        self.embedding_model = MockEmbeddingModel()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents and generate embeddings."""
        self.documents = documents
        # Generate embeddings for all documents
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.encode(contents)

        # Store embeddings with documents
        for doc, embedding in zip(self.documents, embeddings):
            doc["_embedding"] = embedding

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using mock dense retrieval."""
        if not self.documents:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Calculate cosine similarity
        results = []
        for doc in self.documents:
            if "_embedding" in doc:
                doc_embedding = doc["_embedding"]
                # Simple dot product (mock cosine similarity)
                similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))

                results.append(
                    {
                        "id": doc.get("id", f"doc_{len(results)}"),
                        "content": doc.get("content", ""),
                        "score": similarity,
                        "metadata": doc.get("metadata", {}),
                    }
                )

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class MockHybridRetriever:
    """Mock hybrid retriever combining dense and sparse retrieval."""

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        fusion_algorithm: str = "reciprocal_rank_fusion",
        rrf_k: int = 40,  # Optimized RRF parameter (reduced from 60 for better fusion)
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_algorithm = fusion_algorithm
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid search."""
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k, **kwargs)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k, **kwargs)

        if self.fusion_algorithm == "reciprocal_rank_fusion":
            return self._reciprocal_rank_fusion(
                dense_results, sparse_results, self.rrf_k
            )
        else:
            # Simple concatenation and reranking
            combined = dense_results + sparse_results
            return sorted(combined, key=lambda x: x.get("score", 0), reverse=True)[
                :top_k
            ]

    def _reciprocal_rank_fusion(
        self, dense_results: List[Dict], sparse_results: List[Dict], k: int
    ) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion."""
        # Create score dictionaries
        dense_scores = {
            doc.get("id", f"dense_{rank}"): 1.0 / (rank + k)
            for rank, doc in enumerate(dense_results)
        }
        sparse_scores = {
            doc.get("id", f"sparse_{rank}"): 1.0 / (rank + k)
            for rank, doc in enumerate(sparse_results)
        }

        # Combine scores
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        fused_scores = {}

        for doc_id in all_ids:
            dense_score = dense_scores.get(doc_id, 0)
            sparse_score = sparse_scores.get(doc_id, 0)
            fused_scores[doc_id] = dense_score + sparse_score

        # Create final results list
        results = []
        for doc_id, score in sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        ):
            # Find the original document
            doc = None
            for d in dense_results + sparse_results:
                if d.get("id") == doc_id:
                    doc = d
                    break

            if doc:
                results.append({**doc, "score": score})

        return results[: len(dense_results)]  # Return same number as input


class MockQueryExpansion:
    """Mock query expansion for testing."""

    def __init__(self):
        # Enhanced synonym mappings for better expansion
        self.synonym_map = {
            "setup": ["configure", "initialize", "install", "deploy"],
            "api": ["endpoint", "interface", "service", "rest"],
            "deploy": ["production", "release", "launch", "publish"],
            "authentication": ["auth", "login", "security", "credentials"],
            "database": ["db", "storage", "data", "persistence"],
            "migration": ["schema", "update", "change", "version"],
            "monitoring": ["observability", "metrics", "logging", "tracing"],
            "routing": ["selection", "load balancing", "distribution", "dispatch"],
            "environment": ["env", "config", "settings", "variables"],
            "backend": ["server", "api", "service", "infrastructure"],
            "chat": ["conversation", "messaging", "communication", "dialog"],
            "rag": ["retrieval", "augmented", "generation", "search"],
            "llm": ["language model", "ai", "gpt", "model"],
            "provider": ["service", "platform", "vendor", "api"],
        }

    def expand_query(self, query: str) -> List[str]:
        """Expand query with enhanced synonyms and related terms."""
        expansions = [query]
        query_lower = query.lower()

        # Add synonym expansions
        for term, synonyms in self.synonym_map.items():
            if term in query_lower:
                for synonym in synonyms:
                    # Replace the term with its synonym
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expansions.append(expanded)

        # Add compound term expansions (e.g., "api endpoint" -> "rest api")
        compound_mappings = {
            "api endpoint": ["rest api", "web service", "http endpoint"],
            "database migration": ["schema migration", "db update", "data migration"],
            "environment variable": ["env var", "config variable", "system variable"],
            "llm provider": ["ai service", "language model service", "ai provider"],
        }

        for compound, alternatives in compound_mappings.items():
            if compound in query_lower:
                for alt in alternatives:
                    expansions.append(query_lower.replace(compound, alt))

        # Add question pattern variations
        if query_lower.startswith(("how", "what", "where", "when", "why")):
            # Add alternative question forms
            if "how" in query_lower:
                expansions.append(query_lower.replace("how", "what is the process"))
                expansions.append(query_lower.replace("how", "what are the steps"))
            if "what" in query_lower:
                expansions.append(query_lower.replace("what", "which"))
                expansions.append(query_lower.replace("what", "list"))

        return list(set(expansions))  # Remove duplicates


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self):
        pass

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Mock reranking based on simple heuristics."""
        reranked = []
        for doc in documents:
            # Boost score based on query keyword matches
            content = doc.get("content", "").lower()
            query_words = query.lower().split()
            boost = sum(1 for word in query_words if word in content)

            new_score = doc.get("score", 0) + boost * 0.1
            reranked.append({**doc, "score": new_score})

        return sorted(reranked, key=lambda x: x["score"], reverse=True)[:top_k]


class MockRAGService:
    """Mock RAG service for testing algorithms."""

    def __init__(self, enable_enhanced: bool = False):
        self.enable_enhanced = enable_enhanced
        self.documents = []
        self.dense_retriever = MockDenseRetriever()
        self.sparse_retriever = MockBM25Retriever()
        self.hybrid_retriever = None
        self.query_expander = MockQueryExpansion()
        self.reranker = MockCrossEncoderReranker(
            threshold=0.5
        )  # Lower threshold for better latency

        if enable_enhanced:
            self.hybrid_retriever = MockHybridRetriever(
                self.dense_retriever, self.sparse_retriever
            )

    async def add_documents(
        self, documents: List[Dict[str, Any]], collection_name: str = "documents"
    ) -> bool:
        """Add documents to the mock RAG system."""
        try:
            self.documents.extend(documents)
            self.dense_retriever.add_documents(documents)
            self.sparse_retriever.add_documents(documents)
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    async def retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Retrieve documents using the mock RAG system."""
        start_time = time.time()

        try:
            if self.enable_enhanced and self.hybrid_retriever:
                # Enhanced retrieval with query expansion and reranking
                expanded_queries = self.query_expander.expand_query(query)

                # Get results for all expanded queries
                all_results = []
                for exp_query in expanded_queries:
                    results = self.hybrid_retriever.retrieve(exp_query, top_k=top_k * 2)
                    all_results.extend(results)

                # Remove duplicates and rerank
                seen_ids = set()
                unique_results = []
                for result in all_results:
                    doc_id = result.get("id")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_results.append(result)

                # Rerank final results
                final_results = self.reranker.rerank(query, unique_results, top_k=top_k)
            else:
                # Baseline retrieval
                final_results = self.dense_retriever.retrieve(query, top_k=top_k)

            latency = time.time() - start_time

            # Calculate token estimate (rough approximation)
            total_tokens = sum(
                len(result.get("content", "").split()) for result in final_results
            )

            return {
                "results": final_results,
                "latency": latency,
                "tokens": total_tokens,
                "chunks": len(final_results),
            }

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "results": [],
                "latency": time.time() - start_time,
                "tokens": 0,
                "chunks": 0,
            }


class MockCrossEncoderReranker:
    """Optimized mock cross-encoder reranker for testing."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank documents using optimized cross-encoder scoring."""
        reranked = []

        for doc in documents:
            # Optimized relevance calculation
            content = doc.get("content", "").lower()
            query_lower = query.lower()

            # Calculate relevance score based on keyword matches and semantic similarity
            relevance_score = self._calculate_relevance(query_lower, content)

            # Only include documents above threshold for efficiency
            if relevance_score >= self.threshold:
                reranked.append(
                    {
                        **doc,
                        "rerank_score": relevance_score,
                        "score": relevance_score,  # Update the main score
                    }
                )

        # Sort by rerank score and return top_k
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate optimized relevance score."""
        score = 0.0

        # Exact phrase matches get highest score
        if query in content:
            score += 0.9

        # Word matches with position weighting
        query_words = set(query.split())
        content_words = set(content.split())
        word_overlap = len(query_words & content_words)
        if word_overlap > 0:
            score += min(0.3, word_overlap * 0.1)

        # Semantic similarity with expanded mappings
        semantic_keywords = {
            "setup": ["configure", "initialize", "install", "deploy"],
            "api": ["endpoint", "interface", "service", "rest"],
            "deploy": ["production", "release", "launch", "publish"],
            "auth": ["authentication", "login", "security", "credentials"],
            "database": ["db", "storage", "data", "persistence"],
            "monitor": ["observability", "metrics", "logging", "tracing"],
            "routing": ["selection", "load", "balancing", "dispatch"],
            "environment": ["env", "config", "variable", "settings"],
        }

        for term, synonyms in semantic_keywords.items():
            if term in query and any(syn in content for syn in synonyms):
                score += 0.2

        return min(1.0, score)


class MockRAGBenchmarker:
    """Mock RAG benchmarking suite for algorithm testing."""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)
        self.test_collections = {}
        self.test_queries = self._create_test_queries()
        self.results = []

    def _create_test_queries(self) -> List[TestQuery]:
        """Create comprehensive test queries."""
        return [
            TestQuery(
                query="How do I set up the backend environment variables?",
                expected_keywords=[
                    "DATABASE_URL",
                    "JWT_SECRET_KEY",
                    "KAMATERA_HOST",
                    ".env",
                ],
                content_type="general",
            ),
            TestQuery(
                query="What are the main API endpoints for chat functionality?",
                expected_keywords=["chat", "POST", "/api/chat", "websocket"],
                content_type="general",
            ),
            TestQuery(
                query="How does the RAG system work in this project?",
                expected_keywords=[
                    "retrieval",
                    "augmented",
                    "generation",
                    "chromadb",
                    "embedding",
                ],
                content_type="technical",
            ),
            TestQuery(
                query="What authentication methods are supported?",
                expected_keywords=["JWT", "WebAuthn", "passkeys", "bcrypt"],
                content_type="security",
            ),
            TestQuery(
                query="How to deploy the application to production?",
                expected_keywords=["docker", "fly.io", "kamatera", "production"],
                content_type="devops",
            ),
            TestQuery(
                query="What monitoring and observability features are available?",
                expected_keywords=["prometheus", "datadog", "opentelemetry", "metrics"],
                content_type="monitoring",
            ),
            TestQuery(
                query="How does the routing service select LLM providers?",
                expected_keywords=["routing", "provider", "selection", "llm"],
                content_type="technical",
            ),
            TestQuery(
                query="What database migrations are available?",
                expected_keywords=["migration", "database", "alembic", "schema"],
                content_type="database",
            ),
        ]

    def collect_documents(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect documents from project files."""
        collections = {
            "readme_files": [],
            "python_files": [],
            "config_files": [],
            "documentation": [],
        }

        # Walk through project directory
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not any(
                skip in str(file_path)
                for skip in [".git", "__pycache__", "node_modules"]
            ):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                    # Skip empty files or very small files
                    if len(content.strip()) < 100:
                        continue

                    doc = {
                        "id": str(file_path.relative_to(self.project_root)),
                        "content": content,
                        "metadata": {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "file_extension": file_path.suffix,
                            "file_size": len(content),
                        },
                    }

                    # Categorize documents
                    if (
                        file_path.name.lower().startswith("readme")
                        or file_path.name.lower() == "readme.md"
                    ):
                        collections["readme_files"].append(doc)
                    elif file_path.suffix == ".py":
                        collections["python_files"].append(doc)
                    elif file_path.suffix in [
                        ".yml",
                        ".yaml",
                        ".json",
                        ".config",
                        ".cfg",
                    ]:
                        collections["config_files"].append(doc)
                    elif file_path.suffix in [".md", ".txt", ".rst"]:
                        collections["documentation"].append(doc)

                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    continue

        return collections

    def _calculate_quality_score(
        self, query: TestQuery, results: List[Dict[str, Any]]
    ) -> float:
        """Calculate quality score based on keyword matching."""
        if not results:
            return 0.0

        total_score = 0.0
        max_possible_score = len(query.expected_keywords) * len(results)

        for result in results:
            content = result.get("content", "").lower()
            for keyword in query.expected_keywords:
                if keyword.lower() in content:
                    total_score += 1.0

        return total_score / max_possible_score if max_possible_score > 0 else 0.0

    async def run_single_benchmark(
        self, query: TestQuery, rag_service
    ) -> BenchmarkResult:
        """Run a single benchmark query."""
        logger.info(f"Benchmarking: {query.query[:50]}...")

        # Run retrieval
        retrieval_result = await rag_service.retrieve(query.query, top_k=10)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            query, retrieval_result["results"]
        )

        return BenchmarkResult(
            query=query.query,
            baseline_latency=retrieval_result["latency"],
            enhanced_latency=retrieval_result["latency"],  # Same for mock
            baseline_tokens=retrieval_result["tokens"],
            enhanced_tokens=retrieval_result["tokens"],  # Same for mock
            baseline_chunks=retrieval_result["chunks"],
            enhanced_chunks=retrieval_result["chunks"],  # Same for mock
            baseline_score=quality_score,
            enhanced_score=quality_score,  # Same for mock
            improvement_ratio=1.0,  # No improvement in mock
        )

    async def run_benchmarks(self):
        """Run all benchmarks."""
        logger.info("🚀 Starting Mock RAG Benchmarking Suite")
        logger.info("=" * 50)

        # Collect documents
        self.test_collections = self.collect_documents()
        total_docs = sum(len(docs) for docs in self.test_collections.values())
        logger.info(
            f"Collected {total_docs} documents across {len(self.test_collections)} collections"
        )

        for name, docs in self.test_collections.items():
            logger.info(f"  {name}: {len(docs)} documents")

        # Initialize mock RAG systems
        baseline_rag = MockRAGService(enable_enhanced=False)
        enhanced_rag = MockRAGService(enable_enhanced=True)

        # Add documents to both systems
        logger.info("Populating test data...")
        for collection_name, documents in self.test_collections.items():
            logger.info(f"Adding {len(documents)} documents to {collection_name}")

            await baseline_rag.add_documents(documents, collection_name)
            await enhanced_rag.add_documents(documents, collection_name)

        # Run benchmarks
        logger.info("Starting RAG benchmarks...")
        for i, query in enumerate(self.test_queries, 1):
            logger.info(f"Running benchmark {i}/{len(self.test_queries)}")

            # Run baseline
            baseline_result = await self.run_single_benchmark(query, baseline_rag)

            # Run enhanced
            enhanced_result = await self.run_single_benchmark(query, enhanced_rag)

            # Calculate improvement
            if baseline_result.baseline_score > 0:
                improvement_ratio = (
                    enhanced_result.enhanced_score / baseline_result.baseline_score
                )
            else:
                improvement_ratio = 1.0 if enhanced_result.enhanced_score > 0 else 0.0

            # Store result
            result = BenchmarkResult(
                query=query.query,
                baseline_latency=baseline_result.baseline_latency,
                enhanced_latency=enhanced_result.enhanced_latency,
                baseline_tokens=baseline_result.baseline_tokens,
                enhanced_tokens=enhanced_result.enhanced_tokens,
                baseline_chunks=baseline_result.baseline_chunks,
                enhanced_chunks=enhanced_result.enhanced_chunks,
                baseline_score=baseline_result.baseline_score,
                enhanced_score=enhanced_result.enhanced_score,
                improvement_ratio=improvement_ratio,
            )

            self.results.append(result)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No benchmark results available"}

        # Calculate statistics
        baseline_latencies = [r.baseline_latency for r in self.results]
        enhanced_latencies = [r.enhanced_latency for r in self.results]
        baseline_scores = [r.baseline_score for r in self.results]
        enhanced_scores = [r.enhanced_score for r in self.results]
        improvement_ratios = [r.improvement_ratio for r in self.results]

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "valid_results": len(self.results),
            "collections": {
                name: len(docs) for name, docs in self.test_collections.items()
            },
            "performance_metrics": {
                "baseline": {
                    "avg_latency": statistics.mean(baseline_latencies)
                    if baseline_latencies
                    else 0,
                    "median_latency": statistics.median(baseline_latencies)
                    if baseline_latencies
                    else 0,
                    "avg_score": statistics.mean(baseline_scores)
                    if baseline_scores
                    else 0,
                    "median_score": statistics.median(baseline_scores)
                    if baseline_scores
                    else 0,
                },
                "enhanced": {
                    "avg_latency": statistics.mean(enhanced_latencies)
                    if enhanced_latencies
                    else 0,
                    "median_latency": statistics.median(enhanced_latencies)
                    if enhanced_latencies
                    else 0,
                    "avg_score": statistics.mean(enhanced_scores)
                    if enhanced_scores
                    else 0,
                    "median_score": statistics.median(enhanced_scores)
                    if enhanced_scores
                    else 0,
                },
                "improvement": {
                    "avg_ratio": statistics.mean(improvement_ratios)
                    if improvement_ratios
                    else 0,
                    "median_ratio": statistics.median(improvement_ratios)
                    if improvement_ratios
                    else 0,
                    "latency_overhead": (
                        statistics.mean(enhanced_latencies)
                        - statistics.mean(baseline_latencies)
                    )
                    if enhanced_latencies and baseline_latencies
                    else 0,
                },
            },
            "detailed_results": [
                {
                    "query": r.query,
                    "baseline_score": r.baseline_score,
                    "enhanced_score": r.enhanced_score,
                    "improvement_ratio": r.improvement_ratio,
                    "baseline_latency": r.baseline_latency,
                    "enhanced_latency": r.enhanced_latency,
                    "baseline_chunks": r.baseline_chunks,
                    "enhanced_chunks": r.enhanced_chunks,
                }
                for r in self.results
            ],
        }

        return report

    def save_report(self, filename: str = "mock_rag_benchmark_report.json"):
        """Save benchmark report to file."""
        report = self.generate_report()

        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {filename}")
        return report


async def main():
    """Main benchmarking function."""
    benchmarker = MockRAGBenchmarker()

    try:
        await benchmarker.run_benchmarks()
        report = benchmarker.save_report()

        # Print summary
        print("\n📊 Mock Benchmark Summary")
        print("-" * 30)
        print(f"Total Queries: {report['total_queries']}")
        print(f"Valid Results: {report['valid_results']}")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")

        print("\n✅ Mock benchmarking complete!")

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        print(f"❌ Benchmarking failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
