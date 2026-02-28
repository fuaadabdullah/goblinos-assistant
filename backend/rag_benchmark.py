#!/usr/bin/env python3
"""
RAG Benchmarking and Testing Suite
Tests enhanced RAG system against baseline with real document collections.
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


class RAGBenchmarker:
    """Comprehensive RAG benchmarking suite."""

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
                expected_keywords=[
                    "routing",
                    "provider",
                    "selection",
                    "latency",
                    "cost",
                ],
                content_type="technical",
            ),
            TestQuery(
                query="What database migrations are available?",
                expected_keywords=["alembic", "migration", "supabase", "postgres"],
                content_type="database",
            ),
        ]

    def collect_documents(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect documents from the project for testing."""
        logger.info("Collecting documents from project...")

        collections = {
            "readme_files": [],
            "python_files": [],
            "config_files": [],
            "documentation": [],
        }

        # Collect README and documentation files
        for md_file in self.project_root.rglob("*.md"):
            if (
                md_file.name.lower().startswith("readme")
                or "doc" in str(md_file).lower()
            ):
                try:
                    content = md_file.read_text(encoding="utf-8")
                    if len(content.strip()) > 100:  # Only substantial files
                        collections["readme_files"].append(
                            {
                                "id": f"readme_{md_file.name}",
                                "content": content,
                                "metadata": {
                                    "source": str(
                                        md_file.relative_to(self.project_root)
                                    ),
                                    "type": "documentation",
                                    "filename": md_file.name,
                                    "content_type": "general",
                                },
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to read {md_file}: {e}")

        # Collect Python files (key modules)
        key_python_files = [
            "main.py",
            "routing_router.py",
            "chat_router.py",
            "services/rag_service.py",
            "services/enhanced_rag_service.py",
        ]

        for py_file in self.project_root.rglob("*.py"):
            if py_file.name in key_python_files or "service" in str(py_file):
                try:
                    content = py_file.read_text(encoding="utf-8")
                    if len(content.strip()) > 500:  # Only substantial files
                        # Extract docstrings and comments for better content
                        lines = content.split("\n")
                        doc_lines = []
                        in_docstring = False

                        for line in lines[:100]:  # First 100 lines
                            stripped = line.strip()
                            if '"""' in line or "'''" in line:
                                in_docstring = not in_docstring
                                if in_docstring:
                                    doc_lines.append(stripped)
                            elif in_docstring or stripped.startswith("#"):
                                doc_lines.append(stripped)

                        doc_content = "\n".join(doc_lines)
                        if len(doc_content) > 50:
                            collections["python_files"].append(
                                {
                                    "id": f"python_{py_file.name}",
                                    "content": doc_content,
                                    "metadata": {
                                        "source": str(
                                            py_file.relative_to(self.project_root)
                                        ),
                                        "type": "code",
                                        "filename": py_file.name,
                                        "content_type": "code",
                                    },
                                }
                            )
                except Exception as e:
                    logger.warning(f"Failed to read {py_file}: {e}")

        # Collect configuration files
        for config_file in self.project_root.rglob("*.yml"):
            if "config" in str(config_file).lower():
                try:
                    content = config_file.read_text(encoding="utf-8")
                    collections["config_files"].append(
                        {
                            "id": f"config_{config_file.name}",
                            "content": content,
                            "metadata": {
                                "source": str(
                                    config_file.relative_to(self.project_root)
                                ),
                                "type": "configuration",
                                "filename": config_file.name,
                                "content_type": "general",
                            },
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to read {config_file}: {e}")

        # Summary
        total_docs = sum(len(docs) for docs in collections.values())
        logger.info(
            f"Collected {total_docs} documents across {len(collections)} collections"
        )

        for collection_name, docs in collections.items():
            logger.info(f"  {collection_name}: {len(docs)} documents")

        return collections

    async def setup_rag_systems(self):
        """Initialize both baseline and enhanced RAG systems."""
        logger.info("Setting up RAG systems...")

        # Collect documents
        self.test_collections = self.collect_documents()

        # Initialize baseline RAG (current system)
        try:
            from services.rag_service import RAGService

            self.baseline_rag = RAGService(enable_enhanced=False)
            logger.info("✅ Baseline RAG initialized")
        except Exception as e:
            logger.error(f"Failed to initialize baseline RAG: {e}")
            self.baseline_rag = None

        # Initialize enhanced RAG
        try:
            from services.rag_service import RAGService

            self.enhanced_rag = RAGService(enable_enhanced=True)
            logger.info("✅ Enhanced RAG initialized")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced RAG: {e}")
            self.enhanced_rag = None

    async def populate_test_data(self):
        """Add test documents to both RAG systems."""
        logger.info("Populating test data...")

        for collection_name, documents in self.test_collections.items():
            logger.info(f"Adding {len(documents)} documents to {collection_name}")

            # Add to baseline RAG
            if self.baseline_rag:
                success = await self.baseline_rag.add_documents(
                    documents, collection_name
                )
                logger.info(
                    f"Baseline RAG: {'✅' if success else '❌'} {collection_name}"
                )

            # Add to enhanced RAG
            if self.enhanced_rag:
                for doc in documents:
                    content_type = doc.get("metadata", {}).get(
                        "content_type", "general"
                    )
                    success = (
                        await self.enhanced_rag._get_enhanced_service().add_documents(
                            [doc], collection_name, content_type
                        )
                    )
                logger.info(f"Enhanced RAG: ✅ {collection_name}")

    def evaluate_retrieval_quality(self, query: TestQuery, chunks: List[Dict]) -> float:
        """Evaluate retrieval quality based on expected keywords."""
        if not chunks:
            return 0.0

        # Combine all retrieved text
        retrieved_text = " ".join([chunk.get("text", "") for chunk in chunks]).lower()

        # Count matched keywords
        matched_keywords = 0
        for keyword in query.expected_keywords:
            if keyword.lower() in retrieved_text:
                matched_keywords += 1

        # Calculate score (0.0 to 1.0)
        score = matched_keywords / len(query.expected_keywords)

        # Bonus for relevant chunks
        relevant_chunks = sum(
            1
            for chunk in chunks
            if any(
                kw.lower() in chunk.get("text", "").lower()
                for kw in query.expected_keywords
            )
        )
        relevance_bonus = min(0.2, relevant_chunks * 0.05)  # Up to 20% bonus

        return min(1.0, score + relevance_bonus)

    async def benchmark_query(self, query: TestQuery) -> BenchmarkResult:
        """Benchmark a single query against both systems."""
        logger.info(f"Benchmarking: {query.query[:50]}...")

        # Test baseline RAG
        baseline_result = None
        baseline_start = time.time()
        if self.baseline_rag:
            try:
                baseline_result = await self.baseline_rag.rag_pipeline(query.query)
                baseline_latency = time.time() - baseline_start
            except Exception as e:
                logger.warning(f"Baseline RAG failed: {e}")
                baseline_latency = float("inf")
        else:
            baseline_latency = float("inf")

        # Test enhanced RAG
        enhanced_result = None
        enhanced_start = time.time()
        if self.enhanced_rag:
            try:
                enhanced_result = await self.enhanced_rag.enhanced_rag_pipeline(
                    query.query, use_hybrid=True, use_reranking=True, expand_query=True
                )
                enhanced_latency = time.time() - enhanced_start
            except Exception as e:
                logger.warning(f"Enhanced RAG failed: {e}")
                enhanced_latency = float("inf")
        else:
            enhanced_latency = float("inf")

        # Extract metrics
        baseline_tokens = (
            baseline_result.get("context", {}).get("total_tokens", 0)
            if baseline_result
            else 0
        )
        enhanced_tokens = (
            enhanced_result.get("context", {}).get("total_tokens", 0)
            if enhanced_result
            else 0
        )
        baseline_chunks = (
            len(baseline_result.get("context", {}).get("chunks", []))
            if baseline_result
            else 0
        )
        enhanced_chunks = (
            len(enhanced_result.get("context", {}).get("chunks", []))
            if enhanced_result
            else 0
        )

        # Evaluate quality
        baseline_score = (
            self.evaluate_retrieval_quality(
                query, baseline_result.get("context", {}).get("chunks", [])
            )
            if baseline_result
            else 0.0
        )
        enhanced_score = (
            self.evaluate_retrieval_quality(
                query, enhanced_result.get("context", {}).get("chunks", [])
            )
            if enhanced_result
            else 0.0
        )

        # Calculate improvement ratio
        if baseline_score > 0:
            improvement_ratio = enhanced_score / baseline_score
        else:
            improvement_ratio = float("inf") if enhanced_score > 0 else 1.0

        return BenchmarkResult(
            query=query.query,
            baseline_latency=baseline_latency,
            enhanced_latency=enhanced_latency,
            baseline_tokens=baseline_tokens,
            enhanced_tokens=enhanced_tokens,
            baseline_chunks=baseline_chunks,
            enhanced_chunks=enhanced_chunks,
            baseline_score=baseline_score,
            enhanced_score=enhanced_score,
            improvement_ratio=improvement_ratio,
        )

    async def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmarks."""
        logger.info("Starting RAG benchmarks...")

        results = []
        for i, query in enumerate(self.test_queries, 1):
            logger.info(f"Running benchmark {i}/{len(self.test_queries)}")
            try:
                result = await self.benchmark_query(query)
                results.append(result)
                logger.info(".2f")
            except Exception as e:
                logger.error(f"Benchmark failed for query: {query.query[:50]}...: {e}")

        self.results = results
        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No benchmark results available"}

        # Calculate statistics
        valid_results = [
            r
            for r in self.results
            if r.baseline_latency != float("inf") and r.enhanced_latency != float("inf")
        ]

        if not valid_results:
            return {"error": "No valid benchmark results"}

        baseline_latencies = [r.baseline_latency for r in valid_results]
        enhanced_latencies = [r.enhanced_latency for r in valid_results]
        baseline_scores = [r.baseline_score for r in valid_results]
        enhanced_scores = [r.enhanced_score for r in valid_results]
        improvement_ratios = [r.improvement_ratio for r in valid_results]

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "valid_results": len(valid_results),
            "collections": {
                name: len(docs) for name, docs in self.test_collections.items()
            },
            "performance_metrics": {
                "baseline": {
                    "avg_latency": statistics.mean(baseline_latencies),
                    "median_latency": statistics.median(baseline_latencies),
                    "avg_score": statistics.mean(baseline_scores),
                    "median_score": statistics.median(baseline_scores),
                },
                "enhanced": {
                    "avg_latency": statistics.mean(enhanced_latencies),
                    "median_latency": statistics.median(enhanced_latencies),
                    "avg_score": statistics.mean(enhanced_scores),
                    "median_score": statistics.median(enhanced_scores),
                },
                "improvement": {
                    "avg_ratio": statistics.mean(improvement_ratios),
                    "median_ratio": statistics.median(improvement_ratios),
                    "latency_overhead": statistics.mean(enhanced_latencies)
                    - statistics.mean(baseline_latencies),
                },
            },
            "detailed_results": [
                {
                    "query": r.query[:50] + "..." if len(r.query) > 50 else r.query,
                    "baseline_score": round(r.baseline_score, 3),
                    "enhanced_score": round(r.enhanced_score, 3),
                    "improvement_ratio": round(r.improvement_ratio, 3),
                    "baseline_latency": round(r.baseline_latency, 3),
                    "enhanced_latency": round(r.enhanced_latency, 3),
                    "baseline_chunks": r.baseline_chunks,
                    "enhanced_chunks": r.enhanced_chunks,
                }
                for r in valid_results
            ],
        }

        return report

    async def save_report(self, filename: str = "rag_benchmark_report.json"):
        """Save benchmark report to file."""
        report = self.generate_report()
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {filename}")


async def main():
    """Main benchmarking function."""
    print("🚀 RAG Benchmarking Suite")
    print("=" * 50)

    # Initialize benchmarker
    benchmarker = RAGBenchmarker()

    try:
        # Setup systems
        await benchmarker.setup_rag_systems()

        # Populate test data
        await benchmarker.populate_test_data()

        # Run benchmarks
        await benchmarker.run_benchmarks()

        # Generate and save report
        await benchmarker.save_report()

        # Print summary
        report = benchmarker.generate_report()
        print("\n📊 Benchmark Summary")
        print("-" * 30)
        print(f"Total Queries: {report['total_queries']}")
        print(f"Valid Results: {report['valid_results']}")

        perf = report["performance_metrics"]
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")

        print("\n✅ Benchmarking complete!")

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
