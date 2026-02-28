# pyright: reportMissingImports=false
"""
Latency Stress Test Suite
Tests phi3:3.8b and gemma:2b under production-level load.
Measures p50, p95, p99 latency, throughput, and error rates.
"""

import asyncio
import importlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))


@dataclass
class RequestResult:
    """Result of a single request."""

    start_time: float
    end_time: float
    latency_ms: float
    success: bool
    error: str = ""
    tokens: int = 0


class StressTestRunner:
    """Runs stress tests on LLM models."""

    def __init__(self, model_id: str, qps: int, duration_seconds: int):
        self.model_id = model_id
        self.qps = qps
        self.duration_seconds = duration_seconds
        self.results: list[RequestResult] = []

        ollama_base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")
        ollama_api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
        providers = importlib.import_module("providers")
        self.adapter = providers.OllamaAdapter(ollama_api_key, ollama_base_url)

    async def send_request(self, prompt: str) -> RequestResult:
        """Send a single request and record result."""
        start_time = time.time()

        try:
            response = await self.adapter.chat(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150,
            )

            end_time = time.time()
            latency = (end_time - start_time) * 1000
            tokens = len(response.split())

            return RequestResult(
                start_time=start_time,
                end_time=end_time,
                latency_ms=round(latency, 2),
                success=True,
                tokens=tokens,
            )

        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000

            return RequestResult(
                start_time=start_time,
                end_time=end_time,
                latency_ms=round(latency, 2),
                success=False,
                error=str(e),
            )

    async def run_stress_test(self, prompts: list[str]):
        """Run stress test with specified QPS for duration."""
        print(f"\n{'=' * 80}")
        print(f"STRESS TEST: {self.model_id}")
        print(f"{'=' * 80}")
        print(f"Target QPS: {self.qps}")
        print(f"Duration: {self.duration_seconds}s")
        print(f"Expected requests: {self.qps * self.duration_seconds}")

        # Calculate delay between requests
        delay_between_requests = 1.0 / self.qps

        test_start = time.time()
        request_count = 0

        # Create tasks
        tasks = []
        next_request_time = test_start

        while (time.time() - test_start) < self.duration_seconds:
            # Schedule next request
            current_time = time.time()

            if current_time >= next_request_time:
                # Select prompt (cycle through list)
                prompt = prompts[request_count % len(prompts)]

                # Create task
                task = asyncio.create_task(self.send_request(prompt))
                tasks.append(task)

                request_count += 1
                next_request_time += delay_between_requests

            # Small sleep to avoid busy waiting
            await asyncio.sleep(0.01)

        print(f"\n⏳ Sent {request_count} requests, waiting for responses...")

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and store results
        for result in results:
            if isinstance(result, RequestResult):
                self.results.append(result)
            else:
                # Exception occurred
                self.results.append(
                    RequestResult(
                        start_time=0,
                        end_time=0,
                        latency_ms=0,
                        success=False,
                        error=str(result),
                    )
                )

        test_end = time.time()
        total_duration = test_end - test_start

        print(f"✅ Test completed in {total_duration:.2f}s")

        return self.analyze_results(total_duration)

    def analyze_results(self, total_duration: float) -> dict:
        """Analyze test results and calculate metrics."""
        if not self.results:
            return {"error": "No results to analyze"}

        # Separate successful and failed requests
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        total_requests = len(self.results)
        success_count = len(successful)
        failure_count = len(failed)

        # Calculate latency percentiles (successful requests only)
        if successful:
            latencies = sorted([r.latency_ms for r in successful])
            p50 = latencies[int(len(latencies) * 0.50)]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            p50 = p95 = p99 = avg_latency = min_latency = max_latency = 0

        # Calculate throughput
        actual_qps = total_requests / total_duration
        successful_qps = success_count / total_duration

        # Error rate
        error_rate = (failure_count / total_requests * 100) if total_requests > 0 else 0

        # Token metrics
        if successful:
            avg_tokens = statistics.mean([r.tokens for r in successful])
            total_tokens = sum([r.tokens for r in successful])
            tokens_per_second = total_tokens / total_duration
        else:
            avg_tokens = total_tokens = tokens_per_second = 0

        # Error breakdown
        error_types: dict[str, int] = {}
        for r in failed:
            error_type = r.error.split(":")[0] if r.error else "Unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "model": self.model_id,
            "target_qps": self.qps,
            "duration_seconds": self.duration_seconds,
            "total_requests": total_requests,
            "successful_requests": success_count,
            "failed_requests": failure_count,
            "error_rate_percent": round(error_rate, 2),
            "actual_qps": round(actual_qps, 2),
            "successful_qps": round(successful_qps, 2),
            "latency_ms": {
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2),
                "avg": round(avg_latency, 2),
                "min": round(min_latency, 2),
                "max": round(max_latency, 2),
            },
            "tokens": {
                "avg_per_response": round(avg_tokens, 1),
                "total": total_tokens,
                "tokens_per_second": round(tokens_per_second, 2),
            },
            "error_breakdown": error_types,
        }


def print_results(analysis: dict):
    """Print formatted analysis results."""
    print(f"\n{'=' * 80}")
    print("STRESS TEST RESULTS")
    print(f"{'=' * 80}\n")

    print(f"Model: {analysis['model']}")
    print(f"Target QPS: {analysis['target_qps']}")
    print(f"Duration: {analysis['duration_seconds']}s")
    print(f"\n{'Request Stats':-^80}")
    print(f"Total Requests: {analysis['total_requests']}")
    print(
        f"Successful: {analysis['successful_requests']} ({100 - analysis['error_rate_percent']:.1f}%)"
    )
    print(f"Failed: {analysis['failed_requests']} ({analysis['error_rate_percent']:.1f}%)")
    print(f"Actual QPS: {analysis['actual_qps']:.2f}")
    print(f"Successful QPS: {analysis['successful_qps']:.2f}")

    print(f"\n{'Latency (ms)':-^80}")
    lat = analysis["latency_ms"]
    print(f"p50: {lat['p50']:.0f}ms  |  p95: {lat['p95']:.0f}ms  |  p99: {lat['p99']:.0f}ms")
    print(f"avg: {lat['avg']:.0f}ms  |  min: {lat['min']:.0f}ms  |  max: {lat['max']:.0f}ms")

    print(f"\n{'Token Throughput':-^80}")
    tok = analysis["tokens"]
    print(f"Avg tokens/response: {tok['avg_per_response']:.1f}")
    print(f"Total tokens: {tok['total']}")
    print(f"Tokens/second: {tok['tokens_per_second']:.2f}")

    if analysis["error_breakdown"]:
        print(f"\n{'Error Breakdown':-^80}")
        for error_type, count in analysis["error_breakdown"].items():
            print(f"{error_type}: {count}")

    # Performance assessment
    print(f"\n{'Assessment':-^80}")

    # Check if meeting SLA
    target_p95 = 12000 if "phi3" in analysis["model"] else 8000  # phi3: 12s, gemma: 8s
    meets_latency_sla = lat["p95"] <= target_p95
    meets_error_sla = analysis["error_rate_percent"] < 5.0

    if meets_latency_sla and meets_error_sla:
        print("✅ PASS - Meets production SLA")
    else:
        issues = []
        if not meets_latency_sla:
            issues.append(f"p95 latency {lat['p95']:.0f}ms exceeds target {target_p95}ms")
        if not meets_error_sla:
            issues.append(f"error rate {analysis['error_rate_percent']:.1f}% exceeds 5%")
        print(f"❌ FAIL - {', '.join(issues)}")


async def main():
    """Run stress tests."""
    print(f"\n{'=' * 80}")
    print("LATENCY STRESS TEST SUITE")
    print("=" * 80)

    # Test prompts (mix of short and medium requests)
    prompts = [
        "What is the capital of France?",
        "Explain what a REST API is in 2 sentences.",
        "List 3 benefits of cloud computing.",
        "How do you reverse a string in Python?",
        "What is the difference between HTTP and HTTPS?",
        "Explain recursion with a simple example.",
        "What are the SOLID principles?",
        "How does DNS work?",
        "What is a database index?",
        "Explain the concept of a microservice.",
    ]

    # Test configurations
    tests = [
        {
            "model": "gemma:2b",
            "qps": 2,  # Conservative for ultra-low latency model
            "duration": 60,
        },
        {
            "model": "phi3:3.8b",
            "qps": 1,  # Conservative for low-latency conversational model
            "duration": 60,
        },
    ]

    all_results = []

    for test_config in tests:
        runner = StressTestRunner(
            model_id=test_config["model"],
            qps=test_config["qps"],
            duration_seconds=test_config["duration"],
        )

        analysis = await runner.run_stress_test(prompts)
        print_results(analysis)
        all_results.append(analysis)

        # Pause between tests
        print("\n⏸️  Pausing 10s before next test...\n")
        await asyncio.sleep(10)

    # Comparison summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}\n")

    print(
        f"{'Model':<15} {'Target QPS':<12} {'Actual QPS':<12} {'p95 (ms)':<12} {'Error %':<10} {'Status'}"
    )
    print("-" * 80)

    for result in all_results:
        model = result["model"]
        target_qps = result["target_qps"]
        actual_qps = result["actual_qps"]
        p95 = result["latency_ms"]["p95"]
        error_rate = result["error_rate_percent"]

        # Determine status
        target_p95 = 12000 if "phi3" in model else 8000
        meets_sla = p95 <= target_p95 and error_rate < 5.0
        status = "✅ PASS" if meets_sla else "❌ FAIL"

        print(
            f"{model:<15} {target_qps:<12} {actual_qps:<12.2f} {p95:<12.0f} {error_rate:<10.1f} {status}"
        )

    # Save results
    output = {
        "test_date": "2025-12-01",
        "test_duration_per_model": tests[0]["duration"],
        "results": all_results,
    }

    with open("stress_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n📊 Results saved to: stress_test_results.json\n")


if __name__ == "__main__":
    asyncio.run(main())
