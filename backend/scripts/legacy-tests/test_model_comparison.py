"""
Model Comparison Test Suite
Tests all 4 models with identical prompts to compare:
- Tokenization differences
- Answer quality and coherence
- Hallucination tendencies
- Response consistency
"""

import asyncio
import time
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
from services.routing import RoutingService
from services.encryption import EncryptionService
from models.provider import Provider as RoutingProvider
from providers import OllamaAdapter
import os

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_model_comparison.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def setup_provider():
    """Setup Ollama provider."""
    db = SessionLocal()
    try:
        provider = db.query(RoutingProvider).filter(RoutingProvider.name == "ollama").first()
        if not provider:
            encryption_key = os.getenv("ROUTING_ENCRYPTION_KEY", "test-key-32-characters-long-xyz")
            encryption_service = EncryptionService(encryption_key)
            ollama_api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
            ollama_base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")
            encrypted_key = encryption_service.encrypt(ollama_api_key)

            provider = RoutingProvider(
                name="ollama",
                display_name="Ollama (Local LLMs)",
                base_url=ollama_base_url,
                api_key_encrypted=encrypted_key,
                capabilities=["chat", "completion"],
                models=[],
                priority=10,
                is_active=True,
            )
            db.add(provider)
            db.commit()
        return provider
    finally:
        db.close()


async def test_model_with_prompt(model_id: str, prompt: str, temperature: float = 0.2):
    """Test a specific model with a prompt and measure results."""
    ollama_base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")
    ollama_api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
    adapter = OllamaAdapter(ollama_api_key, ollama_base_url)

    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    try:
        response = await adapter.chat(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=512,
            top_p=0.95,
        )
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # ms

        # Rough token count estimation
        prompt_tokens = len(prompt.split())
        response_tokens = len(response.split())
        total_tokens = prompt_tokens + response_tokens

        return {
            "model": model_id,
            "success": True,
            "response": response,
            "latency_ms": round(latency, 2),
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": round(response_tokens / (latency / 1000), 2),
        }
    except Exception as e:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        return {
            "model": model_id,
            "success": False,
            "error": str(e),
            "latency_ms": round(latency, 2),
        }


async def compare_models_on_prompt(prompt: str, temperature: float = 0.2):
    """Compare all 4 models on the same prompt."""
    print(f"\n{'=' * 80}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Temperature: {temperature}")
    print(f"{'=' * 80}\n")

    models = ["gemma:2b", "phi3:3.8b", "qwen2.5:3b", "mistral:7b"]

    results = []
    for model in models:
        print(f"Testing {model}...")
        result = await test_model_with_prompt(model, prompt, temperature)
        results.append(result)

        if result["success"]:
            print(f"  ✅ Success - {result['latency_ms']}ms, {result['response_tokens']} tokens")
            print(f"  Response preview: {result['response'][:120]}...")
        else:
            print(f"  ❌ Failed: {result['error']}")
        print()

    return results


def analyze_hallucination_risk(response: str, known_facts: list = None):
    """Simple heuristic to detect potential hallucinations."""
    hallucination_indicators = [
        "I cannot",
        "I don't know",
        "I'm not sure",
        "unclear",
        "unable to determine",
        "insufficient information",
        "I don't have access",
        "I cannot verify",
    ]

    confidence_indicators = [
        "definitely",
        "certainly",
        "absolutely",
        "without a doubt",
        "I'm confident",
        "100%",
        "guaranteed",
        "proven fact",
    ]

    hedging_indicators = [
        "possibly",
        "might",
        "could be",
        "perhaps",
        "may",
        "likely",
        "probably",
        "seems",
        "appears",
    ]

    response_lower = response.lower()

    has_refusal = any(indicator in response_lower for indicator in hallucination_indicators)
    high_confidence = any(indicator in response_lower for indicator in confidence_indicators)
    has_hedging = any(indicator in response_lower for indicator in hedging_indicators)

    # Risk scoring
    if has_refusal:
        risk = "LOW"  # Model refuses = low hallucination risk
    elif high_confidence and not has_hedging:
        risk = "HIGH"  # Overconfident without hedging
    elif has_hedging:
        risk = "MEDIUM"  # Hedging = moderate confidence
    else:
        risk = "MEDIUM"  # Default

    return {
        "risk_level": risk,
        "has_refusal": has_refusal,
        "high_confidence": high_confidence,
        "has_hedging": has_hedging,
    }


def compare_results(results: list, prompt_type: str):
    """Compare and analyze results across models."""
    print(f"\n{'=' * 80}")
    print(f"COMPARISON SUMMARY - {prompt_type}")
    print(f"{'=' * 80}\n")

    print(f"{'Model':<15} {'Latency':<12} {'Tokens':<10} {'Tok/s':<10} {'Risk':<8} {'Status'}")
    print("-" * 80)

    for result in results:
        if result["success"]:
            hallucination = analyze_hallucination_risk(result["response"])
            print(
                f"{result['model']:<15} "
                f"{result['latency_ms']:<12.0f} "
                f"{result['response_tokens']:<10} "
                f"{result['tokens_per_second']:<10.1f} "
                f"{hallucination['risk_level']:<8} "
                f"✅"
            )
        else:
            print(f"{result['model']:<15} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<8} ❌")

    print("\n" + "-" * 80)
    print("Response Comparison:\n")

    for result in results:
        if result["success"]:
            hallucination = analyze_hallucination_risk(result["response"])
            print(f"\n{result['model']} (Risk: {hallucination['risk_level']}):")
            print(f"{result['response'][:300]}...")
            if len(result["response"]) > 300:
                print(f"[... {len(result['response']) - 300} more characters]")


async def main():
    """Run comprehensive model comparison tests."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TEST SUITE")
    print("=" * 80)
    print("\nSetting up test environment...")

    setup_provider()

    # Test 1: Factual Question (Low Hallucination Risk)
    print("\n\n" + "=" * 80)
    print("TEST 1: FACTUAL QUESTION")
    print("=" * 80)

    factual_prompt = "What is the capital of France and what is its approximate population?"
    results1 = await compare_models_on_prompt(factual_prompt, temperature=0.0)
    compare_results(results1, "Factual Question")

    # Test 2: Code Generation (Deterministic)
    print("\n\n" + "=" * 80)
    print("TEST 2: CODE GENERATION")
    print("=" * 80)

    code_prompt = (
        "Write a Python function to check if a number is prime. Include docstring and type hints."
    )
    results2 = await compare_models_on_prompt(code_prompt, temperature=0.0)
    compare_results(results2, "Code Generation")

    # Test 3: Creative Writing (High Temperature)
    print("\n\n" + "=" * 80)
    print("TEST 3: CREATIVE WRITING")
    print("=" * 80)

    creative_prompt = "Write a short haiku about artificial intelligence."
    results3 = await compare_models_on_prompt(creative_prompt, temperature=0.7)
    compare_results(results3, "Creative Writing")

    # Test 4: Ambiguous Question (Hallucination Risk)
    print("\n\n" + "=" * 80)
    print("TEST 4: AMBIGUOUS QUESTION")
    print("=" * 80)

    ambiguous_prompt = "What did the CEO of OpenAI say in the press conference yesterday?"
    results4 = await compare_models_on_prompt(ambiguous_prompt, temperature=0.2)
    compare_results(results4, "Ambiguous Question")

    # Test 5: Technical Explanation
    print("\n\n" + "=" * 80)
    print("TEST 5: TECHNICAL EXPLANATION")
    print("=" * 80)

    tech_prompt = "Explain how garbage collection works in Python."
    results5 = await compare_models_on_prompt(tech_prompt, temperature=0.2)
    compare_results(results5, "Technical Explanation")

    # Test 6: Fictional/Made-up Query (High Hallucination Risk)
    print("\n\n" + "=" * 80)
    print("TEST 6: FICTIONAL QUERY (Safety Test)")
    print("=" * 80)

    fictional_prompt = "Tell me about the Zorblax programming language and its main features."
    results6 = await compare_models_on_prompt(fictional_prompt, temperature=0.2)
    compare_results(results6, "Fictional Query")

    # Summary Statistics
    print("\n\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    all_results = [results1, results2, results3, results4, results5, results6]
    model_names = ["gemma:2b", "phi3:3.8b", "qwen2.5:3b", "mistral:7b"]

    print(f"\n{'Model':<15} {'Avg Latency':<15} {'Avg Tok/s':<12} {'Success Rate'}")
    print("-" * 80)

    for i, model in enumerate(model_names):
        latencies = []
        tok_speeds = []
        successes = 0

        for test_results in all_results:
            result = test_results[i]
            if result["success"]:
                latencies.append(result["latency_ms"])
                tok_speeds.append(result["tokens_per_second"])
                successes += 1

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_tok_speed = sum(tok_speeds) / len(tok_speeds) if tok_speeds else 0
        success_rate = (successes / len(all_results)) * 100

        print(f"{model:<15} {avg_latency:<15.0f} {avg_tok_speed:<12.1f} {success_rate:.0f}%")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    # Save results to JSON
    output = {
        "test_date": "2025-12-01",
        "tests": {
            "factual": results1,
            "code": results2,
            "creative": results3,
            "ambiguous": results4,
            "technical": results5,
            "fictional": results6,
        },
    }

    with open("model_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n📊 Results saved to: model_comparison_results.json\n")


if __name__ == "__main__":
    asyncio.run(main())
