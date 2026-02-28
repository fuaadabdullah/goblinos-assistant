# pyright: reportMissingImports=false
"""
Test script for local LLM intelligent routing.
Demonstrates routing decisions for various use cases.
"""

import sys
from pathlib import Path

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from services.local_llm_routing import (
    select_model,
    get_system_prompt,
    get_routing_explanation,
    detect_intent,
    get_context_length,
    Intent,
    LatencyTarget,
)


def print_routing_decision(
    test_name: str,
    messages: list,
    intent: Intent = None,
    latency_target: LatencyTarget = LatencyTarget.MEDIUM,
    context_provided: str = None,
    cost_priority: bool = False,
):
    """Print detailed routing decision."""
    print("\n" + "=" * 80)
    print(f"Test Case: {test_name}")
    print("=" * 80)

    # Show input
    print("\nInput:")
    print(f"  Messages: {len(messages)} message(s)")
    for i, msg in enumerate(messages, 1):
        content = msg.get("content", "")
        preview = content[:60] + "..." if len(content) > 60 else content
        print(f"    {i}. [{msg.get('role', 'user')}] {preview}")

    if intent:
        print(f"  Explicit Intent: {intent.value}")
    print(f"  Latency Target: {latency_target.value}")
    if context_provided:
        print(f"  Context Provided: {len(context_provided)} characters")
    if cost_priority:
        print("  Cost Priority: Yes")

    # Get routing decision
    model_id, params = select_model(
        messages=messages,
        intent=intent,
        latency_target=latency_target,
        context_provided=context_provided,
        cost_priority=cost_priority,
    )

    # Get additional info
    detected_intent = intent or detect_intent(messages)
    context_length = get_context_length(messages)
    if context_provided:
        from services.local_llm_routing import estimate_token_count

        context_length += estimate_token_count(context_provided)

    system_prompt = get_system_prompt(detected_intent)
    explanation = get_routing_explanation(model_id, detected_intent, context_length, latency_target)

    # Show routing decision
    print("\n" + "-" * 80)
    print("Routing Decision:")
    print(f"  Selected Model: {model_id}")
    print(f"  Detected Intent: {detected_intent.value}")
    print(f"  Context Length: {context_length} tokens")
    print(f"  Reasoning: {explanation}")

    print("\nRecommended Parameters:")
    for key, value in params.items():
        if value is not None:
            print(f"  {key}: {value}")

    print("\nSystem Prompt:")
    print(f"  {system_prompt[:120]}...")

    print("\n" + "=" * 80)


def main():
    """Run routing tests."""
    print("\n" + "=" * 80)
    print("Local LLM Intelligent Routing Test Suite")
    print("=" * 80)
    print("\nThis test demonstrates how requests are routed to different models")
    print("based on intent, context length, latency requirements, and cost priority.")

    # Test 1: Code generation (high quality needed)
    print_routing_decision(
        test_name="Code Generation - High Quality",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function to implement binary search with error handling and type hints.",
            }
        ],
    )

    # Test 2: Quick status check (ultra-low latency)
    print_routing_decision(
        test_name="Status Check - Ultra Fast",
        messages=[{"role": "user", "content": "What's the status of the deployment?"}],
        latency_target=LatencyTarget.ULTRA_LOW,
    )

    # Test 3: Long document summarization (long context)
    long_doc = "x" * 40000  # ~10k tokens
    print_routing_decision(
        test_name="Long Document Summarization",
        messages=[
            {
                "role": "user",
                "content": "Summarize the key points from this research paper.",
            }
        ],
        context_provided=long_doc,
    )

    # Test 4: Multilingual translation
    print_routing_decision(
        test_name="Multilingual Translation",
        messages=[
            {
                "role": "user",
                "content": "请将这段文字翻译成英文：人工智能正在改变世界。",
            }
        ],
    )

    # Test 5: Creative writing
    print_routing_decision(
        test_name="Creative Writing",
        messages=[
            {
                "role": "user",
                "content": "Write a short, creative story about a robot learning to paint.",
            }
        ],
        intent=Intent.CREATIVE,
    )

    # Test 6: Classification task (ultra fast)
    print_routing_decision(
        test_name="Text Classification - Cost Optimized",
        messages=[
            {
                "role": "user",
                "content": "Classify this email as spam or not spam: 'Congratulations! You won $1M!'",
            }
        ],
        cost_priority=True,
    )

    # Test 7: RAG query with context
    rag_context = """
    Azure Cosmos DB is a fully managed NoSQL and relational database for modern app development.
    It offers single-digit millisecond response times, automatic and instant scalability,
    along with guarantee speed at any scale.
    """
    print_routing_decision(
        test_name="RAG Query with Context",
        messages=[
            {
                "role": "user",
                "content": "Based on the provided documentation, what are the key features of Cosmos DB?",
            }
        ],
        intent=Intent.RAG,
        context_provided=rag_context,
    )

    # Test 8: Conversational chat (low latency)
    print_routing_decision(
        test_name="Conversational Chat - Low Latency",
        messages=[
            {"role": "user", "content": "Hi! How are you?"},
            {
                "role": "assistant",
                "content": "I'm doing well! How can I help you today?",
            },
            {"role": "user", "content": "Can you recommend a good book?"},
        ],
        latency_target=LatencyTarget.LOW,
    )

    # Test 9: Detailed explanation (high quality)
    print_routing_decision(
        test_name="Detailed Explanation - High Quality",
        messages=[
            {
                "role": "user",
                "content": "Explain how transformers work in deep learning, including attention mechanisms.",
            }
        ],
        intent=Intent.EXPLAIN,
    )

    # Test 10: Multi-turn technical support
    print_routing_decision(
        test_name="Technical Support - Balanced",
        messages=[
            {"role": "user", "content": "My API is returning 429 errors"},
            {
                "role": "assistant",
                "content": "That's a rate limit error. How many requests are you making?",
            },
            {"role": "user", "content": "About 100 per second"},
            {
                "role": "assistant",
                "content": "That's quite high. What's your rate limit tier?",
            },
            {"role": "user", "content": "I'm on the free tier"},
        ],
    )

    print("\n" + "=" * 80)
    print("Test Suite Complete")
    print("=" * 80)
    print("\nRouting Summary:")
    print("  • mistral:7b   → High quality, creative, coding, explanations")
    print("  • qwen2.5:3b   → Long context, multilingual, RAG")
    print("  • phi3:3.8b    → Low latency chat, conversational")
    print("  • gemma:2b     → Ultra-fast, classification, status checks")
    print("\n")


if __name__ == "__main__":
    main()
