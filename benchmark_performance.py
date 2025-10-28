"""Benchmark script to measure performance improvements."""

import time
import sys
from datetime import datetime

# Suppress warnings
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from utils.email_handler import Email, MockEmailHandler
from workflows.support_workflow import SupportWorkflow


def benchmark_workflow(use_parallel=True, use_cache=True, num_emails=3):
    """
    Benchmark the workflow with different configurations.

    Args:
        use_parallel: Enable parallel processing
        use_cache: Enable response caching
        num_emails: Number of emails to test

    Returns:
        Dict with timing results
    """
    print("\n" + "=" * 80)
    print(f"BENCHMARKING: parallel={use_parallel}, cache={use_cache}")
    print("=" * 80)

    # Initialize workflow
    start_init = time.time()
    workflow = SupportWorkflow(
        max_revisions=0, use_parallel=use_parallel, use_cache=use_cache  # No retries for benchmark
    )
    init_time = time.time() - start_init
    print(f"Initialization time: {init_time:.2f}s\n")

    # Get test emails
    email_handler = MockEmailHandler("support@taskflowpro.com")
    emails = email_handler.check_new_emails()[:num_emails]

    print(f"Processing {len(emails)} emails...\n")

    # Process emails and measure time
    times = []

    for i, email in enumerate(emails, 1):
        print(f"\n{'─' * 80}")
        print(f"Email {i}/{len(emails)}: {email.subject}")
        print(f"{'─' * 80}")

        start = time.time()
        result = workflow.process_email(email)
        elapsed = time.time() - start

        times.append(elapsed)
        print(f"\n⏱️  Processing time: {elapsed:.2f}s")
        print(f"Status: {result['status']}")
        print(f"Category: {result['category']}")

        if i < len(emails):
            time.sleep(1)  # Brief pause between emails

    # Calculate statistics
    avg_time = sum(times) / len(times) if times else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0

    return {
        "init_time": init_time,
        "times": times,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "total_time": sum(times),
    }


def compare_configurations():
    """Compare different optimization configurations."""
    print("\n" + "🚀" * 40)
    print("PERFORMANCE OPTIMIZATION BENCHMARK")
    print("🚀" * 40)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    num_test_emails = 3

    # Configuration 1: All optimizations OFF
    print("\n\n📊 Configuration 1: BASELINE (No optimizations)")
    print("-" * 80)
    results_baseline = benchmark_workflow(
        use_parallel=False, use_cache=False, num_emails=num_test_emails
    )

    # Configuration 2: Only parallel processing
    print("\n\n📊 Configuration 2: PARALLEL PROCESSING ONLY")
    print("-" * 80)
    results_parallel = benchmark_workflow(
        use_parallel=True, use_cache=False, num_emails=num_test_emails
    )

    # Configuration 3: All optimizations ON
    print("\n\n📊 Configuration 3: FULL OPTIMIZATION (Parallel + Cache)")
    print("-" * 80)
    results_optimized = benchmark_workflow(
        use_parallel=True, use_cache=True, num_emails=num_test_emails
    )

    # Print comparison
    print("\n\n" + "=" * 80)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\n{'Configuration':<30} {'Avg Time':<15} {'Total Time':<15} {'Speedup'}")
    print("-" * 80)

    baseline_avg = results_baseline["avg_time"]

    print(
        f"{'Baseline (no optimization)':<30} {results_baseline['avg_time']:>10.2f}s    "
        f"{results_baseline['total_time']:>10.2f}s    1.00x"
    )

    parallel_speedup = (
        baseline_avg / results_parallel["avg_time"] if results_parallel["avg_time"] > 0 else 1.0
    )
    print(
        f"{'+ Parallel processing':<30} {results_parallel['avg_time']:>10.2f}s    "
        f"{results_parallel['total_time']:>10.2f}s    {parallel_speedup:.2f}x"
    )

    optimized_speedup = (
        baseline_avg / results_optimized["avg_time"] if results_optimized["avg_time"] > 0 else 1.0
    )
    print(
        f"{'+ Full optimization':<30} {results_optimized['avg_time']:>10.2f}s    "
        f"{results_optimized['total_time']:>10.2f}s    {optimized_speedup:.2f}x"
    )

    print("\n" + "=" * 80)
    print("✅ KEY FINDINGS")
    print("=" * 80)

    improvement = (
        ((baseline_avg - results_optimized["avg_time"]) / baseline_avg * 100)
        if baseline_avg > 0
        else 0
    )

    print(f"\n✓ Parallel processing speedup: {parallel_speedup:.2f}x faster")
    print(f"✓ Full optimization speedup: {optimized_speedup:.2f}x faster")
    print(f"✓ Overall improvement: {improvement:.1f}% faster")
    print(f"✓ Time saved per email: {baseline_avg - results_optimized['avg_time']:.2f}s")

    # Extrapolate to larger volumes
    emails_per_hour_baseline = 3600 / baseline_avg if baseline_avg > 0 else 0
    emails_per_hour_optimized = (
        3600 / results_optimized["avg_time"] if results_optimized["avg_time"] > 0 else 0
    )

    print(f"\n📊 THROUGHPUT ESTIMATES:")
    print(f"   Baseline: ~{emails_per_hour_baseline:.0f} emails/hour")
    print(f"   Optimized: ~{emails_per_hour_optimized:.0f} emails/hour")
    print(f"   Increase: +{emails_per_hour_optimized - emails_per_hour_baseline:.0f} emails/hour")

    print("\n💡 NOTES:")
    print("   • Cache hits would make subsequent similar emails <1s (instant)")
    print("   • GPU + quantization provides additional 2-3x speedup")
    print("   • Token optimization already included (128-400 tokens vs 512)")

    print("\n" + "=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        compare_configurations()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
