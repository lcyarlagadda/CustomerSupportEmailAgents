"""Utilities for parallel processing to speed up workflows."""

import concurrent.futures
from typing import Callable, List, Any, Dict
import time


class ParallelExecutor:
    """Helper class for running tasks in parallel."""

    def __init__(self, max_workers: int = 2):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum number of parallel workers (default: 2)
        """
        self.max_workers = max_workers

    def run_parallel(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of task dictionaries with 'func' and optional 'args', 'kwargs'
                Example: [
                    {'func': my_function, 'args': (arg1, arg2), 'kwargs': {'key': 'value'}},
                    {'func': another_function, 'args': (arg3,)}
                ]

        Returns:
            List of results in the same order as tasks
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for task in tasks:
                func = task["func"]
                args = task.get("args", ())
                kwargs = task.get("kwargs", {})

                future = executor.submit(func, *args, **kwargs)
                futures.append(future)

            # Wait for all tasks to complete
            results = [future.result() for future in futures]

            return results

    def run_parallel_with_timeout(
        self, tasks: List[Dict[str, Any]], timeout: float = 30.0
    ) -> List[Any]:
        """
        Execute tasks in parallel with a timeout.

        Args:
            tasks: List of task dictionaries
            timeout: Maximum time to wait for all tasks (seconds)

        Returns:
            List of results (None for tasks that timeout)
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for task in tasks:
                func = task["func"]
                args = task.get("args", ())
                kwargs = task.get("kwargs", {})

                future = executor.submit(func, *args, **kwargs)
                futures.append(future)

            # Wait with timeout
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    print(f"Task timed out after {timeout}s")
                    results.append(None)

            return results


def benchmark_speedup(sequential_func: Callable, parallel_func: Callable, *args, **kwargs):
    """
    Benchmark the speedup from parallel execution.

    Args:
        sequential_func: Function that runs sequentially
        parallel_func: Function that runs in parallel
        *args, **kwargs: Arguments to pass to both functions

    Returns:
        Dict with timing results and speedup factor
    """
    # Time sequential execution
    start_seq = time.time()
    sequential_func(*args, **kwargs)
    seq_time = time.time() - start_seq

    # Time parallel execution
    start_par = time.time()
    parallel_func(*args, **kwargs)
    par_time = time.time() - start_par

    speedup = seq_time / par_time if par_time > 0 else 1.0

    return {
        "sequential_time": seq_time,
        "parallel_time": par_time,
        "speedup": speedup,
        "improvement_percent": (1 - par_time / seq_time) * 100 if seq_time > 0 else 0,
    }