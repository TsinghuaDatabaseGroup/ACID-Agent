"""
Test suite for per-task caching functionality.

Tests the following scenarios:
1. Scenario 1: Run workload from scratch (no cache), verify single central cache at the end, 
   then rerun with cache - no computation should occur.
2. Scenario 2: Run workload with multiple workers, simulate one worker crash,
   then rerun with cache - only missing tasks should be recomputed.
"""
import json
import os
import pytest
import shutil
import time
from unittest.mock import patch, MagicMock

from benchmark import benchmark as benchmark_module
from benchmark.benchmark import Benchmark
from benchmark.evaluator import Evaluator
from benchmark.executor import Executor
from benchmark.benchmark_utils import merge_task_caches, get_most_recent_cache
from systems import DummySystem


@pytest.fixture
def workload_name():
    """Workload to use for testing."""
    return "biomedical"


@pytest.fixture
def run_subtasks():
    """Whether to run subtasks during tests."""
    return True


class PerTaskCacheTestBase:
    """Base class for per-task caching tests."""
    
    def setup_method(self):
        """Create results directory and ensure cache is clean before each test."""
        self.results_dir = "results_per_task_caching"
        os.makedirs(self.results_dir, exist_ok=True)
        cache_dir = os.path.join(self.results_dir, "response_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    
    # def teardown_method(self):
        # """Clean up after each test."""
        # if os.path.exists(self.results_dir):
            # shutil.rmtree(self.results_dir)
    
    def get_cache_files(self, basename: str) -> dict:
        """Get all cache files in the cache directory, categorized by type.
        
        Returns:
            Dict with keys:
            - 'central': list of central cache files (single timestamp)
            - 'task': list of task cache files (contain _task_)
            - 'all': all cache files
        """
        cache_dir = os.path.join(self.results_dir, "response_cache")
        if not os.path.exists(cache_dir):
            return {'central': [], 'task': [], 'all': []}
        
        all_files = [f for f in os.listdir(cache_dir) if f.startswith(basename)]
        central_files = [f for f in all_files if "_task_" not in f and f.endswith('.json')]
        task_files = [f for f in all_files if "_task_" in f]
        
        return {
            'central': central_files,
            'task': task_files,
            'all': all_files
        }
    
    def count_cache_entries(self, filename: str) -> int:
        """Count the number of task results in a cache file."""
        cache_dir = os.path.join(self.results_dir, "response_cache")
        cache_path = os.path.join(cache_dir, filename)
        
        if not os.path.exists(cache_path):
            return 0
        
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        return len(data) if isinstance(data, list) else 1


class TestScenario1SingleRun(PerTaskCacheTestBase):
    """
    Scenario 1: Run workload from scratch, verify single central cache,
    then rerun with cache - no computation.
    """
    
    def test_scenario_1_initial_run_creates_central_cache(self, workload_name, run_subtasks):
        """Test that first run creates per-task caches that are merged into a single central cache."""
        dataset_dir = f"data/{workload_name}/input"
        
        # Load workload to verify test data
        with open(f"workload/{workload_name}.json", 'r') as f:
            workload = json.load(f)
        num_tasks = len(workload)
        
        # Calculate expected number of cache entries (including subtasks if enabled)
        def count_all_tasks(tasks):
            """Recursively count all tasks including subtasks."""
            count = len(tasks)
            for task in tasks:
                if "subtasks" in task and run_subtasks:
                    count += count_all_tasks(task["subtasks"])
            return count
        
        expected_cache_entries = count_all_tasks(workload)
        
        # Create benchmark and run
        benchmark = Benchmark(
            system_name="DummySystem",
            task_fixture_directory="benchmark/fixtures",
            system_output_directory=os.path.join(self.results_dir, "system_output"),
            use_system_cache=False,  # No pre-existing cache
            cache_system_output=True,  # Enable caching
            verbose=True,
            run_subtasks=run_subtasks,
            evaluate_pipeline=False,
            num_workers=1
        )
        
        # Run benchmark
        results = benchmark.run_benchmark(
            dataset_directory=dataset_dir,
            results_directory=self.results_dir,
            workload_path=f"workload/{workload_name}.json"
        )
        
        # Check cache structure after first run
        cache_files = self.get_cache_files(workload_name)
        
        # After merge, should have no task files (they're cleaned up) and 1 central cache
        assert len(cache_files['task']) == 0, f"Task cache files should be cleaned up, but found: {cache_files['task']}"
        assert len(cache_files['central']) == 1, f"Should have exactly 1 central cache file, but found: {cache_files['central']}"
        
        central_cache = cache_files['central'][0]
        num_cached = self.count_cache_entries(central_cache)
        assert num_cached == expected_cache_entries, f"Central cache should contain {expected_cache_entries} results (tasks + subtasks), but contains {num_cached}"
        
        print(f"✓ Initial run successful: created central cache '{central_cache}' with {num_cached} entries")
    
    def test_scenario_1_rerun_with_cache_no_computation(self, workload_name, run_subtasks):
        """Test that rerun with cache doesn't perform any computation."""
        dataset_dir = f"data/{workload_name}/input"
        
        # Load workload
        with open(f"workload/{workload_name}.json", 'r') as f:
            workload = json.load(f)
        num_tasks = len(workload)
        
        # First run - create cache
        benchmark1 = Benchmark(
            system_name="DummySystem",
            task_fixture_directory="benchmark/fixtures",
            system_output_directory=os.path.join(self.results_dir, "system_output"),
            use_system_cache=False,
            cache_system_output=True,
            verbose=False,
            run_subtasks=run_subtasks,
            evaluate_pipeline=False,
            num_workers=1
        )
        benchmark1.run_benchmark(
            dataset_directory=dataset_dir,
            results_directory=self.results_dir,
            workload_path=f"workload/{workload_name}.json"
        )
        
        # Get timestamp of first central cache
        cache_files = self.get_cache_files(workload_name)
        first_central_cache = cache_files['central'][0]
        first_cache_time = os.path.getmtime(os.path.join(self.results_dir, "response_cache", first_central_cache))
        
        # Wait a bit to ensure timestamps differ
        time.sleep(0.1)
        
        # Second run - with cache enabled, should not create new cache files
        benchmark2 = Benchmark(
            system_name="DummySystem",
            task_fixture_directory="benchmark/fixtures",
            system_output_directory=os.path.join(self.results_dir, "system_output2"),
            use_system_cache=True,  # Use cache this time
            cache_system_output=True,
            verbose=False,
            run_subtasks=run_subtasks,
            evaluate_pipeline=False,
            num_workers=1
        )
        
        # Mock serve_query to track if it's called
        with patch.object(DummySystem, 'serve_query', wraps=DummySystem.serve_query) as mock_serve:
            results = benchmark2.run_benchmark(
                dataset_directory=dataset_dir,
                results_directory=self.results_dir,
                workload_path=f"workload/{workload_name}.json"
            )
            
            # Check that serve_query was never called (all results from cache)
            assert mock_serve.call_count == 0, f"serve_query should not be called when using cache, but was called {mock_serve.call_count} times"
        
        # Verify results are still correct
        assert len(results) == num_tasks
        
        print(f"✓ Rerun with cache successful: no computation performed, got {len(results)} results from cache")


class TestScenario2PartialCrash(PerTaskCacheTestBase):
    """
    Scenario 2: Run with multiple workers, simulate one worker crash,
    then rerun with cache - only missing tasks should be computed.
    """
    
    def test_scenario_2_worker_crash_and_recovery(self, workload_name, run_subtasks, monkeypatch):
        """Test that after worker crash, only missing tasks are recomputed."""
        dataset_dir = f"data/{workload_name}/input"
        num_workers = 4
        
        # Load workload
        with open(f"workload/{workload_name}.json", 'r') as f:
            workload = json.load(f)
        
        # Calculate which tasks would be assigned to each worker (round-robin)
        # Match Benchmark.run_benchmark task ordering
        task_ids = [task["id"] for task in workload]
        task_ids = sorted(task_ids, key=lambda x: int(x.split('-')[-1]))
        partitions = [[] for _ in range(num_workers)]
        for idx, tid in enumerate(task_ids):
            partitions[idx % num_workers].append(tid)
        partitions = [p for p in partitions if len(p) > 0]  # Remove empty partitions
        
        # Worker that will "crash" - let's say worker 0 crashes after executing half its tasks
        crashing_worker = 3
        crashed_task_ids = set(partitions[crashing_worker][:1])
        missing_task_ids = set(partitions[crashing_worker][1:])
        
        print()
        print(f"  Tasks that will not be executed: {missing_task_ids}")
        
        # First run - simulate crash
        benchmark1 = Benchmark(
            system_name="DummySystem",
            task_fixture_directory="benchmark/fixtures",
            system_output_directory=os.path.join(self.results_dir, "system_output"),
            use_system_cache=False,
            cache_system_output=True,
            verbose=False,
            run_subtasks=run_subtasks,
            evaluate_pipeline=False,
            num_workers=num_workers
        )

        monkeypatch.setattr(Evaluator, "evaluate_results", lambda self, responses: responses)
        
        class InlineExecutor:
            def __init__(self, max_workers=None):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, *iterables):
                for args in zip(*iterables):
                    yield func(*args)

            def submit(self, fn, *args, **kwargs):
                class Future:
                    def result(self_inner):
                        return fn(*args, **kwargs)
                return Future()

        # Override executor partition to simulate crash without multiprocessing
        original_run_executor = Benchmark._run_executor_partition

        def run_executor_with_crash(
            idx,
            partition,
            system,
            system_name,
            workload,
            results_directory,
            verbose,
            run_subtasks,
            use_deepresearch_subset,
            use_truth_subset,
            system_output_dir,
            cache_system_output,
            num_partitions,
        ):
            if idx == crashing_worker:
                crashed_workload = [t for t in workload if t['id'] in crashed_task_ids]
                executor = Executor(
                    system=system,
                    system_name=system_name,
                    workload=crashed_workload,
                    results_directory=results_directory,
                    worker_id=idx,
                    verbose=verbose,
                    run_subtasks=run_subtasks,
                    use_deepresearch_subset=use_deepresearch_subset,
                    use_truth_subset=use_truth_subset,
                    system_output_dir=system_output_dir,
                )
                return executor.run_workload(cache_system_output=cache_system_output)

            return original_run_executor(
                idx,
                partition,
                system,
                system_name,
                workload,
                results_directory,
                verbose,
                run_subtasks,
                use_deepresearch_subset,
                use_truth_subset,
                system_output_dir,
                cache_system_output,
                num_partitions,
            )

        monkeypatch.setattr(benchmark_module, "ProcessPoolExecutor", InlineExecutor)
        monkeypatch.setattr(Benchmark, "_run_executor_partition", staticmethod(run_executor_with_crash))

        results1 = benchmark1.run_benchmark(
            dataset_directory=dataset_dir,
            results_directory=self.results_dir,
            workload_path=f"workload/{workload_name}.json"
        )
        
        # Check cache after crash
        cache_dir = os.path.join(self.results_dir, "response_cache")
        cached_files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
        
        # We should have per-task caches for the crashed tasks and a central cache
        task_cache_count = len([f for f in cached_files if "_task_" in f])
        central_count = len([f for f in cached_files if "_task_" not in f and f.endswith('.json')])
        
        print(f"✓ After crash: {task_cache_count} per-task cache files, {central_count} central cache files")

        # Determine missing tasks based on actual cache contents
        cache_files = self.get_cache_files(workload_name)
        central_cache = cache_files['central'][0]
        cache_dir = os.path.join(self.results_dir, "response_cache")
        with open(os.path.join(cache_dir, central_cache), 'r') as f:
            cached_results = json.load(f)
        cached_task_ids = {r.get('task_id') for r in cached_results if r.get('task_id')}
        missing_top_level_ids = set(task_ids) - cached_task_ids

        def collect_subtask_ids(tasks):
            ids = []
            for task in tasks:
                if "subtasks" in task:
                    for subtask in task["subtasks"]:
                        ids.append(subtask["id"])
                        ids.extend(collect_subtask_ids([subtask]))
            return ids

        subtask_ids_by_parent = {}
        for task in workload:
            subtask_ids_by_parent[task["id"]] = collect_subtask_ids([task])

        missing_task_ids = set(missing_top_level_ids)
        if run_subtasks:
            for task_id in missing_top_level_ids:
                missing_task_ids.update(subtask_ids_by_parent.get(task_id, []))
        
        print(f"  Missing tasks determined from cache: {missing_task_ids}")
        # Second run - with cache, should only compute missing tasks
        benchmark2 = Benchmark(
            system_name="DummySystem",
            task_fixture_directory="benchmark/fixtures",
            system_output_directory=os.path.join(self.results_dir, "system_output2"),
            use_system_cache=True,  # Use cache
            cache_system_output=True,
            verbose=False,
            run_subtasks=run_subtasks,
            evaluate_pipeline=False,
            num_workers=num_workers
        )
        
        # Track which tasks are computed in second run
        computed_query_ids = []
        original_serve_query = DummySystem.serve_query
        
        def mock_serve_query_track(self, query, query_id="default_name-0", subset_files=None):
            computed_query_ids.append(query_id)
            return original_serve_query(self, query, query_id, subset_files)
        
        with patch.object(DummySystem, 'serve_query', mock_serve_query_track):
            results2 = benchmark2.run_benchmark(
                dataset_directory=dataset_dir,
                results_directory=self.results_dir,
                workload_path=f"workload/{workload_name}.json"
            )
        
        # Verify that only missing tasks were computed (include subtasks)
        computed_task_ids = set(computed_query_ids)
        assert computed_task_ids == missing_task_ids, \
            f"Should only compute missing tasks {missing_task_ids}, but computed {computed_task_ids}"
        
        # Verify final results contain all tasks
        result_ids = {r['task_id'] for r in results2}
        assert result_ids == set([t['id'] for t in workload]), \
            f"Final results should contain all tasks, but only contain {result_ids}"
        
        print(f"✓ Recovery successful: only {len(missing_task_ids)} missing tasks were recomputed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
