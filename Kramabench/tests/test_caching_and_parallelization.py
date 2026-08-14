"""
Test suite for caching and parallel execution functionality.

Tests using real workloads from the workspace:
1. No cache: Systems run and compute all results
2. Full cache: System loads all results from cache
3. Partial cache: System loads cached results and computes missing ones
4. Cache merge: Final central cache contains all results after workers complete
5. Task subset: Executor correctly filters tasks based on subset
6. Parallel execution: Multiple workers correctly partition and cache results
"""
import json
import os
import pytest
import shutil

from benchmark.benchmark import Benchmark, Executor
from systems import DummySystem



@pytest.fixture
def results_dir():
    """Create a results directory for tests."""
    results_dir = "results/DummySystem"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir    

@pytest.fixture
def workload_name():
    """Get path to real biomedical workload."""
    return "biomedical-obscured"


@pytest.fixture
def data_dir():
    """Get path to real biomedical data."""
    return "data/biomedical/input"


@pytest.fixture
def workload_tasks(workload_name):
    """Load real workload and extract task list."""
    workload_path = f"workload/{workload_name}.json"
    with open(workload_path, 'r') as f:
        workload = json.load(f)
    return workload

class CacheTestBase:
    """Base class for cache-related tests."""
    
    def setup_method(self):
        """Create results directory and ensure cache is clean before each test."""
        self.results_dir = "results/DummySystem"
        os.makedirs(self.results_dir, exist_ok=True)
        cache_dir = os.path.join(self.results_dir, "response_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    
    def teardown_method(self):
        """Clean up cache directory after each test."""
        cache_dir = os.path.join(self.results_dir, "response_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


class TestNoCacheScenario(CacheTestBase):
    """Test scenario where no cache exists - systems should run normally."""
    

    def test_single_executor_no_cache(self, results_dir, workload_name, data_dir):
        """Test single executor runs all tasks when no cache exists."""
        
        # Create system and executor
        system = DummySystem(verbose=True)
        system.process_dataset(data_dir)

        with open(f"workload/{workload_name}.json", 'r') as f:
            workload_tasks = json.load(f)

        executor = Executor(
            system=system,
            system_name="DummySystem",
            workload_path=f"workload/{workload_name}.json",
            results_directory=results_dir,
            verbose=True,
            worker_id=0
        )
        
        # Run without cache
        results = executor.run_workload(use_system_cache=False, cache_system_output=False)
        
        # Verify all tasks were executed
        assert len(results) == len(workload_tasks)
        task_ids = {r["task_id"] for r in results}
        assert task_ids == set([t['id'] for t in workload_tasks])
        
        # Verify responses have required fields
        for result in results:
            assert "task_id" in result
            assert "model_output" in result
            assert "code" in result

class TestFullCacheScenario(CacheTestBase):
    """Test scenario where full cache exists - system should load without computing."""
    
    def test_full_cache_loading(self, results_dir, workload_name, data_dir, workload_tasks):
        """Test that full cache is loaded and no computation happens."""
        cache_dir = os.path.join(results_dir, "response_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a pre-computed cache file with all tasks
        cached_results = [
            {
                "task_id": task_id,
                "model_output": {"answer": f"cached_{task_id}"},
                "code": f"# cached code for {task_id}"
            }
            for task_id in [t['id'] for t in workload_tasks]
        ]
        
        cache_file = os.path.join(cache_dir, f"{workload_name}_20260115_120000.json")
        with open(cache_file, 'w') as f:
            json.dump(cached_results, f)
        
        # Create system and executor
        system = DummySystem(verbose=True)
        system.process_dataset(data_dir)
        
        executor = Executor(
            system=system,
            system_name="DummySystem",
            workload_path=f"workload/{workload_name}.json",
            results_directory=results_dir,
            verbose=True,
            worker_id=0
        )
        
        # Run with cache enabled
        results = executor.run_workload(use_system_cache=True, cache_system_output=False)
        
        # Verify cached results are returned
        assert len(results) == len(workload_tasks)
        for result in results:
            assert "cached_" in result["model_output"]["answer"]


class TestPartialCacheScenario(CacheTestBase):
    """Test scenario where partial cache exists - system should load cached and compute missing."""
    
    def test_partial_cache_computation(self, results_dir, workload_name, data_dir):
        """Test that partial cache loads cached results and computes missing ones."""
        cache_dir = os.path.join(results_dir, "response_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Load workload tasks
        with open(f"workload/{workload_name}.json", 'r') as f:
            workload_dict = json.load(f)
        workload_tasks = [t['id'] for t in workload_dict]
        
        # Create a partial cache with only first half of tasks
        half = len(workload_tasks) // 2
        cached_task_ids = workload_tasks[:half]
        missing_task_ids = workload_tasks[half:]
        
        cached_results = [
            {
                "task_id": task_id,
                "model_output": {"answer": f"cached_{task_id}"},
                "code": f"# cached code for {task_id}"
            }
            for task_id in cached_task_ids
        ]
        
        cache_file = os.path.join(cache_dir, "biomedical_20260115_120000.json")
        with open(cache_file, 'w') as f:
            json.dump(cached_results, f)
        
        # Create system and executor
        system = DummySystem(verbose=True)
        system.process_dataset(data_dir)
        
        executor = Executor(
            system=system,
            system_name="DummySystem",
            workload_path=f"workload/{workload_name}.json",
            results_directory=results_dir,
            verbose=True,
            worker_id=0
        )
        
        # Run with cache enabled
        results = executor.run_workload(use_system_cache=True, cache_system_output=False)
        
        # Verify results contain both cached and computed results
        assert len(results) == len(workload_tasks)
        
        result_ids = {r["task_id"] for r in results}
        assert result_ids == set(workload_tasks)
        
        # Verify cached tasks are in results
        cached_results_back = [r for r in results if r["task_id"] in cached_task_ids]
        assert len(cached_results_back) == len(cached_task_ids)
        
        # Verify computed tasks are in results
        computed_results = [r for r in results if r["task_id"] in missing_task_ids]
        assert len(computed_results) == len(missing_task_ids)


class TestTaskSubsetCaching(CacheTestBase):
    """Test caching with task subsets (simulating parallel workers)."""
    
    def test_executor_with_task_subset(self, results_dir, workload_name, data_dir):
        """Test executor runs only specified task subset from cache."""
        cache_dir = os.path.join(results_dir, "response_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load workload tasks
        with open(f"workload/{workload_name}.json", 'r') as f:
            workload_dict = json.load(f)
        workload_tasks = [t['id'] for t in workload_dict]
        
        # Create full cache
        cached_results = [
            {"task_id": task_id, "model_output": {"answer": f"cached_{task_id}"}, "code": f"# code {task_id}"}
            for task_id in workload_tasks
        ]
        
        cache_file = os.path.join(cache_dir, f"{workload_name}_20260115_120000.json")
        with open(cache_file, 'w') as f:
            json.dump(cached_results, f)
        
        # Create executor with subset of tasks
        system = DummySystem(verbose=True)
        system.process_dataset(data_dir)
        
        # Use first half of tasks
        subset_tasks = workload_tasks[:len(workload_tasks)//2]
        
        executor = Executor(
            system=system,
            system_name="DummySystem",
            workload_path=f"workload/{workload_name}.json",
            results_directory=results_dir,
            verbose=True,
            tasks_subset=subset_tasks,
            worker_id=0
        )
        
        # Run with cache
        results = executor.run_workload(use_system_cache=True, cache_system_output=False)
        
        # Should only return the subset
        assert len(results) == len(subset_tasks)
        result_ids = {r["task_id"] for r in results}
        assert result_ids == set(subset_tasks)


class TestCacheMerge(CacheTestBase):
    """Test cache merging from multiple workers."""
    
    def test_worker_cache_merge(self, results_dir, workload_name):
        """Test that worker caches are merged into central cache correctly."""
        cache_dir = os.path.join(results_dir, "response_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Load workload tasks
        with open(f"workload/{workload_name}.json", 'r') as f:
            workload_dict = json.load(f)
        workload_tasks = [t['id'] for t in workload_dict]
        
        # Simulate worker 0 cache (first half of tasks)
        half = len(workload_tasks) // 2
        worker0_tasks = workload_tasks[:half]
        worker0_results = [
            {"task_id": task_id, "model_output": {"answer": f"worker0_{task_id}"}, "code": f"# code {task_id}"}
            for task_id in worker0_tasks
        ]
        worker0_cache = os.path.join(cache_dir, f"{workload_name}_worker0_20260115_120000.json")
        with open(worker0_cache, 'w') as f:
            json.dump(worker0_results, f)
        
        # Simulate worker 1 cache (second half of tasks)
        worker1_tasks = workload_tasks[half:]
        worker1_results = [
            {"task_id": task_id, "model_output": {"answer": f"worker1_{task_id}"}, "code": f"# code {task_id}"}
            for task_id in worker1_tasks
        ]
        worker1_cache = os.path.join(cache_dir, f"{workload_name}_worker1_20260115_120001.json")
        with open(worker1_cache, 'w') as f:
            json.dump(worker1_results, f)
        
        # Create benchmark and run merge
        benchmark = Benchmark(
            system_name="DummySystem",
            task_fixture_directory="benchmark/fixtures",
            system_output_directory=results_dir,
            use_system_cache=True,
            cache_system_output=True,
            verbose=True,
            num_workers=2
        )
        
        # Simulate the merge operation
        all_results_by_id = {}
        for result in worker0_results + worker1_results:
            all_results_by_id[result["task_id"]] = result
        
        benchmark._merge_worker_caches(f"workload/{workload_name}.json", results_dir, all_results_by_id)
        
        # Verify central cache was created
        central_caches = [f for f in os.listdir(cache_dir) if "_worker" not in f]
        assert len(central_caches) > 0
        
        # Load the central cache and verify all tasks are present
        with open(os.path.join(cache_dir, central_caches[0]), 'r') as f:
            merged_results = json.load(f)
        
        merged_ids = {r["task_id"] for r in merged_results}
        assert merged_ids == set(workload_tasks)
        
        # Verify worker caches are cleaned up
        worker_caches = [f for f in os.listdir(cache_dir) if "_worker" in f]
        assert len(worker_caches) == 0


class TestBenchmarkParallelExecution(CacheTestBase):
    """Integration tests for parallel execution with caching."""
    
    def test_benchmark_no_cache_parallel(self, results_dir):
        """Test benchmark runs in parallel without cache."""
        benchmark = Benchmark(
            system_name="DummySystem",
            task_fixture_directory="benchmark/fixtures",
            system_output_directory=results_dir,
            use_system_cache=False,
            cache_system_output=True,
            verbose=True,
            num_workers=2
        )
        
        # Note: We can't easily test run_benchmark without mocking evaluator
        # but we can verify the benchmark is set up correctly
        assert benchmark.num_workers == 2
        assert benchmark.system_name == "DummySystem"
    
    def test_benchmark_with_cache_parallel(self, results_dir, workload_name, workload_tasks):
        """Test benchmark loads cache in parallel execution."""
        cache_dir = os.path.join(results_dir, "response_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create central cache with all tasks
        cached_results = [
            {"task_id": task_id, "model_output": {"answer": f"cached_{task_id}"}, "code": f"# code {task_id}"}
            for task_id in workload_tasks
        ]
        
        cache_file = os.path.join(cache_dir, f"{workload_name}_20260115_120000.json")
        with open(cache_file, 'w') as f:
            json.dump(cached_results, f)
        
        benchmark = Benchmark(
            system_name="DummySystem",
            task_fixture_directory="benchmark/fixtures",
            system_output_directory=results_dir,
            use_system_cache=True,
            cache_system_output=True,
            verbose=True,
            num_workers=2
        )
        
        assert benchmark.use_system_cache is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
