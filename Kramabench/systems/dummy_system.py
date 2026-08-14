"""
Dummy system for testing the benchmarking pipeline functionality.
This system does not perform any real processing and returns placeholder results.
"""
import random
import os
import time
from benchmark.benchmark_api import System
from typing import List, Dict, Any


class DummySystem(System):
    """
    A minimal dummy system for testing the benchmarking and caching infrastructure.
    
    - process_dataset: Stores filenames from the dataset directory
    - serve_query: Returns a placeholder response in the standard format
    """

    def __init__(self, name: str = "DummySystem", *args, **kwargs):
        """Initialize the dummy system."""
        super().__init__(name, *args, **kwargs)
        self.dataset_files: List[str] = []

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """
        Process dataset by storing filenames from the directory.
        
        Args:
            dataset_directory: Path to the dataset directory
        """
        self.dataset_directory = dataset_directory
        self.dataset_files = []
        
        if os.path.isdir(dataset_directory):
            for root, dirs, files in os.walk(dataset_directory):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), dataset_directory)
                    self.dataset_files.append(relative_path)
        
        if self.verbose:
            print(f"DummySystem: Processed {len(self.dataset_files)} files from {dataset_directory}")

    def serve_query(
        self, 
        query: str, 
        query_id: str = "default_name-0", 
        subset_files: List[str] | None = None
    ) -> Dict[str, Any]:
        """
        Serve a query with a placeholder response.
        
        Returns a response in the standard format:
        {
            "explanation": {
                "answer": placeholder answer text
            },
            "pipeline_code": placeholder Python code,
            "token_usage": 0,
            "token_usage_input": 0,
            "token_usage_output": 0
        }
        
        Args:
            query: The query string
            query_id: Unique identifier for the query
            subset_files: Optional list of files to use
            
        Returns:
            Dict with explanation, pipeline_code, and token usage fields
        """

        # Simulate some processing delay
        time.sleep(random.uniform(0.05, 0.5))

        answer = f"A dummy answer"
        
        # Create placeholder pipeline code
        pipeline_code = """
# Dummy pipeline code
def process_data():
    result = {"answer": "dummy result"}
    return result

if __name__ == "__main__":
    output = process_data()
    print(output)
"""
        response = {
            "explanation": {
                "answer": answer,
                "id": query_id,
            },
            "pipeline_code": pipeline_code,
            "token_usage": 0,
            "token_usage_input": 0,
            "token_usage_output": 0
        }
        

        # Log that the query is being served when verbose mode is enabled
        if self.verbose:
            print(f"DummySystem: Serving query {query_id}")
        
        return response
