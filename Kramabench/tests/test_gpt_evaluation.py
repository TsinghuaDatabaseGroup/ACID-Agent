import os
from benchmark.benchmark_utils import get_most_recent_cache
from benchmark.llm_tools import GPTInterface
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Temporary test
if __name__ == "__main__":
    # Check if API key is set
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Create a test instance
    try:
        # Use a model that supports JSON response format
        gpt_interface = GPTInterface(model="gpt-5")
        print("Using gpt-5 model")
    except Exception as e:
        try:
            # Fall back to regular gpt-5 and modify the method temporarily
            gpt_interface = GPTInterface(model="gpt-5")
            print("Using gpt-5 model (JSON response format may not work)")
        except Exception as e2:
            print(f"Error creating GPTInterface: {e2}")
            exit(1)
    
    # Sample data for testing
    sample_task = {
        "query": "Analyze water quality data from multiple monitoring stations",
        "subtasks": [
            {"step": "Load and merge water quality datasets from different stations"},
            {"step": "Clean and preprocess the data, handling missing values"},
            {"step": "Calculate water quality indices and trend analysis"},
            #{"step": "Generate visualization and summary report"}
        ]
    }
    
    sample_pipeline = """
    1. Load water quality data from CSV files
    2. Merge datasets using station ID as key
    3. Clean data by removing outliers and interpolating missing values
    4. Calculate Water Quality Index (WQI) for each station
    5. Perform time series analysis to identify trends
    6. Create charts showing WQI trends over time
    7. Generate final report with recommendations
    """
    
    print("Testing evaluate_data_pipeline...")
    print(f"Pipeline: {sample_pipeline[:100]}...")
    print(f"Task has {len(sample_task['subtasks'])} subtasks")
    
    try:
        result = gpt_interface.evaluate_data_pipeline(sample_pipeline, sample_task)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Try a simpler test without JSON format
        print("Trying alternative approach without JSON format...")
        
        # Test the message formatting
        subtasks = sample_task.get("subtasks", [])
        messages = gpt_interface._format_pipeline_evaluation_messages(subtasks, sample_pipeline, sample_task)
        print(f"Formatted {len(messages)} messages successfully")
        print(f"Last message content preview: {messages[-1]['content'][0]['text'][:200]}...")
        
    print("Done!")