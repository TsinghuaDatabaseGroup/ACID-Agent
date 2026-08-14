# type: ignore
import os
import sys
import re
import fnmatch
sys.path.append("./")
from benchmark.benchmark_utils import print_error, print_warning

import json
import pandas as pd
import subprocess
from typeguard import typechecked
from typing import Dict, List

from benchmark.benchmark_api import System
from .baseline_prompts import QUESTION_PROMPT, QUESTION_PROMPT_NO_DATA
from .baseline_utils import get_table_string, clean_nan, extract_code, preview_text
from .generator_utils import Generator, OllamaGenerator


class BaselineLLMSystem(System):
    """
    A baseline system that uses a large language model (LLM) to process datasets and serve queries.
    """

    def __init__(self, model: str, name="baseline", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dataset_directory = None  # TODO(user): Update me
        self.model = model
        self.llm = Generator(model, verbose=self.verbose)

        # Initialize directories for output and intermediate files
        self.variance = kwargs.get("variance", "one_shot")
        # variance has to be one_shot or few_shot
        assert self.variance in [
            "one_shot",
            "few_shot",
        ], f"Invalid variance: {self.variance}. Must be one_shot or few_shot."

        self.supply_data_snippet = kwargs.get("supply_data_snippet", False)        
        self.verbose = kwargs.get("verbose", False)

        self.debug = False
        self.output_dir = kwargs.get("output_dir", os.path.join(os.getcwd(), "testresults"))

        # Ablation studies: add additional parameters
        self.number_sampled_rows = kwargs.get("number_sampled_rows", 100)
        self.max_tries = kwargs.get("max_tries", 5)

        self.question_output_dir = None  # to be set in run()
        self.question_intermediate_dir = None  # to be set in run()

    def _init_output_dir(self, query_id: str) -> None:
        """
        Initialize the output directory for the question.
        :param question: Question object
        """
        question_output_dir = os.path.join(self.output_dir, query_id)
        if not os.path.exists(question_output_dir):
            os.makedirs(question_output_dir)
        self.question_output_dir = question_output_dir
        self.question_intermediate_dir = os.path.join(
            self.question_output_dir, "_intermediate"
        )
        if not os.path.exists(self.question_intermediate_dir):
            os.makedirs(self.question_intermediate_dir)

    def get_input_data(self, filenames) -> str:
        """
        Get the input data from the data sources.
        Naively read the first 10 rows of each data source.
        :return: Data as a string
        """
        rows = getattr(self, "number_sampled_rows", 10)
        data_string = ""
        for file_name in filenames:
            data_string += f"\nFile name: {file_name}\n"

            if file_name not in self.dataset:
                data_string += "Status: not loaded in dataset.\n" + "=" * 20 + "\n"
                continue

            data = self.dataset[file_name]

            # Case 1: Pandas DataFrame
            if isinstance(data, pd.DataFrame):
                data_string += f"Type: table (pandas.DataFrame)\n"
                data_string += f"Column data types:\n{data.dtypes}\n"
                data_string += "Table:\n"
                try:
                    data_string += get_table_string(data, row_limit=rows)
                except Exception:
                    # Very defensive: fallback to DataFrame.head() string if helper fails
                    data_string += str(data.head(rows))
                data_string += "\n" + "=" * 20 + "\n"
                continue

            # Case 2: Plain text
            if isinstance(data, str):
                data_string += "Type: text\n"
                data_string += "Preview (first lines):\n"
                data_string += preview_text(data, max_lines=rows)
                data_string += "\n" + "=" * 20 + "\n"
                continue

            # Fallback: unknown object type
            data_string += f"Type: {type(data).__name__}\n"
            data_string += f"String preview:\n{str(data)[:1000]}"
            data_string += ("\n... (truncated)" if len(str(data)) > 1000 else "")
            data_string += "\n" + "=" * 20 + "\n"

        return data_string

    def generate_error_handling_prompt(
        self, code_fp: str, error_fp: str, output_fp: str
    ) -> str:
        """
        Generate a prompt for the LLM to handle errors in the code.
        :param code_fp: Path to the code file
        :param error_fp: Path to the error file
        :return: Prompt string
        """
        with open(code_fp, "r") as f:
            code = f.read()
        with open(error_fp, "r") as f:
            errors = f.read()
        with open(output_fp, "r") as f:
            outputs = f.read()

        prompt = f"""
        Your task is to fix the following Python code based on the provided error messages.
        Code:
        {code}

        Errors:
        {errors}

        Output:
        {outputs}

        Please provide the fixed code. 
        Mark the code with ````python` and ````python` to indicate the start and end of the code block.
        """
        return prompt

    def generate_prompt(self, query: str, subset_files:List[str]=[]) -> str:
        """
        Generate a prompt for the LLM based on the question.
        :param query: str
        :param subset_files: list of strings with filenames
        :return: Prompt string
        """
        # Generate the RAG plan
        # TODO: use process_dataset() to get the data
        # get the file names from the dataset directory
        if len(subset_files):
            file_names = []
            all_file_names = list(self.dataset.keys())
            for pattern in subset_files:
                # print(self.dataset.keys())
                # assert f in self.dataset.keys(), f"File {f} is not in dataset!"
                # Relaxed the assertion to a warning
                matching = [
                    f
                    for f in all_file_names
                    if fnmatch.fnmatch(f, pattern)
                    or fnmatch.fnmatch(os.path.basename(f), os.path.basename(pattern))
                ]
                if len(matching) == 0:
                    print(f"WARNING: File {pattern} is not in dataset!")
                else:  # only extend if there are matches
                    file_names.extend(matching)
        else:
            file_names = list(self.dataset.keys())

        print(f"Using {len(file_names)} files for the prompt: {file_names}")
        data = self.get_input_data(file_names)

        file_paths = [os.path.join(self.dataset_directory, file) for file in file_names]

        example_json = {
            "id": "main-task",
            "query": query,
            "data_sources": ["file1.csv", "file2.csv"],
            "subtasks": [
                {
                    "id": "subtask-1",
                    "query": "What is the exceedance rate in 2022?",
                    "data_sources": ["water-body-testing-2022.csv"],
                },
                {
                    "id": "subtask-2",
                    "query": "What is the column name for the exceedance rate?",
                    "data_sources": ["water-body-testing-2022.csv"],
                },
            ],
        }
        if self.supply_data_snippet:
            prompt = QUESTION_PROMPT.format(
                query=query,
                file_names=str(file_names),
                data=data,
                example_json=json.dumps(example_json, indent=4),
                file_paths=str(file_paths),  # ", ".join(file_paths)
            )
        else:
            prompt = QUESTION_PROMPT_NO_DATA.format(
                query=query,
                file_names=str(file_names),
                example_json=json.dumps(example_json, indent=4),
                file_paths=str(file_paths),  # ", ".join(file_paths)
            )

        # Save the prompt to a txt file
        prompt_fp = os.path.join(self.question_output_dir, "prompt.txt")
        with open(prompt_fp, "w") as f:
            f.write(prompt)
        if self.verbose:
            print_warning(f"{self.name}: Prompt saved to {prompt_fp}")
        return prompt

    def extract_response(self, response, try_number: int):
        """
        Process the LLM response.
        :param response: LLM response string
        :return: Processed response
        """
        json_fp = ""
        code_fp = ""
        # Save the full response to a txt file
        response_fp = os.path.join(self.question_output_dir, "initial_response.txt")
        with open(response_fp, "w") as f:
            f.write(response)
        if self.verbose:
            print_warning(f"{self.name}: Response saved to {response_fp}")

        # Assume the step-by-step plan is fixed after the first try
        if try_number == 0:
            # Extract the JSON array from the response
            json_response = extract_code(response, pattern=r"```json(.*?)```")
            # print("Extracted JSON:", json_response)
            # Save the JSON response to a file
            json_fp = os.path.join(self.question_output_dir, "answer.json")
            with open(json_fp, "w") as f:
                f.write(json_response)
            if self.verbose:
                print(f"{self.name}: JSON response saved to {json_fp}")

        # Extract the code from the response
        code = extract_code(response, pattern=r"```python(.*?)```")
        # print("Extracted Code:", code)
        # Save the code to a file
        code_fp = os.path.join(
            self.question_output_dir, "_intermediate", f"pipeline-{try_number}.py"
        )  # {question.id}-{try_number}
        with open(code_fp, "w") as f:
            f.write(code)
        if self.verbose:
            print(f"{self.name}: Code saved to {code_fp}")

        return json_fp, code_fp

    def execute_code(self, code_fp, try_number: int):
        """
        Execute the code in the file and save the output.
        :param code_fp: Path to the code file
        :return: Execution result
        """
        # Execute the code and save error messages
        result = subprocess.run(
            ["python", code_fp],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Save the printed output of the code execution
        output_fp = os.path.join(
            self.question_intermediate_dir, f"pipeline-{try_number}_out.json"
        )
        with open(output_fp, "w") as f:
            # Clean NaN values to null for strict JSON compliance
            stdout = clean_nan(result.stdout)
            f.write(stdout)

        # Save only compile/runtime errors
        error_fp = os.path.join(
            self.question_intermediate_dir, f"errors-{try_number}.txt"
        )
        with open(error_fp, "w") as f:
            f.write(result.stderr)

        return output_fp, error_fp

    @typechecked
    def _format_answer_on_fail(
        self,
        answer: Dict | List,
        error_msg: str = "Warning: No answer found in the Python pipeline.",
    ) -> Dict[str, str | Dict | List]:
        formatted_answer = {}
        for step in answer:
            if step["id"] == "main-task":
                formatted_answer |= step
                if "answer" not in formatted_answer:
                    formatted_answer["answer"] = error_msg
            else:
                if "subtasks" in formatted_answer:
                    found = False
                    for subtask in formatted_answer["subtasks"]:
                        if subtask["id"] == step["id"]:
                            found = True
                        if "answer" not in subtask:
                            subtask["answer"] = error_msg
                    if not found:
                        formatted_answer["subtasks"].append(step)
        return formatted_answer

    @typechecked
    def process_response(
        self, json_fp, output_fp, error_fp=None
    ) -> Dict[str, str | Dict | List]:
        """
        Process the response and fill in the JSON response with the execution result.
        :param question: Question object
        :param json_fp: Path to the JSON file
        :param output_fp: Path to the output file
        """
        # Load the JSON response
        try:
            with open(json_fp, "r") as f:
                answer = json.load(f)
            # Check if items in answer are not dicts
            for step in answer:
                if isinstance(step, list) or isinstance(step, str):
                    print(f"ERROR: {self.name}: ** ERRORS ** answer is not a dict: {step}")
                    return self._format_answer_on_fail(answer, f"** ERRORS ** answer is not a dict: {step}")
        except json.JSONDecodeError as e:
            print(f"ERROR: {self.name}: ** ERRORS ** decoding answer JSON: {e}")
            return {"id": "main-task", "answer": "SUT failed to answer this question."}

        # Check if the error file is empty
        if error_fp is not None:
            if os.path.getsize(error_fp) > 0:
                print_error(
                    f"ERROR: {self.name}: ** ERRORS ** found in {error_fp}. Skipping JSON update."
                )
                return self._format_answer_on_fail(
                    answer, f"** ERRORS ** found during execution in {error_fp}"
                )

        # Load the output file
        try:
            with open(output_fp, "r") as f:
                output = json.load(f)
        except json.JSONDecodeError as e:
            print_error(f"ERROR: {self.name}: ** ERRORS ** decoding output JSON: {e}")
            return self._format_answer_on_fail(
                answer, f"** ERRORS ** decoding output in {output_fp}"
            )

        # Fill in the JSON answer with the execution result
        for step in answer:
            id = "main-task"
            if id in output:
                step["answer"] = output[id]
            else:
                step["answer"] = "Warning: No answer found in the Python pipeline."
            # Update the subtasks with the output
            subtasks = step.get("subtasks", [])
            for subtask in subtasks:
                subtask_id = subtask["id"]
                if subtask_id in output:
                    subtask["answer"] = output[subtask_id]
                else:
                    subtask["answer"] = (
                        "Warning: No answer found in the Python pipeline."
                    )

        overall_answer = {"system_subtasks_responses": []}
        for step in answer:
            if step["id"] == "main-task":
                overall_answer |= step
            else:
                overall_answer["system_subtasks_responses"].append(step)

        # Save the updated JSON answer
        with open(json_fp, "w") as f:
            # Clean NaN values to null for strict JSON compliance
            overall_answer = clean_nan(overall_answer)
            json.dump(overall_answer, f, indent=4)
        if self.verbose:
            print_warning(f"{self.name}: Updated JSON answer saved to {json_fp}")
        return overall_answer

    @typechecked
    def run_one_shot(self, query: str, query_id: str, subset_files:List[str]=[]) -> Dict[str, str | Dict | List]:
        """
        This function demonstrates a simple one-shot LLM approach to solve the LLMDS benchmark.
        """
        self._init_output_dir(query_id)

        # Generate the prompt
        prompt = self.generate_prompt(query, subset_files)
        if self.debug:
            print(f"{self.name}: Prompt:", prompt)

        # Get the model's response
        messages = [
            {"role": "system", "content": "You are an experienced data scientist."},
            {"role": "user", "content": prompt},
        ]
        response = self.llm(messages)
        if self.debug:
            print_warning(f"{self.name}: Response:", response)

        # Process the response
        json_fp, code_fp = self.extract_response(response, try_number=0)
        # Execute the code (if necessary)
        output_fp, error_fp = self.execute_code(code_fp, try_number=0)

        token_count = self.llm.total_tokens
        token_count_input = self.llm.input_tokens
        token_count_output = self.llm.output_tokens
        if self.verbose:
            print_warning(f"{self.name}: Response token count: {token_count}")
            print_warning(f"{self.name}: Input token count: {token_count_input}")
            print_warning(f"{self.name}: Output token count: {token_count_output}")

        # Save the token count to a file
        token_fp = os.path.join(self.question_output_dir, "_intermediate", f"token_count-0.txt")
        with open(token_fp, "w") as f:
            token_data = {
                    "token_usage": token_count,
                    "token_usage_input": token_count_input,
                    "token_usage_output": token_count_output,
                }
            json.dump(token_data, f, ensure_ascii=False, indent=4)

        answer = None
        pipeline_code = None

        # Read the pipeline code regardless of the execution result for semantic eval
        with open(code_fp, "r") as f:
            pipeline_code = f.read()

        # Fill in JSON response with the execution result
        answer = self.process_response(json_fp, output_fp, error_fp)
        output_dict = {"explanation": answer, "pipeline_code": pipeline_code, "token_usage": token_count, "token_usage_input": token_count_input, "token_usage_output": token_count_output}

        return output_dict

    @typechecked
    def run_few_shot(self, query: str, query_id: str, subset_files:List[str]=[]) -> Dict[str, str | Dict | List]:
        """
        This function demonstrates a simple few-shot LLM approach to solve the LLMDS benchmark.
        """
        self._init_output_dir(query_id)

        # Generate the prompt
        prompt = self.generate_prompt(query, subset_files)
        if self.debug:
            print(f"{self.name}: Prompt:", prompt)

        messages = [
            {"role": "system", "content": "You are an experienced data scientist."},
        ]

        answer = None
        pipeline_code = None

        total_token_usage = 0
        total_token_usage_input = 0
        total_token_usage_output = 0
        for try_number in range(self.max_tries):
            messages.append({"role": "user", "content": prompt})
            # Get the model's response
            response = self.llm(messages)
            if self.debug:
                print(f"{self.name}: Response:", response)
            messages.append({"role": "assistant", "content": response})

            # Process the response
            if try_number == 0:
                json_fp, code_fp = self.extract_response(response, try_number)
            else:
                _, code_fp = self.extract_response(response, try_number)

            # Parse token count and clean the response
            token_count = self.llm.total_tokens
            token_count_input = self.llm.input_tokens
            token_count_output = self.llm.output_tokens
            total_token_usage += token_count
            total_token_usage_input += token_count_input
            total_token_usage_output += token_count_output
            if self.verbose:
                print_warning(f"{self.name}: Response token count: {token_count}")
                print_warning(f"{self.name}: Input token count: {token_count_input}")
                print_warning(f"{self.name}: Output token count: {token_count_output}")
            # Save the token count to a file
            token_fp = os.path.join(self.question_output_dir, "_intermediate", f"token_count-{try_number}.json")
            with open(token_fp, "w") as f:
                token_data = {
                    "token_usage": token_count,
                    "token_usage_input": token_count_input,
                    "token_usage_output": token_count_output,
                }
                json.dump(token_data, f, ensure_ascii=False, indent=4)

            # Execute the code (if necessary)
            output_fp, error_fp = self.execute_code(code_fp, try_number)
            # print("Execution Result:", result)

            if os.path.getsize(error_fp) > 0:
                prompt = self.generate_error_handling_prompt(
                    code_fp, output_fp, error_fp
                )
            else:
                # Fill in JSON response with the execution result
                answer = self.process_response(json_fp, output_fp, error_fp=None)
                # Read the pipeline code
                with open(code_fp, "r") as f:
                    pipeline_code = f.read()
                output_dict = {"explanation": answer, "pipeline_code": pipeline_code}
                break

        if answer is None or pipeline_code is None:
            answer = {
                "id": "main-task",
                "answer": f"Pipeline not successful after {self.max_tries} tries.",
            }
            output_dict = {"explanation": answer, "pipeline_code": ""}
        output_dict["token_usage"] = total_token_usage
        output_dict["token_usage_input"] = total_token_usage_input
        output_dict["token_usage_output"] = total_token_usage_output
        return output_dict

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """
        Process the dataset located in the specified directory.
        The dataset can contain files in various formats (e.g., PDF, CSV).
        """
        self.dataset_directory = dataset_directory
        self.dataset = {}
        # read the files
        for dirpath, _, filenames in os.walk(dataset_directory):
            for fname in filenames:
                rel_path = os.path.relpath(os.path.join(dirpath, fname), dataset_directory)
                file_path = os.path.join(dirpath, fname)

                #print(f"Loading file: {file_path}")

                if fname.lower().endswith(".csv"):
                    try:
                        self.dataset[rel_path] = pd.read_csv(
                            file_path,
                            engine="python",
                            on_bad_lines="skip",
                        )
                        continue
                    except UnicodeDecodeError:
                        try:
                            self.dataset[rel_path] = pd.read_csv(
                                file_path,
                                engine="python",
                                on_bad_lines="skip",
                                encoding="latin-1",
                            )
                            continue
                        except Exception:
                            pass
                    except Exception:
                        pass

                # Try generic text read
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        self.dataset[rel_path] = f.read()
                    continue
                except Exception:
                    pass

                # Try binary read
                try:
                    with open(file_path, "rb") as f:
                        self.dataset[rel_path] = f.read()
                    continue
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    self.dataset[rel_path] = ""  # Mark as failed

    @typechecked
    def serve_query(self, query: str, query_id: str = "default_name-0", subset_files:List[str]=[]) -> Dict:
        """
        Serve a query using the LLM.
        The query should be in natural language, and the response can be in either natural language or JSON format.
        :param query: str
        :param query_id: str
        :param subset_files: list of strings with filenames to include in the experiments
        :return: output_dict {"explanation": answer, "pipeline_code": pipeline_code}
        """
        # TODO: Implement the logic to handle different types of queries
        if self.variance == "one_shot":
            output_dict = self.run_one_shot(query, query_id, subset_files)
        elif self.variance == "few_shot":
            output_dict = self.run_few_shot(query, query_id, subset_files)
        return output_dict


class BaselineLLMSystemOllama(BaselineLLMSystem):
    """
    A baseline system that uses a large language model (LLM) to process datasets and serve queries.
    """

    def __init__(self, model: str, name="baseline", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.llm = OllamaGenerator(model=model)


def main():
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    questions = [
        {
            "id": "environment-easy-1",
            "query": "In 2018, how many bacterial exceedances were observed in freshwater beaches?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-medium-1",
            "query": "For the freshwater beaches, what is the difference between the percentage exceedance rate in 2023 and the historic average from 2002 to 2022?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-hard-1",
            "query": "For the marine beaches, what is the Pearson product-moment correlation from 2002 to 2023 between the rainfall amount in inches during the months June, July, August, and September and the percentage exceedance rate?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
    ]

    # Process each question
    out_dir = os.path.join(current_dir, "../testresults/run2")
    baseline_llm = BaselineLLMSystem(
        model="gpt-4o", output_dir=out_dir, variance="few_shot", verbose=True
    )
    # Process the dataset
    baseline_llm.process_dataset(questions[0]["dataset_directory"])
    for question in questions:
        print(f"Processing question: {question['id']}")
        # For debugging purposes, also input question.id
        baseline_llm.serve_query(question["query"], question["id"])

if __name__ == "__main__":
    main()
