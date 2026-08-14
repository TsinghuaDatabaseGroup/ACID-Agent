from textwrap import dedent
import sys
sys.path.append("./")


SINGLE_AGENT_TASK_PROMPT_TEMPLATE = """
            Workload name: {dataset_name}
            dataset_directory = {dataset_directory}
            Answer the question: {query}; Use the {dataset_name} dataset. 
            Subset files: {subset_files}
            
            If the subset files is not empty, use only those files to answer the question; otherwise, you have to figure out which files to use on your own.
            If you see unusual or unexpected intermediate results, use the critique agent to evaluate the step and provide feedback.
            DO NOT assume the correctness of the intermediate results and final results. Be skeptical and ensure they are correct by cross-checking with the dataset.
            Especially, if you see missing or incomplete intermediate results (e.g., nan or 0.0), question yourself if it's a coding issue, a logic design issue, or an actual data issue.
            After every step, ask yourself if every step is necessary and if the final answer is correct.
            **Report the final answer (just the answer, no explanation needed) AND the complete code pipeline used to get there in your final response, by following the instructions below.**
            IMPORTANT (use the given write_file tool): 
            - Write your final answer to {answer_path}
            - Write your complete code pipeline to {pipeline_code_path}
            - DO NOT write anything else besides using the write_file tool. E.g., NO intermediate results should be written.
            You can end the task after writing the files.
            IMPORTANT: Never generate plots. Do NOT import matplotlib or any visualization library.
            Do NOT call any plotting functions (plt.figure, plt.plot, plt.show, seaborn, plotly, etc).
            The environment is headless and will crash if graphical code is used.
            Return ONLY numerical or textual results.
        """

REFLEXION_TASK_PROMPT_TEMPLATE = """
            Workload name: {dataset_name}
            dataset_directory = {dataset_directory}
            Answer the question: {query}; Use the {dataset_name} dataset. 
            Subset files: {subset_files}
            
            If the subset files is not empty, use only those files to answer the question; otherwise, you have to figure out which files to use on your own.
            If you see unusual or unexpected intermediate results, use the critique agent to evaluate the step and provide feedback.
            IMPORTANT: Pass along the dataset path to the critique agent so it can access the data files if needed.
            DO NOT assume the correctness of the intermediate results and final results. Be skeptical and ensure they are correct by cross-checking with the dataset.
            Especially, if you see missing or incomplete intermediate results (e.g., nan or 0.0), question yourself if it's a coding issue, a logic design issue, or an actual data issue.
            After every step, ask yourself if every step is necessary and if the final answer is correct.
            **Report the final answer (just the answer, no explanation needed) AND the complete code pipeline used to get there in your final response, by following the instructions below.**
            IMPORTANT (use the given write_file tool): 
            - Write your final answer to {answer_path}
            - Write your complete code pipeline to {pipeline_code_path}
            - DO NOT write anything else besides using the write_file tool. E.g., NO intermediate results should be written.
            You can end the task after writing the files.
            IMPORTANT: Never generate plots. Do NOT import matplotlib or any visualization library.
            Do NOT call any plotting functions (plt.figure, plt.plot, plt.show, seaborn, plotly, etc).
            The environment is headless and will crash if graphical code is used.
            Return ONLY numerical or textual results.
        """

CRITIQUE_AGENT_SYSTEM_PROMPT = """
                An independent agent that acts as a critical reviewer for each step in a multi-step agent workflow.

                The critique_agent evaluates the most recent plan and corresponding tool output (observation) 
                produced by a primary agent, such as a CodeAgent. It provides constructive feedback, flags 
                issues, and suggests improvements or next actions without directly modifying the environment.

                This agent is intended to serve as a lightweight oversight mechanism in agentic systems, helping 
                to improve reliability, reduce errors, and promote clearer reasoning in autonomous workflows.

                Invoke this agent after each step of the primary agent to ensure that the actions taken are appropriate and well-reasoned.
            """

CRITIQUE_AGENT_PROMPT_TEMPLATE = """The agent has completed the following step:

### Your Job:
- Identify any issues, errors, or logical gaps
- Suggest the next action (if appropriate)
- Be concise but critical

Reply with your reasoning and recommendation.
"""


PDT_TASK_PROMPT_TEMPLATE = dedent("""
                Workload name: {dataset_name}
                dataset_directory = {dataset_directory}
                Answer the question: {query}; Use the {dataset_name} dataset.
                Subset files: {subset_files}

                You are part of a Planner → Subtask Decomposer → Tool/Code Executor (PDT) architecture.
                Your role will be determined by the orchestrator calling you (decomposer_agent, executor_agent).
                If the subset files is not empty, use only those files to answer the question; otherwise, you have to figure out which files to use on your own.
                IMPORTANT: Never generate plots. Do NOT import matplotlib or any visualization library.
                Do NOT call any plotting functions (plt.figure, plt.plot, plt.show, seaborn, plotly, etc).
                The environment is headless and will crash if graphical code is used.
                Return ONLY numerical or textual results.
                """)

DECOMPOSER_PROMPT_TEMPLATE = dedent("""
    You are the decomposer_agent in a Planner → Subtask Decomposer → Tool/Code Executor architecture.

    Workload name: {workload_name}

    User task:
    {task_prompt}

    Decompose this into a sequence of atomic subtasks that a code agent can execute.
    Each subtask should:
    - be as independent and concrete as possible,
    - be implementable by executing Python over the workload data,
    - have explicit dependencies.
    - 5 is the MAX number of subtasks you can create. 

    STRICTLY output valid JSON with this schema and nothing else:

    [
      {{"id": 1, "description": "...", "depends_on": []}},
      {{"id": 2, "description": "...", "depends_on": [1]}},
      ...
    ]
""")

EXECUTOR_PROMPT_TEMPLATE = dedent("""
            You are the executor_agent in a Planner → Decomposer → Executor (PDT) architecture.

            Global workload name: {workload_name}

            Global user task:
            {task_prompt}

            You are now executing Subtask {sid}.

            Subtask {sid} description:
            \"\"\"{desc}\"\"\"

            Summaries of completed dependency subtasks:
            {deps_text}

            Your job:
            - Focus ONLY on Subtask {sid}.
            - Use Python code (and the available tools) to complete this subtask over the workload dataset.
            - Print intermediate results for debugging and verification.
            - At the end, return a concise textual summary of:
            - What you did
            - Key intermediate results
            - Any important caveats or assumptions

            Return only the answer (no explanation).
            """)

EXECUTOR_PROMPT_TEMPLATE_LAST_STEP = dedent("""
                IMPORTANT: This is the FINAL subtask. After completing it, you MUST write the final answer and complete code pipeline to files using the write_file tool.
                - Write your final answer to {answer_path}
                - Write your complete code pipeline to {pipeline_code_path}
                - DO NOT write anything else besides using the write_file tool. E.g., NO intermediate results should be written.
                You can end the task after writing the files.
                IMPORTANT: Never generate plots. Do NOT import matplotlib or any visualization library.
                Do NOT call any plotting functions (plt.figure, plt.plot, plt.show, seaborn, plotly, etc).
                The environment is headless and will crash if graphical code is used.
                Return ONLY numerical or textual results.
                """)

DECOMPOSER_DESCRIPTION = (
                "Given a user task and a high-level PLAN from the planner_agent, "
                "this agent decomposes the plan into a small sequence of atomic subtasks "
                "that a code executor can implement. It MUST output a JSON list of subtasks "
                "with the schema:\n\n"
                "[\n"
                "  {\"id\": 1, \"description\": \"...\", \"depends_on\": []},\n"
                "  {\"id\": 2, \"description\": \"...\", \"depends_on\": [1]},\n"
                "  ...\n"
                "]\n\n"
                "Each description should be concrete enough to be implemented in Python "
                "with access to the workload dataset.\n\n"
                "The list MUST contain between 1 and 5 subtasks (inclusive). "
                "If more than 5 steps seem necessary, merge related steps."
            )

EXECUTOR_DESCRIPTION = (
                "A Python-executing agent that solves a SINGLE subtask at a time. "
                "Given the global task, the high-level plan, and the current subtask "
                "description, it should write and run Python code over the workload "
                "dataset to complete that subtask. It should print intermediate results, "
                "and return a concise textual summary of what was done and the key outputs. "
                "Use write_file tool to record the end-to-end code pipeline and final answer."
            )