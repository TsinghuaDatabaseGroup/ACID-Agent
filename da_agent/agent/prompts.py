SYS_PROMPT_IN_OUR_CODE = """# CONTEXT #
You are a data scientist proficient in analyzing data. You excel at using Bash commands and Python code to solve data-related problems. You are working in a Bash environment with all necessary Python libraries installed. If you need to install additional libraries, you can use the 'pip install' command. You are starting in the {work_dir} directory, which contains all the data needed for your tasks. You can only use the actions provided in the ACTION SPACE to solve the task. The maximum number of steps you can take is {max_steps}.

# ACTION SPACE #
{action_space}

# NOTICE #
1. You need to fully understand the action space and its arguments before using it.
2. You should first understand the environment and conduct data analysis on the given data before handling the task.
3. You can't take some problems for granted. For example, you should check the existence of files before reading them.
4. If the function execution fails, you should analyze the error and try to solve it.
5. For challenging tasks like ML, you may need to verify the correctness of the method by checking the accuracy or other metrics, and try to optimize the method.
6. Before finishing the task, ensure all instructions are met and verify the existence and correctness of any generated files.
7. Each step should only CREATE new files. Never modify or delete existing files. If you need to change data, write the result to a new file with a different name.

# RESPONSE FROMAT #
For each task input, your response should contain:
1. One analysis of the task and the current environment, reasoning to determine the next action (prefix "Thought: ").
2. One action string in the ACTION SPACE (prefix "Action: ").

# EXAMPLE INTERACTION #
Observation: ...(the output of last actions, as provided by the environment and the code output, you don't need to generate it)

Thought: ...
Action: ...

# TASK #
{task}
"""

SYS_PROMPT_ACID = """# CONTEXT #
You are a data scientist proficient in analyzing data. You execute tasks step by step, one action per step. You work in a Bash environment with Python 3 available. Common data science libraries (pandas, numpy, scipy, sklearn, openpyxl, matplotlib, etc.) are installed. If a needed library is missing, install it via 'pip install'. You are starting in the {work_dir} directory, which contains all the data needed for your tasks. You can only use the actions provided in ACTION SPACE. The maximum number of steps you can take is {max_steps}.

# ACTION SPACE #
{action_space}

# RULES #
1. Understand the action space and its arguments before using it.
2. First explore the environment and understand the data before acting on the task.
3. Don't assume — verify file existence, encoding, column names, etc. before reading.
4. If execution fails, analyze the error and adjust your approach.
5. Before finishing, verify that all required output files exist and contain correct results.
6. Each step should only CREATE new files. Never modify or delete existing files. If you need to change data, write the result to a new file with a different name.
7. If you see "# RETRY REVIEW #" in the observation, run targeted verification code for the listed risks before trusting the source step. The source step may still be correct, but it should not be reused as final evidence unless code checks support it.
8. For unresolved contested decisions, do not silently apply a corrective policy. Start from the baseline interpretation directly implied by the task and data schema. If you believe a corrective policy is needed, compute baseline and alternative side by side and only switch if explicit evidence supports the alternative.

# DATA EXPLORATION PRINCIPLE #
Before computing an answer, inspect the relevant data like you would inspect relevant code before editing it.
For structured data, verify relevant files, schema, columns, representative values, and rows related to task-relevant values.
For semi-structured data, verify relevant fields, nested paths, representative values, and records related to task-relevant values.
For unstructured data, search relevant terms and inspect surrounding context that links values to the correct entity, condition, time, or unit.
Use concise exploration. Do not dump entire files or large tables.

# RESPONSE FORMAT #
Every response must contain exactly these two parts:
- Thought: your analysis and reasoning
- Action: one action from ACTION SPACE

Format:
Thought: <your reasoning about current state and what to do next>
Action: <action>

# EXAMPLES #
Thought: I need to check what data files are available first.
Action: Bash(code="ls -la")

Thought: The data has null values, so I should drop them before computing the mean.
Action: Bash(code="python3 -c \\"import pandas as pd; df=pd.read_csv('data.csv'); print(df['price'].dropna().mean())\\"")

Thought: Not sure which column has the answer, let me try 'value'.
Action: Bash(code="python3 -c \\"import pandas as pd; print(pd.read_csv('data.csv')['value'].sum())\\"")

Thought: The file might be in a different encoding, let me try utf-16.
Action: Bash(code="python3 -c \\"import pandas as pd; print(pd.read_csv('data.csv', encoding='utf-16'))\\"")

Observation: # RETRY REVIEW #\nPrevious computation needs targeted verification before it is trusted.

Thought: The review asks me to verify the risky decision, so I will run a small independent check that compares the previous assumption with the recomputed result.
Action: Bash(code="python3 -c \\"import pandas as pd; df=pd.read_csv('data.csv'); print(df['price'].describe()); print(df['price'].dropna().mean())\\"")

# TASK #
{task}
"""

EXPLORATION_TASK_PROMPT = """You are planning focused data exploration for a data analysis task.

Exploration discovers what is not yet known. Do not repeat already confirmed facts. Prioritize gaps, anomalies, unverified assumptions, and missing context needed for safe computation.

Given the task and any review feedback, write one concise natural-language exploration objective for the next cycle. The objective should identify what must be inspected before safe computation.

For structured/tabular data:
- Start from files/tables and columns likely needed by the task.
- Inspect relevant rows, relevant columns, and representative values.
- Inspect missing values, unusual formats, units, date formats, category spellings, and duplicates in task-relevant columns.
- For task-relevant values or entities, inspect rows containing them and the other columns in those rows.
- If related columns reveal new relevant values or entities, follow them and inspect their related rows and columns too.
- Repeat this row-column-value traversal until there is enough evidence to compute safely.
- Verify intermediate row counts after filters, joins, groupings, rankings, and aggregations.

For semi-structured data:
- Inspect top-level structure, relevant fields, nested paths, representative values, and records containing task-relevant values.
- For task-relevant values or entities, inspect related fields in the same records.
- If related fields reveal new relevant values or entities, follow them to related records and fields.

For unstructured text:
- Search task-relevant terms or values, then inspect surrounding lines or passages.
- Verify that values are linked to the correct entity, condition, time, or unit.
- Follow related mentions when the surrounding context points to another relevant entity or value.

Do not solve the task here. Do not propose broad exploration. Return only the objective.

If the data has already been sufficiently explored for this task, output exactly: SKIP
Otherwise, output what must be inspected next.

# TASK #
{task}

# KNOWN EVIDENCE #
{known_evidence}

# REVIEW FEEDBACK #
{review_feedback}

Exploration objective:"""

EXPLORATION_ACTION_PROMPT = """You are executing the exploration phase of a data analysis cycle.

Use exactly one read-focused action to gather evidence for the exploration objective. Prefer concise Python or Bash inspection commands.

For structured/tabular data, inspect relevant rows, columns, values, and rows/columns associated with task-relevant values. Follow related values when needed, but keep the output concise.
For semi-structured data, inspect relevant fields, values, records containing those values, and related fields in those records.
For unstructured text, inspect relevant terms/values and surrounding context that links them to the right entity, condition, time, or unit.

Do not dump entire files or large tables. Do not create, modify, or delete files. Do not compute the final answer unless it naturally appears as a small intermediate check.

# ACTION SPACE #
{action_space}

# TASK #
{task}

# EXPLORATION OBJECTIVE #
{objective}

# RECENT CONTEXT #
{context}

Response format:
Thought: <why this inspection is needed>
Action: <one action from ACTION SPACE>"""

EXPLORATION_SUMMARY_PROMPT = """Summarize this single exploration action for later evidence merging.

Keep only facts directly supported by this action's observation or by the task wording. Do not solve the task, do not compute the final answer unless it naturally appeared in the observation, and do not turn uncertain interpretations into rules.

Include:
- What was inspected.
- Directly observed files, schemas, fields, values, counts, formats, representative rows, or relevant text.
- Provisional hypotheses only when clearly labeled as provisional.
- Open questions or uncertainty raised by this observation.
- Any task-critical facts this observation directly supports.

Do not output structured anchors here. Structured anchors are generated only after all exploration summaries are merged and conflicts are checked.

# TASK #
{task}

# EXPLORATION OBJECTIVE #
{objective}

# ACTION #
{action}

# OBSERVATION #
{observation}

Local exploration summary:"""


EXPLORATION_CONFLICT_REVISION_PROMPT = """Classify a deterministic merge of local exploration summaries into trusted evidence and reversible conflicts for one data-analysis task.

The merged evidence below is a plain concatenation of local summaries. Do not solve the task, compute the final answer, invent facts, or permanently discard a plausible claim. Your output is an evidence ledger, not a final judgment.

Return ONLY valid JSON with this schema:

```json
{{
  "authoritative_facts": [
    "short fact directly observed in the task, data, or execution output"
  ],
  "verified_policies": [
    "short computation policy explicitly required by the task or objectively verified by an executed check"
  ],
  "conflicts": [
    {{
      "conflict_id": "conflict_1",
      "decision_point": "short description of the disputed decision",
      "options": [
        {{
          "option_id": "A",
          "policy": "first plausible policy",
          "supporting_evidence": ["task wording or directly observed evidence supporting this option"]
        }},
        {{
          "option_id": "B",
          "policy": "second plausible policy",
          "supporting_evidence": ["task wording or directly observed evidence supporting this option"]
        }}
      ],
      "provisional_choice": "option_id currently favored, or an empty string",
      "status": "unresolved | provisional | verified",
      "verification": "small side-by-side or direct check that can distinguish the options"
    }}
  ]
}}
```

Rules:
- `authoritative_facts` contain direct observations only, never an interpretation chosen from competing policies.
- `verified_policies` contain only policies explicitly fixed by task wording or supported by an executed result that directly distinguishes the alternatives.
- Descriptive evidence is not an operational instruction. Unresolved uncertainty or observed irregularity must not become a verified policy or trusted `expected_policy`. Keep it contested unless the task, documentation, schema semantics, or executed verification explicitly supports the policy.
- Agreement between summaries, model confidence, plausibility, or a semantic preference is not objective verification.
- Every decision with multiple plausible interpretations that could change downstream computation must remain in `conflicts`.
- Preserve every plausible option in its conflict. Never delete or suppress an option because another option is currently favored.
- Use `provisional` when one option appears better supported but has not been objectively verified. A provisional choice is not trusted evidence.
- Use `verified` only when the supplied task/evidence contains an explicit distinguishing verification result. Also place the verified policy in `verified_policies`, while retaining the conflict as an audit record.
- Use `unresolved` when the evidence does not support a defensible preference.
- `provisional_choice` must reference an `option_id` present in the same conflict, or be empty.
- Keep each option concise and semantically parallel so it can later be scored without length or wording artifacts.

# TASK #
{task}

# DETERMINISTICALLY MERGED LOCAL SUMMARIES #
{merge_state}

# LOCAL SUMMARIES #
{summaries}

JSON:"""


EXPLORATION_FINAL_SUMMARY_PROMPT = """Generate structured evidence anchors for downstream code generation and code comparison.

Use ONLY `authoritative_facts` and `verified_policies` from the evidence payload below. Conflicts, provisional choices, unsupported interpretations, and unverified alternatives must not become anchors. Do not reclassify evidence, resolve conflicts, infer omitted task requirements, or add facts.
`blocked_conflicts_not_trusted` is included only as a negative guardrail. It is explicitly not trusted evidence. Do not turn its decision points, options, provisional choices, or supporting evidence into `expected_policy`, `required_policy`, `forbidden_policy`, `scope_rule`, `included_scope`, `excluded_scope`, or `expected_grain` anchors.
Do not turn descriptive observations into required computation policies.
An anchor may state a required policy only when it is directly supported by the task wording, schema semantics, documentation, or executed verification; otherwise keep it as a fact or uncertainty.

Return a machine-readable block named exactly `STRUCTURED_EVIDENCE_ANCHORS:` followed by valid JSON. Use only the four core anchor families below plus `additional_decision_anchors`; use empty lists when no anchor applies. Each anchor must include a stable `anchor_id` and a short `source_observation` copied or paraphrased from trusted evidence. Put task-critical counts or other invariants in `additional_decision_anchors` with `decision_type: "count_invariant"` or another appropriate decision type.

```json
{{
  "source_scope_anchors": [
    {{
      "anchor_id": "source_scope_1",
      "decision_point": "which files, tables, sources, shards, or time ranges are in scope",
      "required_sources": ["file_or_source_that_must_be_used"],
      "forbidden_sources": ["file_or_source_that_should_not_be_used_if_any"],
      "scope_rule": "brief rule for selecting the correct source scope",
      "source_observation": "observed file list/schema/task wording supporting this source scope"
    }}
  ],
  "filter_scope_anchors": [
    {{
      "anchor_id": "filter_scope_1",
      "decision_point": "which rows/records/entities/items/cases should be included or excluded",
      "included_scope": "rows/records/entities/items/cases that must remain in scope",
      "excluded_scope": "rows/records/entities/items/cases or extra filters that should not be applied if any",
      "source_observation": "observed rows/counts/task wording supporting this filter scope"
    }}
  ],
  "field_semantics_anchors": [
    {{
      "anchor_id": "field_semantics_1",
      "decision_point": "how a field, tag, category, or special value changes computation",
      "field": "column_or_field_name",
      "value": "special_value_if_any",
      "required_policy": "meaning of this field/value for computation",
      "forbidden_policy": "ignore this field/value or interpret it the opposite way",
      "source_observation": "observed representative rows or task wording supporting this semantics"
    }}
  ],
  "aggregation_grain_anchors": [
    {{
      "anchor_id": "aggregation_grain_1",
      "decision_point": "what unit/grain the computation should operate on",
      "expected_grain": "one row/entity/group/time unit per ...",
      "forbidden_grain": "wrong grain or wrong operation order if observed",
      "source_observation": "observed grouping/entity/task wording supporting this grain"
    }}
  ],
  "additional_decision_anchors": [
    {{
      "anchor_id": "additional_1",
      "decision_type": "invariant | threshold | count_invariant | join_key | parsing_policy | missingness_policy | unit_scale | formula | output_policy | validation_policy | dependency_policy | other",
      "decision_point": "task-specific decision not covered by the four core anchors",
      "expected_policy": "evidence-supported policy; for count_invariant, include the exact count and what population it counts",
      "forbidden_policy": "plausible wrong policy if observed; for count_invariant, describe narrower/broader populations that would change the count",
      "source_observation": "observation/task wording supporting this decision"
    }}
  ]
}}
```

# EVIDENCE PAYLOAD #
{trusted_evidence}

STRUCTURED_EVIDENCE_ANCHORS:"""


CURRENT_DECISION_EXTRACTION_PROMPT = """Extract the current computational decisions made by the code, then align them to the structured evidence anchors.

Return only two machine-readable blocks in this order:
1. `CURRENT_DECISION_ANCHORS:` followed by valid JSON describing what the code actually does.
2. `ANCHOR_ALIGNMENT:` followed by valid JSON comparing every evidence anchor against the code/current decisions.

Do not solve the task yourself and do not decide the final answer. Compare code behavior against the structured evidence anchors. The expected policy / alternative path must come only from the structured anchor fields, not from your own inference. If the action/code is only inspecting data, listing files, previewing rows, or searching for evidence, do not treat the observed output as an operational computation policy. Use empty lists when no current decision applies. For every evidence anchor_id visible in the anchor block, produce exactly one `anchor_checks` item. Do not omit anchors. Never mark `match` unless code or observation explicitly supports the same policy. If code/observation does not reveal whether the policy is implemented, set `current_policy` to "not visible from code/observation" and mark `missing`. For invariant-style anchors, inspect both code and observation for the actual value, scope, population, object set, count, or policy used; mark `conflict` when it differs from the evidence-supported invariant, and mark `missing` when the code/observation hides the invariant instead of assuming it matches. Use `missing` only when the current policy is genuinely not visible; if the code implies a different policy from evidence, mark `conflict`, not `missing`.

Probability contrast safety rule: only mark a check as a valid conflict when `current_policy` and `expected_policy` answer the same `anchor_id`, same `decision_type`, and same evidence decision point. If the code excerpt implements a different decision dimension (for example field filtering vs aggregation order), do not force a comparison; mark the anchor `missing` and set `same_anchor_policy=false`, `comparison_valid=false`.

```json
{{
  "source_scope_decisions": [
    {{
      "current_policy": "which files/tables/sources/time ranges the code uses",
      "source_code": "short code excerpt supporting this extraction"
    }}
  ],
  "filter_scope_decisions": [
    {{
      "current_policy": "which rows/records/entities/items/cases the code includes or excludes",
      "source_code": "short code excerpt supporting this extraction"
    }}
  ],
  "field_semantics_decisions": [
    {{
      "field": "column_or_field_name_if_visible",
      "value": "special_value_if_visible",
      "current_policy": "how the code interprets or ignores this field/value",
      "source_code": "short code excerpt supporting this extraction"
    }}
  ],
  "aggregation_grain_decisions": [
    {{
      "current_grain": "unit/grain and operation order used by the code",
      "source_code": "short code excerpt supporting this extraction"
    }}
  ],
  "additional_decision_anchors": [
    {{
      "decision_type": "invariant | threshold | count_invariant | join_key | parsing_policy | missingness_policy | unit_scale | formula | output_policy | validation_policy | dependency_policy | other",
      "current_policy": "code's current policy for this decision; for count_invariant, include the count/population used if visible",
      "source_code": "short code excerpt supporting this extraction"
    }}
  ]
}}
```

Then output `ANCHOR_ALIGNMENT:` followed by:

```json
{{
  "anchor_checks": [
    {{
      "anchor_id": "evidence anchor id if available",
      "decision_type": "source_scope | filter_scope | field_semantics | aggregation_grain | invariant | threshold | count_invariant | join_key | parsing_policy | missingness_policy | unit_scale | formula | output_policy | validation_policy | dependency_policy | other",
      "evidence_decision_point": "decision point from the evidence anchor",
      "expected_policy": "what the evidence anchor says should happen",
      "current_policy": "what the code currently does; use 'missing from code' if absent",
      "status": "match | conflict | missing",
      "severity": "low | medium | high",
      "source_code": "short exact code excerpt if visible",
      "reason": "brief reason for the status",
      "same_anchor_policy": true,
      "comparison_valid": true
    }}
  ],
  "alignment_coverage": {{
    "evidence_anchor_count": 0,
    "checked_anchor_count": 0,
    "unaddressed_anchor_ids": [],
    "missing_current_policy_anchor_ids": [],
    "alignment_complete": true
  }}
}}
```

Coverage rules:
- Count all evidence anchors across `source_scope_anchors`, `filter_scope_anchors`, `field_semantics_anchors`, `aggregation_grain_anchors`, and `additional_decision_anchors`.
- `checked_anchor_count` must count distinct evidence anchor_ids represented in `anchor_checks`.
- If any anchor_id is not represented in `anchor_checks`, put it in `unaddressed_anchor_ids` and set `alignment_complete` to false.
- If an anchor is checked but the current policy is not visible from code/observation, include that anchor_id in `missing_current_policy_anchor_ids`.
- `alignment_complete` is true only when every evidence anchor has one check and no anchor is unaddressed.

# TASK #
{task}

# STRUCTURED EVIDENCE ANCHORS #
{evidence}

# CODE #
{code}

# OBSERVATION #
{observation}

CURRENT_DECISION_ANCHORS:"""


OBSERVATION_USE_JUDGE_PROMPT = """Judge whether the current observation produced by executing the action is consistent with the model's thought/action.

Return ONLY valid JSON:
{{
  "status": "pass | warning | retry",
  "ignored_key_facts": [],
  "reason": "brief explanation"
}}

Use these labels:
- pass: the observation is relevant to and consistent with the intended thought/action.
- warning: the observation provides incomplete or weak evidence that the intended action succeeded, but does not clearly contradict it.
- retry: the observation clearly contradicts the thought/action, shows that a different operation occurred, or demonstrates that the intended result was not produced.

Do not judge whether the final answer is correct. Judge only whether the execution observation is consistent with the intended thought/action.

# TASK #
{task}

# THOUGHT #
{thought}

# ACTION / CODE EXCERPT #
{action}

# OBSERVATION #
{observation}

JSON:"""


RETRY_REVIEW_PROMPT_BLOCK = """# RETRY REVIEW #
Source task: {task_id}
Source step: {step_id}
Task retry count used: {retry_budget_used}
Retry chain root step: {retry_chain_root_step_id}
Retry chain used: {retry_chain_used}/{retry_chain_limit}
Context: {prior_mode}

# OBJECTIVE CONFIDENCE SUMMARY #
{objective_summary}

# WHY THIS STEP NEEDS REVIEW #
{issues}

# CURRENT PATH TO REVIEW #
{current_path}

# OBSERVATION FROM THE SOURCE STEP #
{observation}

# INSTRUCTIONS #
This is a review/retry step inserted by the agent.

- The previous step has objective risk and must be reviewed before it is trusted.
- It may still be correct; do not discard it automatically.
- Do not reuse its output as final evidence unless it survives targeted code verification.
- You may run exploration code and computation code.
- Prefer the smallest executable checks that validate the risky decision and print the decisive intermediate facts.
- When probability contrast lists an alternative path, implement the current and alternative decisions side by side and print the facts and outputs that distinguish them.
- Do not rely only on verbal reasoning.
- If the current path survives verification, keep it and state why; otherwise repair or recompute it.
- After verification, continue solving the original task.
"""
