import json
import pandas as pd
import os
import subprocess

# Read all results from sut/domain/response_cache/
def read_results_cache(cache_path, workload):
    for filename in os.listdir(cache_path):
        if filename.startswith(workload) and filename.endswith('.json'):
            with open(os.path.join(cache_path, filename), 'r') as f:
                data = json.load(f)
                return data
    return []

def read_evaluation(eval_path, workload):
    for filename in os.listdir(eval_path):
        if filename.startswith(workload) and filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(eval_path, filename))
            return df

workloads=["archeology", "astronomy", "biomedical", "legal", "environment", "wildfire"]
sut_name=["DeepseekR1", "Llama3_3Instruct", "Qwen2_5Coder", "GPT4o", "GPTo3", "Claude35"]
sut_name=["DeepseekR1", "Llama3_3Instruct", "GPT4o", "GPTo3"]

PRUNE = True

checking = []
for results_dir in ['results_full_0', 'results_full_1', 'results_full_2']:
    for workload in workloads:
        gold_path = f'workload/{workload}.json'
        with open(gold_path, 'r') as f:
            gold_data = json.load(f)

        for sysname in sut_name:
            for variant in ['Naive', 'OneShot', 'FewShot']:
                sut = 'BaselineLLMSystem'+sysname+variant
                cache_path = f'{results_dir}/{sut}/response_cache/'
                results_cache = read_results_cache(cache_path, workload)
                eval_path = f'{results_dir}/{sut}'


                for idx, task in enumerate(results_cache):
                    try:
                        gold_task = [t for t in gold_data if t['id'] == task['task_id']][0]
                    except:
                        print("could not find gold task for ", task['task_id'])
                        breakpoint()
                        continue
                                    
                    task_id = task['task_id']
                    if not gold_task['answer_type'] == 'string_approximate':
                        continue

                    predicted = task['model_output']['answer']
                    target = gold_task['answer']

                    if PRUNE:
                        if str(predicted) == "SUT failed to answer this question.":
                            continue
                        if "* ERRORS **" in str(predicted):
                            continue

                    checking.append({
                        'results_dir': results_dir,
                        'workload': workload,
                        'sut': sut,
                        'task_id': task_id,
                        'target': target,
                        'predicted': predicted,
                    })

df_checking = pd.DataFrame(checking)
if PRUNE:
    output_path = f'llm_strings_pruned.csv'
else:
    output_path = f'llm_strings_full.csv'
df_checking.to_csv(output_path, index=False)


total_count = 0
for workload in workloads:
    gold_path = f'workload/{workload}.json'
    with open(gold_path, 'r') as f:
        gold_data = json.load(f)
    
    count_str = 0
    count_lst = 0
    for task in gold_data:
        if task['answer_type'] == 'string_approximate':
            count_str += 1
        elif task['answer_type'] == 'list_approximate':
            if isinstance(task.get('answer', [0])[0], str):
                count_lst += 1

        for subtask in task.get('subtasks', []):
            if subtask['answer_type'] == 'string_approximate':
                count_str += 1
            elif subtask['answer_type'] == 'list_approximate':
                if isinstance(subtask.get('answer', [0])[0], str):
                    count_lst += 1

    total_count += count_str + count_lst
    print(f'Workload: {workload}, string_approximate tasks: {count_str}, list_approximate tasks: {count_lst}')

print(f'Total string_approximate and list_approximate tasks across all workloads: {total_count}')