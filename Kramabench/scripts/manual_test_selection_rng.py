import json
import random

r = 20
excluded = set({1, 3, 5})
candidate_list = []
for i in range(r):
    if not i+1 in excluded:
        candidate_list.append(i+1)

chosen = random.sample(candidate_list, 10)
result = {
    "workload": "environment",
    "chosen": chosen
}
with open("environment_dr_manual_chosen.json", 'w+') as f:
    json.dump(result, f, indent=4)