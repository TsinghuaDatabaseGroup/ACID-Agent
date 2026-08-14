import csv
import os
import re
from collections import defaultdict
import json


rootPath = "./results"
useSubTasks = False
useLLMEval = True

def handleFile(file):    
    data_as_dicts = []
    with open(file, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)  # First row as header

        totalTasks = 0
        totalTaskScore = 0

        for row in reader:
            row_dict = dict(zip(headers, row))
            
            if len(row_dict["task_id"].split('-')) == 4 and not useSubTasks:
                continue
            elif len(row_dict["task_id"].split('-')) != 4 and useSubTasks:
                continue

            if useLLMEval:
                if row_dict["metric"] == "llm_code_eval":
                    tasks = row_dict["value"]
                    numTrueInTasks = tasks.count("true")
                    numFalseInTasks = tasks.count("false")
                    #print(row_dict["metric"], row_dict["task_id"], tasks, numTrueInTasks, numFalseInTasks)

                    if numTrueInTasks == 0 and numFalseInTasks == 0:
                        continue
                    else:
                        totalTaskScore += numTrueInTasks / (numTrueInTasks + numFalseInTasks)
                        totalTasks += 1
                        continue
                else:
                    continue
            
            
            if row_dict["metric"] == "success":
                totalTaskScore += float(row_dict["value"])
            elif row_dict["metric"] == "f1" or row_dict["metric"] == "f1_approximate":
                totalTaskScore += float(row_dict["value"])
            elif row_dict["metric"] == "llm_paraphrase":
                totalTaskScore += float(row_dict["value"])
            elif row_dict["metric"] == "mean_relative_absolute_error":
                totalTaskScore += 1 / ( 1 + float(row_dict["value"]) )
            else:
                continue

            totalTasks += 1
    
        if totalTasks == 0:
            return (None, 0)
        return (totalTaskScore/totalTasks, totalTasks)

if __name__ == "__main__":

    models = ["DeepseekR1", "Gemma3", "GPT4o", "GPTo3", "Llama3_3", "Qwen2_5"]
    versions = ["FewShot", "Naive", "OneShot"]

    folders = os.listdir(rootPath)

    #drop "deep-research" folder
    folders = [folder for folder in folders if folder != "deep-research" and os.path.isdir(os.path.join(rootPath, folder))]

    for folder in folders:
        fullPath = os.path.join(rootPath, folder)
        files = [f for f in os.listdir(fullPath) if f.endswith(".csv")]
        files.sort()
        
        fileByFirstKey = [i.split('_')[0] for i in files] 
        fileByFirstKey = defaultdict(list)
        for file in files:
            fileByFirstKey[file.split('_')[0]].append(file)
        
        bestFiles = {}
        for key, value in fileByFirstKey.items():
            if len(value) > 1:
                bestFiles[key] = max(value, key=lambda x: (x.split('_')[2], x.split('_')[3]))
            else:
                bestFiles[key] = value[0]

        modelInFolder = [i for i in models if i in folder]
        versionInFolder = [i for i in versions if i in folder]
        
        result_dict = {}
        for bestFile in bestFiles.values():
            if bestFile.endswith(".json_measures.csv"):
                continue
            rep_str = json.dumps(modelInFolder+versionInFolder)
            score, cnt = handleFile(os.path.join(fullPath, bestFile))
            if score is None:
                print(rep_str)
                continue
            if rep_str not in result_dict:
                result_dict[rep_str] = score * cnt
            else:
                result_dict[rep_str] += score * cnt
            print(versionInFolder, modelInFolder, bestFile, os.path.join(fullPath, bestFile), (score, cnt))
        for k, v in result_dict.items():
            print(k, 100 * v / 104)