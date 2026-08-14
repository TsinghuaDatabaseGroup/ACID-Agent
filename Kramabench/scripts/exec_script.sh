for workload in "archeology" "astronomy" "biomedical" "legal" "environment" "wildfire"; do
for sut_name in "DeepseekR1" "Llama3_3Instruct" "Qwen2_5Coder" "GPT4o" "GPTo3"; do
 for config in "Naive" "OneShot" "FewShot"; do
#  for config in "Naive"; do
    echo "Running workload: $workload with SUT: $sut_name$config"
    python evaluate.py --sut BaselineLLMSystem$sut_name$config --workload $workload.json --dataset_name $workload --use_system_cache
done
done
done