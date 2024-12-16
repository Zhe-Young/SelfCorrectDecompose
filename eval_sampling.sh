tasks=('gsm8k' 'ifeval' 'humaneval')
model_name='llama3_8b_chat'
model_path='/local/path/to/your/model'
for task in ${tasks[@]}
do
    python eval_sampling.py \
    --model_name $model_name \
    --model_path $model_path \
    --task $task \
    --sampling_times 10 \
    --use_vllm

    python analysis_sampling.py \
    --model_name $model_name \
    --task $task \
    --sampling_times 10
done
