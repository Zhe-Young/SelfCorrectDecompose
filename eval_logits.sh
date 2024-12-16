tasks=('mmlu' 'boolq' 'commonsense_qa')
model_name='llama3_8b_chat'
model_path='/local/path/to/your/model'
for task in ${tasks[@]}
do
    python eval_logits.py \
    --model_name $model_name \
    --model_path $model_path \
    --task $task

    python analysis_logits.py \
    --model_name $model_name \
    --task $task
done