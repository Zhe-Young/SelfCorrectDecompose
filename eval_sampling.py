from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from tqdm import tqdm
import json
import torch
import argparse
import os
from vllm import LLM, SamplingParams

from check.check import check_correctness

global sampling_params


def get_examples(task):
    test_examples = []
    with open(f'./data/{task}.jsonl') as f:
        for line in f.readlines():
            test_examples.append(json.loads(line))
    return test_examples


def generate_answer(use_vllm,model, tokenizer, datum, task, sampling_times=1):

    if task=='ifeval':
        q1 = datum['prompt']
    elif task=='humaneval':
        q1 = datum['instruction']
    else:
        q1 = datum['question']
    
    a1_list = generate_answer1(use_vllm, model, tokenizer, q1, sampling_times)
    a2_list = generate_answer2(use_vllm, model, tokenizer, q1,q1, a1_list)
    return a1_list,a2_list




def generate_answer1(use_vllm, model, tokenizer, q1, sampling_times=1):

    messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {"role": "user", "content": q1}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    a1_list = generate(use_vllm,model,tokenizer,[prompt],num_return_sequences=sampling_times)

    return a1_list

def generate_answer2(use_vllm, model, tokenizer, q1,q2, a1_list):
    prompts = []
    for a1 in a1_list:
        messages=[
            {"role": "system", "content": "You are a helpful assistant!"},
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": q2},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    a2_list = generate(use_vllm,model,tokenizer,prompts,num_return_sequences=1)


    return a2_list

def generate(use_vllm,model,tokenizer,prompts,num_return_sequences=1):
    answer_list = []
    if use_vllm:
        if num_return_sequences!=1:
            new_prompts = []
            for prompt in prompts:
                new_prompts+=[prompt]*num_return_sequences
            prompts = new_prompts
        global sampling_params
        result = model.generate(prompts,sampling_params)
        answer_list = [ result[i].outputs[0].text for i in range(len(prompts)) ]
    else:
        for prompt in prompts:
            toks = tokenizer([prompt], padding=False, return_tensors="pt").to(model.device)
            orig_len = toks["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **toks, max_new_tokens=1500, do_sample=True, num_return_sequences=num_return_sequences
                )
            for i in range(num_return_sequences):
                answer = tokenizer.decode(out[i,orig_len:],skip_special_tokens=True)
                answer_list.append(answer)
    return answer_list


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--sampling_times", type=int, default=1)
    parser.add_argument("--device", type=str, default='auto')
    parser.add_argument("--debug",action='store_true')
    parser.add_argument("--use_vllm",action='store_true')
    parser.add_argument("--force",action='store_true')
    args = parser.parse_args()
    model_name = args.model_name.split("/")[-1]
    model_path = args.model_path if args.model_path!='' else args.model_name


    assert args.task in ['humaneval','gsm8k','ifeval'], ''#'Only \'code\', \'math\', \'instruction_following\' are supported for args.task'
    if args.task == 'ifeval':
        import nltk
        nltk.download('punkt')
        nltk.download('punkt_tab')

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

    if args.use_vllm:
        generation_config = GenerationConfig.from_pretrained(model_path)
        dic = generation_config.to_dict()
        global sampling_params
        sampling_params = SamplingParams(temperature=dic['temperature'], top_k=dic['top_k'], top_p=dic['top_p'], max_tokens=2048, stop=[tokenizer.eos_token,"<|eot_id|>",'[/INST]','<|system|>','<|user|>','<|assistant|>'])  

        os.environ["VLLM_USE_MODELSCOPE"] = "false"
        model = LLM(
            model=model_path,
            dtype='bfloat16',
            tensor_parallel_size=torch.cuda.device_count(),
            disable_custom_all_reduce=True,
            enforce_eager=True,
            trust_remote_code=True
        )
    elif args.device!='auto':
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,trust_remote_code=True)
        device = torch.device(args.device)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map='auto',trust_remote_code=True)

    print("Model Loaded")

    test_examples = get_examples(args.task)
    if args.debug:
        test_examples = test_examples[:5]

    output_path = f"./log/{model_name.replace('.','_')}_{args.task}_sampling{args.sampling_times}{'_debug' if args.debug else ''}.jsonl"
    if not os.path.exists(output_path):
        f = open(output_path,'w')
        f.close()

    if args.force:
        write_mode = 'w'
    else:
        write_mode = 'a'
        processed_keys = set()
        with open(output_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                data = json.loads(line)
                processed_keys.add(data['id'])
        test_examples = [d for d in test_examples if d['id'] not in processed_keys]

    with open(output_path, write_mode, encoding='utf-8') as writer:
        for datum in tqdm(test_examples,desc=f'{args.task}'):
            a1_list,a2_list = generate_answer(args.use_vllm, model, tokenizer, datum, args.task, args.sampling_times)
            for a1,a2 in zip(a1_list,a2_list):
                correct1 = check_correctness(args.task, datum, a1)
                correct2 = check_correctness(args.task, datum, a2)
                tmp = {
                    'id':datum['id'],
                    "answer1":a1,
                    'answer2':a2,
                    'correct1':correct1,
                    'correct2':correct2,
                    }
                writer.write(json.dumps(tmp,ensure_ascii=False) + '\n')
            writer.flush()        
        
     

if __name__ == "__main__":
    main()
