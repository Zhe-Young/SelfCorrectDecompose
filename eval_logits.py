from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
import json
import torch
import argparse
import os

from check.check import check_correctness



def get_examples(task):
    test_examples = []
    with open(f'./data/{task}.jsonl') as f:
        for line in f.readlines():
            test_examples.append(json.loads(line))
    return test_examples

def get_probs(model,tokenizer,prompt,choices):
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
    ).to(model.device)
    answer_encoding = tokenizer(
        choices,
        return_tensors='pt',
        add_special_tokens=False
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    assert answer_encoding.input_ids.shape[1] == 1
    logits = logits[0][-1]
    all_logits = logits[answer_encoding.input_ids.flatten()]
    probs = torch.softmax(all_logits,dim=-1)   
    return probs.tolist()

def get_answer(model, tokenizer, datum, task):

    q1 = datum['question']

    if task == 'mmlu':
        q1 = f"The following is a multiple choice question (with answers) about {datum['subject']}.\n\n"
        q1+= f"{datum['question']}\n"
        for k,v in zip(['A','B','C','D'],datum['choices']):
            q1 += f"{k}. {v}\n"
        q1+= "Please directly answer the right choice.\n"
        choices = ['A','B','C','D']
    elif task=='boolq':
        q1+= "? (please answer true or false)."
        choices = ['false','true']
    else:
        choices = ['A','B','C','D','E'][:len(datum['choices'])]
        q1 = f"The following is a multiple choice questions (with answers).\n\n"
        q1+= f"{datum['question']}\n"
        for k,v in zip(choices,datum['choices']):
            q1 += f"{k}. {v}\n"
        q1+= "Please directly answer the right choice.\n"


    messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {"role": "user", "content": q1},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    probs1 = get_probs(model,tokenizer,prompt+"The answer is:",choices)

    probs2_list = []
    for answer in choices:
        messages=[
            {"role": "system", "content": "You are a helpful assistant!"},
            {"role": "user", "content": q1},
            {"role": "assistant", "content": f"The answer is: {answer}\n"},
            {"role": "user", "content": q1},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        probs2 = get_probs(model,tokenizer,prompt+"The answer is:",choices)
        probs2_list.append(probs2)
    
    return datum,probs1,probs2_list


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--device", type=str, default='auto')
    parser.add_argument("--debug",action='store_true')
    parser.add_argument("--force",action='store_true')
    args = parser.parse_args()
    model_name = args.model_name.split("/")[-1]
    model_path = args.model_path if args.model_path!='' else args.model_name


    assert args.task in ['boolq','mmlu','commonsense_qa']
    if args.device!='auto':
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        device = torch.device(args.device)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map='auto',trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

    print("Model Loaded")

    test_examples = get_examples(args.task)
    if args.debug:
        test_examples = test_examples[:5]

    output_path = f"./log/{model_name.replace('.','_')}_{args.task}_probs{'_debug' if args.debug else ''}.jsonl"
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

            _,probs1,probs2_list = get_answer(model, tokenizer, datum, args.task)

            tmp = {
                'id':datum['id'],
                'answer': datum['answer'],
                "probs1": probs1,
                "probs2_list": probs2_list,
                }
            writer.write(json.dumps(tmp,ensure_ascii=False) + '\n')
            writer.flush()
        
     

if __name__ == "__main__":
    main()
