import argparse
import json
import numpy as np 
from generate_qoe_trace import generate_tds_requirements

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

LEN=100000

# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json

def extract_first_sen(content,
                      max_prompt_len = 2000):
    result = []
    for k, item in enumerate(content):
        prompt = '' 
        for i in range(len( item["conversations"])): 
            if i != len( item["conversations"])-1:
                prompt += item["conversations"][i]['value']
            else:
                tokens = tokenizer.tokenize(item["conversations"][i]['value'])
                response_length = len(tokens)  

        last_tokens = tokenizer.tokenize(prompt)[-max_prompt_len:]
        if len(last_tokens) < 1:
            continue
        prompt = tokenizer.convert_tokens_to_string(last_tokens) 
        d = {'id': item['id'], 
            'prompt': prompt, 
            'output_len': response_length}
        result.append(d)
        if k >= 3000:
            break
        if k%500 == 0:
            print(f"{k} rows processed.")
    print(f"{len(result)} rows multi rounds ShareGPT dataset created.")
    return result

def main(args ):
    content = json.load(open(args["in_file"], "r"))
    content = extract_first_sen(content, LEN )

    entries = []
    speed = generate_tds_requirements([79.29, 7.03,6.92,3.58,3.17], [0.31, 0.15, 0.19, 0.15, 0.13, 0.16], 500)
    i = 0
    for data in content:

        if data['prompt'] == '':
            continue
        latency = speed[i%len(speed)] if speed[i%len(speed)] < 1 else 0.2

        input_len = len(tokenizer.tokenize(data['prompt']))
        entry = {
            'ttft': max(1, input_len // 5000),
            'latency': latency,
            'output_len': int(data['output_len']),
        }
        d = {'prompt': data['prompt']}
        entry.update(d)
        entries.append(entry)
        i += 1
    with open(args['qoe_out_file'], 'w') as file:
        json.dump(entries, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--in-file", type=str, default = '/data/amberljc/prompt_dataset/sharegpt/sg_90k_part1_html_cleaned.json' )
    parser.add_argument("--out-file", type=str, default =f"/data/amberljc/prompt_dataset/sharegpt/sg_90k_part1_html_cleaned_{LEN}.json")
    parser.add_argument("--qoe-out-file", type=str, default =f"sharegpt_multi_qoe_trace.json")
    args = parser.parse_args()
    main(vars(args))
