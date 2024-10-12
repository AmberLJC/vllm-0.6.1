import os
import json
from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-200K")
from generate_qoe_trace import generate_tds_requirements


def read_all_code_questions(path_to_code = '/data/amberljc/prompt_dataset/APPS/train/'):
    directories = [d for d in os.listdir(path_to_code) if os.path.isdir(os.path.join(path_to_code, d))]

    for directory in directories:
        files = [f for f in os.listdir(path_to_code+directory) if os.path.isfile(os.path.join(path_to_code+directory, f))]
        prompt = ''
        response_length = 0
        for file in files:
            if file == 'question.txt' or file == 'input_output.json' or file == 'starter_code.py':
                with open(path_to_code+directory+'/'+file, 'r') as f:
                    prompt += f.read()
            elif file == 'solutions.json':
                with open(path_to_code+directory+'/'+file, 'r') as f:
                    response_length = len(tokenizer.tokenize(f.read()))
        # print(f"Prompt: {prompt}, Response Length: {response_length}")
        question = {'prompt': prompt, 'output_len': response_length}
        yield question

def generate_qoe_app(output_path='code_qoe_trace.json'):
    entries = []
    speed = generate_tds_requirements([79.29, 7.03,6.92,3.58,3.17], [0.31, 0.15, 0.19, 0.15, 0.13, 0.16], 500)
    i = 0
    for data in read_all_code_questions():
        latency = speed[i%len(speed)] if speed[i%len(speed)] < 1 else 0.2
        entry = {
            'ttft': 1, # TODO: add trace for TTFT
            'latency': latency,
            'output_len': int(data['output_len']),
        }
        if data['prompt'] == '':
            continue
        d = {'prompt': data['prompt']}
        entry.update(d)
        entries.append(entry)
        i += 1
        if i % 500 == 0:
            print(f"{i} rows processed.")
    with open(output_path, 'w') as file:
        json.dump(entries, file, indent=4)

if __name__ == "__main__":
    generate_qoe_app()