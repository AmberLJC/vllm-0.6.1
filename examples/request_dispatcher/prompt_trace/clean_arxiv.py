import numpy as np
import json
from generate_qoe_trace import generate_tds_requirements
file_path = '/data/amberljc/prompt_dataset/arxiv-dataset/selected_arvix.txt'
np.random.seed(0)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")


def read_arivx_data(file_path):
    # read each line of the file
    with open(file_path, 'r') as file:
        for data in file:
            data = json.loads(data)
            text = ''    
            for k, v in data.items():
                if k == 'sections' or k == 'abstract_text' or k == 'article_text':
                    for section in v: 
                        text += (str(section))
                        # break
                elif k == 'article_id' or k == 'labels' or k == 'section_names':
                    pass
                else:
                    text += str(v)
            
            yield text
                

def generate_qoe_arvix(output_path='arvix_qoe_trace.json'):
    entries = []
    speed = generate_tds_requirements([79.29, 7.03,6.92,3.58,3.17], [0.31, 0.15, 0.19, 0.15, 0.13, 0.16], 500)
    i = 0
    random_output_len = np.random.normal(600, 150, 500) 
    for data in read_arivx_data(file_path):

        latency = speed[i%len(speed)] if speed[i%len(speed)] < 1 else 0.2
        output_len = random_output_len[i%len(random_output_len)]
        
        if data == '':
            continue
        input_len = len(tokenizer.tokenize(data))
        d = {'prompt': data}
        entry = {
            'ttft': max(1, input_len // 5000),
            'latency': latency,
            'output_len': int(output_len),
        }
        entry.update(d)
        entries.append(entry)
        i += 1
    with open(output_path, 'w') as file:
        json.dump(entries, file, indent=4)

generate_qoe_arvix()