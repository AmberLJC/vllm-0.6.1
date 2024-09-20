import random
import pickle
import json
import argparse
from itertools import accumulate
  

def generate_tds_requirements(
    population_distribution: list[float] = [28.04, 51.87, 11.19, 5.56, 3.33],
    speed_distribution: list[float] = [0.196, 0.230, 0.240, 0.249, 0.264],
    num_samples: int = 100  # Number of read speeds to generate
):
    random.seed(0)
    read_speeds = []  # List to store generated read speeds

    accumulated_sum = list(accumulate(population_distribution))
    
    for _ in range(num_samples):
        rand = random.uniform(0, 100)
        for i, threshold in enumerate(accumulated_sum):
            if rand <= threshold:
                read_speeds.append(speed_distribution[i])
                break
            else:
                read_speeds.append(accumulated_sum[-1])  # Add the last speed if none of the thresholds are met
    return read_speeds
 
def generate_trace_file(file_path: str = '../dataset/sharegpt/ShareGPT_V3_filtered_3000.json', 
                        output_path: str = 'sharegpt_qoe_trace.json',
                        trace = 'text',
                        ):
    # Creating the dictionary
    entries = []

    # Populate the dictionary with 10,000 entries
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Generate read speed requirements
    if trace == 'test':
        speed = [0.15 for _ in range(len(data))]
    elif trace == 'voice':
        speed = generate_tds_requirements([28.04, 51.87, 11.19, 5.56, 3.33],  [0.196, 0.230, 0.240, 0.249, 0.264], len(data))
    else:
        speed = generate_tds_requirements([79.29, 7.03,6.92,3.58,3.17], [0.31, 0.15, 0.19, 0.15, 0.13, 0.16], len(data))
 
    for i in range(len(data)): 
        entry = {
            'ttft': 1, # TODO: add trace for TTFT
            'latency': speed[i], 
        }
        # data[i]['prompt'] = data[i]['prompt'][-max_prompt_len:]
        entry.update(data[i])
        entries.append(entry)

    # Serialize the dictionary using pickle
    with open(output_path, 'w') as file:
        json.dump(entries, file, indent=4)

    print(f"{len(entries)} QoE trace file created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="../dataset/sharegpt/ShareGPT_V3_filtered_3000.json")
    parser.add_argument("--output-path", type=str, default="sharegpt_qoe_trace.json")
    parser.add_argument("--trace",  type=str, default="text", choices=['text', 'voice', 'test'])
    args = parser.parse_args()
    generate_trace_file(args.file_path, 
                        args.output_path,
                        args.trace)
    
# python text_generation_trace.py --file-path ../dataset/lima.json --output-path lima_qoe_trace.json 
# python text_generation_trace.py --file-path ../dataset/sg_90k_part1_html_cleaned_multi.json --output-path sharegpt_multi_qoe_trace.json
# python text_generation_trace.py --file-path ../dataset/sg_90k_part1_html_cleaned_multi_1000.json --output-path sharegpt_multi_1000_qoe_trace.json  
# python text_generation_trace.py --file-path ../dataset/sg_90k_part1_html_cleaned_multi_1000.json --output-path sharegpt_multi_1000_voice_qoe_trace.json --trace voice
# python text_generation_trace.py --file-path ../dataset/sharegpt/ShareGPT_V3_filtered_3000.json --output-path sharegpt_voice_qoe_trace.json --trace voice
