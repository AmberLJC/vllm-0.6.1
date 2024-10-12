import argparse
import json
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct") 

def analyze_length(input_file: str):
    print(f"Analyzing trace: {input_file}")
    input_len_list = []
    output_len_list = []
    i = 0
    with open(input_file, 'r') as file:
        prompt_trace = json.load(file)
        for data in prompt_trace:
            input_len = len(tokenizer.tokenize(data['prompt']))
            output_len = data['output_len']
            input_len_list.append(input_len)
            output_len_list.append(output_len)
            i += 1
            if i > 500:
                break
    print(f"Average input length: {np.mean(input_len_list)}")
    print(f"Input length std: {np.std(input_len_list)}")
    print(f"Average output length: {np.mean(output_len_list)}")
    print(f"Output length std: {np.std(output_len_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze text length using a tokenizer.")
    
    # Add arguments
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input file.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    analyze_length(args.input_file)

# python analyze_length.py --input-file code_qoe_trace.json     