import json
import numpy as np
np.random.seed(0)

# read trace file
def alter_sharegpt_trace_file(file_path: str, output_file_path: str):   
    with open(file_path, 'r') as file:
        data = json.load(file)
    output_len = np.random.normal(10203, 1230, len(data))
    for i, trace in enumerate(data):
        trace['output_len'] = int(output_len[i])
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)


alter_sharegpt_trace_file('sharegpt_qoe_trace.json', 'short_long_qoe_trace.json')