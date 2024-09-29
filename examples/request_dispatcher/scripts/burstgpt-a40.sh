#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3-mini-128k-instruct"
    local arrival="burstgpt"
    local time_index=$2
    echo "--------------- [BurstGPT] Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --preemption-mode swap \
        --swap-space 30 \
        --tensor-parallel-size 2 &

    sleep 45
    
# ================================== code: short-long  ================================ 
    python request_dispatcher.py --model "$model_name" \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --time-range 'hour' \
        --time-index "$time_index" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 
    # id: 385 - 943 requests 
    # id: 206 - 922 requests  
    # id: 538 - 1075 requests

# run_model "fcfs" 552 
run_model "qoe-avg" 552

run_model "fcfs" 1213 
run_model "qoe-avg" 1213

run_model "fcfs" 1273 
run_model "qoe-avg" 1273




#  >>>>>>> 2024-09-26 04:59-microsoft-Phi-3-mini-128k-instruct-sharegpt-multi-burstgpt*1103-1.0(0.2)-hour-552-fcfs.json (1103 requests) <<<<<<<<<
# Avg Qoe: 0.94, Perfect Qoe: 0.44. Throughput: 0.03 req/s. TTFT 1.52 s. Pause frequency: 0.04. Avg response 410.44. 
# Total time taken: 3601.546678543091. 
#  >>>>>>> 2024-09-27 02:16-microsoft-Phi-3-mini-128k-instruct-sharegpt-multi-burstgpt*1103-1.0(0.2)-hour-552-qoe-avg.json (1103 requests) <<<<<<<<<
# Avg Qoe: 0.94, Perfect Qoe: 0.44. Throughput: 0.03 req/s. TTFT 1.53 s. Pause frequency: 0.04. Avg response 409.97. 
