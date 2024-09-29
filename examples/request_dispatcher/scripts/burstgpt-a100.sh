#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3.5-MoE-instruct" 
    local arrival="burstgpt"
    local time_index=$2
    echo "--------------- [BurstGPT] Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-seqs 512 \
        --max-num-batched-tokens 200000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --tensor-parallel-size 8 &

    sleep 70
    
# ================================== arxiv ================================ 
    python request_dispatcher.py --model "$model_name" \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --time-range 'hour' \
        --time-index "$time_index" \
        --scheduling "$SCHEDULE" \
        --prompt-trace arxiv

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 
    # id: 496 - 5.6k requests 
    # id: 1020 - 3.7k requests   
    # 374 - 2k requests

# run_model "fcfs" 496 
# run_model "qoe-avg" 496

run_model "fcfs" 374 
run_model "qoe-avg" 374
 