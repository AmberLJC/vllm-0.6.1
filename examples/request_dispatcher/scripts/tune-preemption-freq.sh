#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9
# # GPU blocks: 24377, # CPU blocks: 61440
# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3.5-MoE-instruct" # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    local arrival="gamma"
    local preemption_freq=$2
    echo "--------------- Start $SCHEDULE for $model_name w/ preemption_freq = $preemption_freq ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-batched-tokens 150000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --preemption-freq "$preemption_freq" \
        --tensor-parallel-size 4 &

    sleep 60
    
# ==================================   arxiv  ================================  
 

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 250 \
    #     --arrival-rate 0.05 \
    #     --max-tokens 25000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.2 \
    #     --prompt-trace arxiv  

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 250 \
    #     --arrival-rate 0.08 \
    #     --max-tokens 25000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.2 \
    #     --prompt-trace arxiv  

# ==================================  sharegpt-multi  ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 2 \
        --max-tokens 25000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi \
        --burst 0.1 
 
    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
}  
# run_model "fcfs" 0
run_model "qoe-avg" 1
run_model "qoe-avg" 6
run_model "qoe-avg" 10
run_model "qoe-avg" 4
run_model "qoe-avg" 2

