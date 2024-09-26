#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3.5-MoE-instruct" # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    local arrival="gamma"
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-batched-tokens 150000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --preemption-mode swap \
        --swap-space 30 \
        --tensor-parallel-size 8 &

    sleep 60
    
 
# ==================================   arxiv  ================================  

#  >>>>>>> 2024-09-25 05:42-microsoft-Phi-3.5-MoE-instruct-arxiv-gamma*249-1.0(0.2)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.92, Perfect Qoe: 0.12. Throughput: 0.07 req/s. TTFT 6.15 s. Pause frequency: 0.28. Avg response 601.65. 
# Total time taken: 362.20244789123535. 
    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 1 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace arxiv 


# ==================================  sharegpt-multi  ================================ 
 
#  >>>>>>> 2024-09-25 05:48-microsoft-Phi-3.5-MoE-instruct-sharegpt-multi-gamma*249-10.0(0.2)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.90, Perfect Qoe: 0.24. Throughput: 0.06 req/s. TTFT 1.31 s. Pause frequency: 0.10. Avg response 417.78. 
# Total time taken: 393.9879024028778. 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 10 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi 


# ================================== code ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 1000 \
        --arrival-rate 1.5 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace code

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 


# run_model "fcfs"
# sleep 10
run_model "qoe-avg"
