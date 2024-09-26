#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="01-ai/Yi-34B-200K"
    local arrival="gamma" 
    local max_tokens=24000 # control output length
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-batched-tokens 100000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --tensor-parallel-size 8 &

    sleep 60
    
# ==================================   arxiv  ================================  
    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 1 \
        --max-tokens "$max_tokens" \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace arxiv 


# ==================================  sharegpt-multi  ================================ 
 
    python request_dispatcher.py --model "$model_name" \
        --num-requests 500 \
        --arrival-rate 1 \
        --max-tokens "$max_tokens"  \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi 

# ================================== code ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 1000 \
        --arrival-rate 2 \
        --max-tokens "$max_tokens"  \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace code
 


    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 

run_model "fcfs"

# sleep 10

# run_model "qoe"
