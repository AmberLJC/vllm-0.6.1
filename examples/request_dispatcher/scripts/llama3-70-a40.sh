#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3-mini-128k-instruct"
    local arrival="gamma"
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --tensor-parallel-size 2 &

    sleep 40
    
# ================================== code: short-long  ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 500 \
        --arrival-rate 4 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace code


# ================================== short-long  ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 500 \
        --arrival-rate 2 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace short-long

# ==================================   arxiv  ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 100 \
        --arrival-rate 0.15 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace arxiv 


# ==================================  sharegpt-multi  ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 1000 \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi 

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 



run_model "qoe"

sleep 10

run_model "fcfs"
