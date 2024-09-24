#!/bin/bash

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="meta-llama/Meta-Llama-3.1-70B"
    echo "--------------- Start $SCHEUDLING for llama3-70B ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve meta-llama/Meta-Llama-3.1-70B \
        --max-num-batched-tokens 100000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --tensor-parallel-size 8 \
        --preemption-mode swap \
        --swap-space 30 &

    sleep 80 

    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 25 \
        --arrival-rate 0.1 \
        --max-tokens 16384 \
        --arrival-trace periodic_poisson \
        --scheduling "$SCHEDULE" \
        --prompt-trace arxiv \
        --max-tokens 12800 
 
    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 25 \
        --arrival-rate 0.2 \
        --max-tokens 16384 \
        --arrival-trace periodic_poisson \
        --scheduling "$SCHEDULE" \
        --prompt-trace arxiv \
        --max-tokens 12800 

    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 100 \
        --arrival-rate 0.15 \
        --max-tokens 16384 \
        --arrival-trace gamma \
        --scheduling "$SCHEDULE" \
        --prompt-trace arxiv \
        --max-tokens 12800 
    
    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 100 \
        --arrival-rate 0.1 \
        --max-tokens 16384 \
        --arrival-trace gamma \
        --scheduling "$SCHEDULE" \
        --prompt-trace arxiv \
        --max-tokens 12800 

# ==================================    short-long  ================================ 

    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 25 \
        --arrival-rate 0.1 \
        --max-tokens 12800 \
        --arrival-trace periodic_poisson \
        --scheduling "$SCHEDULE" \
        --prompt-trace short-long
    
    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 25 \
        --arrival-rate 0.05 \
        --max-tokens 12800 \
        --arrival-trace periodic_poisson \
        --scheduling "$SCHEDULE" \
        --prompt-trace short-long 

    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 100 \
        --arrival-rate 0.2 \
        --max-tokens 12800 \
        --arrival-trace gamma \
        --scheduling "$SCHEDULE" \
        --prompt-trace short-long 

# ==================================  sharegpt-multi  ================================ 

    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 50 \
        --arrival-rate 20 \
        --max-tokens 12800 \
        --arrival-trace periodic_poisson \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi 

    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 500 \
        --arrival-rate 20 \
        --max-tokens 12800 \
        --arrival-trace gamma \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi 

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 

run_model "qoe"

sleep 30

run_model "fcfs"
