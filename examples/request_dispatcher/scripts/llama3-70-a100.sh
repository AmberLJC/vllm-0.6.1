#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="meta-llama/Meta-Llama-3.1-70B"
    local arrival="gamma"
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-batched-tokens 100000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --tensor-parallel-size 8 &

    sleep 100
    
# ================================================================   arxiv  ======================================================================== 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 500 \
        --arrival-rate 0.08 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.2 \
        --prompt-trace arxiv 

# ==========================================================================  sharegpt-multi  ================================================================================== 
# =====================================  0.8-0.2  ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 1000 \
        --arrival-rate 1.5 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.2 \
        --prompt-trace sharegpt-multi  

# ========================================================================== code ============================================================== 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 1000 \
        --arrival-rate 0.2 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.2 \
        --prompt-trace code 

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done running $SCHEDULE for $model_name"
} 
 

# run_model "fcfs"
# sleep 10
run_model "qoe-avg"


