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
        --max-num-seqs 512 \
        --max-num-batched-tokens 200000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --tensor-parallel-size 8 &

        # --preemption-mode swap \
        # --swap-space 20 \
    
    sleep 80
    
# ==================================   arxiv  ================================  

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 200 \
    #     --arrival-rate 0.8 \
    #     --max-tokens 30000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.1 \
    #     --prompt-trace arxiv 
    
    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 200 \
    #     --arrival-rate 0.8 \
    #     --max-tokens 30000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.5 \
    #     --prompt-trace arxiv 


# ==================================  sharegpt-multi  ================================ 

# ========== TODO ==========
    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 1000 \
    #     --arrival-rate 5.5 \
    #     --max-tokens 30000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.01 \
    #     --prompt-trace sharegpt-multi 
 
    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 1000 \
    #     --arrival-rate 5.5 \
    #     --max-tokens 30000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.005 \
    #     --prompt-trace sharegpt-multi 
 

# ================================== code ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 2000 \
        --arrival-rate 1.7 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --burst 0.1 \
        --scheduling "$SCHEDULE" \
        --prompt-trace code

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 


run_model "fcfs"
sleep 10
run_model "qoe-avg"

