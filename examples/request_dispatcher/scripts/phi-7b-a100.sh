#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3-mini-128k-instruct"  
    local arrival="gamma"
    local preemption_freq=0.2
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-seqs 512 \
        --max-num-batched-tokens 200000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --preemption_freq "$preemption_freq" \
        --tensor-parallel-size 4 & 
    
    sleep 66
    
# ==================================   arxiv  ================================  

    python request_dispatcher.py --model "$model_name" \
        --num-requests 500 \
        --arrival-rate 0.3 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.075 \
        --prompt-trace arxiv   
        
# ==================================  sharegpt-multi  ================================ 
  
    python request_dispatcher.py --model "$model_name" \
        --num-requests 1000 \
        --arrival-rate 3 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.05 \
        --prompt-trace sharegpt-multi    
        
# ================================== code ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 1000 \
        --arrival-rate 0.6 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --burst 0.003 \
        --scheduling "$SCHEDULE" \
        --prompt-trace code 

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done running $SCHEDULE for $model_name"
} 


run_model "fcfs"
sleep 10
run_model "qoe-avg"

