#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1   
    # local model_name="meta-llama/Meta-Llama-3.1-70B"
    local model_name="microsoft/Phi-3.5-MoE-instruct" 
    local arrival="gamma" 
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-batched-tokens 300000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --preemption_freq 0.1 \
        --trust-remote-code \
        --tensor-parallel-size 4 &

    sleep 80 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 100 \
        --arrival-rate 0.2 \
        --max-tokens 100000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 100 \
        --prompt-trace prefill 

    # Terminate python processes
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done running $SCHEDULE for $model_name"
} 


run_model "fcfs"
# sleep 10 