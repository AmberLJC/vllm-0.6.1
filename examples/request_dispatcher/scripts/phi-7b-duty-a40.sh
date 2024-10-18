#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3-mini-128k-instruct"  
    local arrival="duty"
    local preemption_freq=$2
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-seqs 512 \
        --max-num-batched-tokens 200000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --preemption_freq "$preemption_freq" \
        --preemption-mode "swap" \
        --swap-space 20 \
        --tensor-parallel-size 4 & 
    
    sleep 66
      
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 2 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --duration 300 \
        --scheduling "$SCHEDULE"-"$preemption_freq" \
        --prompt-trace sharegpt-multi     
        
# ================================== code ================================ 

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 750 \
    #     --arrival-rate 0.8 \
    #     --max-tokens 10000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 10 \
    #     --prompt-trace code

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done running $SCHEDULE for $model_name on A40"
} 

run_model "fcfs" 1 
sleep 10

run_model "qoe-avg" 1  
# sleep 10  
