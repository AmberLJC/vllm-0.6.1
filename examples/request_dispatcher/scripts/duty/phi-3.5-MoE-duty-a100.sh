#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3.5-MoE-instruct" # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    local arrival="duty"
    local preemption_freq=0.3
    echo "--------------- [Duty Cycle] Start $SCHEDULE $arrival ($preemption_freq)  for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-seqs 512 \
        --max-num-batched-tokens 200000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --preemption_freq "$preemption_freq" \
        --tensor-parallel-size 8 &

        # --preemption-mode swap \
        # --swap-space 20 \
    
    sleep 90
    python scripts/send.py 'health'
# # # ==================================   arxiv  ================================  

    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.45 \
    #     --max-tokens 30000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.3 \
    #     --height 2 \
    #     --prompt-trace arxiv   
# # # ==================================  sharegpt-multi  ================================ 
    # python scripts/send.py 'health'
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 2.3 \
    #     --max-tokens 30000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.3 \
    #     --height 2 \
    #     --prompt-trace sharegpt-multi      
 
 
# ==================================  code  ================================ 
    python scripts/send.py 'health'
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 1.2 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --width 0.35 \
        --height 2 \
        --scheduling "$SCHEDULE" \
        --prompt-trace code 
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 1.2 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --width 0.45 \
        --height 2 \
        --scheduling "$SCHEDULE" \
        --prompt-trace code 
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 1.2 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --width 0.4 \
        --height 2 \
        --scheduling "$SCHEDULE" \
        --prompt-trace code  
        

    # Terminate python processes
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "[Duty Cycle] Done running $SCHEDULE for $model_name"
} 


run_model "fcfs"
# sleep 5
run_model "qoe-avg"


python send.py "[Duty Cycle] Exps for $model_name"
