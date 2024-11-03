#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    # local model_name="microsoft/Phi-3.5-MoE-instruct" 
    local model_name="meta-llama/Meta-Llama-3.1-70B"
    local arrival="burstgpt"
    local time_index=$2
    local preemption_freq=0.5
    echo "--------------- [BurstGPT] Start $SCHEDULE ($preemption_freq) for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-seqs 256 \
        --max-num-batched-tokens 100000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --preemption_freq "$preemption_freq" \
        --tensor-parallel-size 8 &

    sleep 66
    
    python request_dispatcher.py --model "$model_name" \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --time-range 'hour' \
        --time-index "$time_index" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi  

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    python scripts/send.py "[BurstGPT] Done running $SCHEDULE for $model_name"
} 
    # id: 496 - 5.6k requests 
    # id: 1020 - 3.7k requests   
    # 374 - 2k requests
    # 453 - 1.7k requests
    # 1387 - 3.2k requests
    # 849 - 1.5k requests
    # 1189
    
# run_model "fcfs" 2004
# run_model "qoe-avg" 2004
  

# run_model "fcfs" 2069
# run_model "qoe-avg" 2069

  
run_model "fcfs" 2089
# run_model "qoe-avg" 2060
# run_model "qoe-avg" 2004



 
 