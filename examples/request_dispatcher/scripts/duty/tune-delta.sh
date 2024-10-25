#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="meta-llama/Meta-Llama-3.1-70B"
    local arrival="duty" 
    local real_delta=$2
    echo "--------------- [Tune Delta-t=$2 ] Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-batched-tokens 100000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --preemption_freq "$real_delta" \
        --tensor-parallel-size 8 &

    sleep 60
    
# ==========================================================================  sharegpt-multi  ================================================================================== 
    python scripts/send.py 'health'
 
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.35 \
        --height 2 \
        --prompt-trace sharegpt-multi    

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done running tune delta-t"
} 
 

run_model "qoe-avg" 10
run_model "qoe-avg" 30
run_model "qoe-avg" 20
run_model "qoe-avg" 70
run_model "qoe-avg" 90
run_model "qoe-avg" 110
run_model "qoe-avg" 50

