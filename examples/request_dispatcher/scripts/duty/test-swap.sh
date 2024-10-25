#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="meta-llama/Meta-Llama-3.1-70B"
    local arrival="duty" 
    echo "--------------- [Test Swap] Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-batched-tokens 100000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --preemption-freq 0.1 \
        --preemption-mode swap \
        --swap-space 30 \
        --tensor-parallel-size 8 &

    sleep 90
    
    
# ==========================================================================  sharegpt-multi  ================================================================================== 
    python scripts/send.py 'health'

    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.25 \
        --height 2 \
        --prompt-trace sharegpt-multi     
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.3 \
        --height 2 \
        --prompt-trace sharegpt-multi     
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.35 \
        --height 2 \
        --prompt-trace sharegpt-multi     
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.4 \
        --height 2 \
        --prompt-trace sharegpt-multi     
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.45 \
        --height 2 \
        --prompt-trace sharegpt-multi          
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.35 \
        --height 2 \
        --prompt-trace sharegpt-multi     
             
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.35 \
        --height 1.2 \
        --prompt-trace sharegpt-multi          
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.35 \
        --height 1.6 \
        --prompt-trace sharegpt-multi          
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.35 \
        --height 2.4 \
        --prompt-trace sharegpt-multi          
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.35 \
        --height 2.8 \
        --prompt-trace sharegpt-multi     
    
    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done tseting swapping $SCHEDULE for $model_name"
} 


run_model "qoe-avg"


