#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="meta-llama/Meta-Llama-3.1-70B"
    local arrival="duty" 
    local preemption_freq=0.1
    echo "--------------- [Duty Cycle] Start $SCHEDULE for $model_name ----------" >> results.log

    if [ "$SCHEDULE" == "sarathi" ]; then
        vllm serve "$model_name" \
            --max-num-batched-tokens 100000 \
            --scheduling-strategy fcfs \
            --load-format dummy \
            --preemption_freq "$preemption_freq" \
            --enable-chunked-prefill \
            --disable-sliding-window \
            --tensor-parallel-size 8 &

    else 
        vllm serve "$model_name" \
            --max-num-batched-tokens 100000 \
            --scheduling-strategy "$SCHEDULE" \
            --load-format dummy \
            --preemption_freq "$preemption_freq" \
            --tensor-parallel-size 8 &
    fi


    sleep 60
    
# ================================================================   arxiv  ======================================================================== 
    # python scripts/send.py 'health'
  
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.1 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.25 \
        --height 2 \
        --prompt-trace arxiv  
   
 
# ==========================================================================  sharegpt-multi  ================================================================================== 
    python scripts/send.py 'health'

    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.25 \
    #     --height 2 \
    #     --prompt-trace sharegpt-multi    
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.3 \
    #     --height 2 \
    #     --prompt-trace sharegpt-multi    
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.35 \
    #     --height 2 \
    #     --prompt-trace sharegpt-multi   
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.4 \
    #     --height 2 \
    #     --prompt-trace sharegpt-multi   
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.45 \
    #     --height 2 \
    #     --prompt-trace sharegpt-multi   

    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.35 \
    #     --height 1.2 \
    #     --prompt-trace sharegpt-multi   
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.35 \
    #     --height 1.6 \
    #     --prompt-trace sharegpt-multi   
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.35 \
    #     --height 2.4 \
    #     --prompt-trace sharegpt-multi   
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.35 \
    #     --height 2.8 \
    #     --prompt-trace sharegpt-multi   
        
# ========================================================================== code ============================================================== 
    # python scripts/send.py 'health'
    # python request_dispatcher.py --model "$model_name" \
    #     --arrival-rate 0.2 \
    #     --max-tokens 30000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --width 0.25 \
    #     --height 2 \
    #     --prompt-trace code  
    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done running $SCHEDULE for $model_name"
} 
 

# run_model "lqf"
# # sleep 10
run_model "sarathi"


