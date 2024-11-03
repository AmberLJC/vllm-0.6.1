#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9
export PYTHONPATH=/vllm:$PYTHONPATH
# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3-mini-128k-instruct"  
    local arrival="duty"
    local preemption_freq=0.3
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    if [ "$SCHEDULE" == "sarathi" ]; then
        vllm serve "$model_name" \
            --max-num-seqs 512 \
            --max-num-batched-tokens 200000 \
            --scheduling-strategy fcfs \
            --load-format dummy \
            --trust-remote-code \
            --enable-chunked-prefill \
            --disable-sliding-window \
            --preemption_freq "$preemption_freq" \
            --tensor-parallel-size 4 & 
    else
        vllm serve "$model_name" \
            --max-num-seqs 512 \
            --max-num-batched-tokens 200000 \
            --scheduling-strategy "$SCHEDULE" \
            --load-format dummy \
            --trust-remote-code \
            --preemption_freq "$preemption_freq" \
            --tensor-parallel-size 4 & 
    fi
    
    sleep 70
    python scripts/send.py 'health'
# ==================================   arxiv  ================================  

    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.28 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.25 \
        --height 2 \
        --prompt-trace arxiv
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.28 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.3 \
        --height 2 \
        --prompt-trace arxiv
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.28 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.35 \
        --height 2 \
        --prompt-trace arxiv
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 0.28 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.4 \
        --height 2 \
        --prompt-trace arxiv
    

# ==================================  sharegpt-multi  ================================ 
    python scripts/send.py 'health'
 
    python request_dispatcher.py --model "$model_name" \
        --arrival-rate 1.8 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --width 0.25 \
        --height 2 \
        --prompt-trace sharegpt-multi      
        
# # ================================== code ================================ 
#     python scripts/send.py 'health'
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.25 \
#         --height 2 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.3 \
#         --height 2 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.35 \
#         --height 2 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.4 \
#         --height 2 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.35 \
#         --height 2 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.45 \
#         --height 2 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.35 \
#         --height 1.2 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.35 \
#         --height 1.6 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.35 \
#         --height 2.4 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  
#     python request_dispatcher.py --model "$model_name" \
#         --arrival-rate 0.6 \
#         --max-tokens 30000 \
#         --arrival-trace "$arrival" \
#         --width 0.35 \
#         --height 2.8 \
#         --scheduling "$SCHEDULE" \
#         --prompt-trace code  


    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done running $SCHEDULE for $model_name"
} 


# run_model "fcfs"
# sleep 10
# run_model "qoe-avg"


# run_model 'lqf'
# sleep 10
run_model 'sarathi'

