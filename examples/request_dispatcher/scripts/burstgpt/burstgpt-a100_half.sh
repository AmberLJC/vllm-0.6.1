#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3.5-MoE-instruct" 
    # local model_name="meta-llama/Meta-Llama-3.1-70B"
    local arrival="burstgpt"
    local time_index=$2
    local preemption_freq=0.05
    echo "--------------- [BurstGPT] Start $SCHEDULE ($preemption_freq) for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-seqs 512 \
        --max-num-batched-tokens 200000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --preemption_freq "$preemption_freq" \
        --tensor-parallel-size 8 &


    while true; do
        response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models)

        if [ "$response" -eq 200 ]; then
            echo "vLLM Ready!"
            break
        else
            echo "Waiting for VLLM to become healthy... (HTTP $response)"
            sleep 5
        fi
    done


    python scripts/send.py 'health'

    python request_dispatcher.py --model "$model_name" \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --time-range 'half' \
        --time-index "$time_index" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi 

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    python scripts/send.py "[BurstGPT] Done running $SCHEDULE for $model_name"
}  

# run_model "fcfs" 248 # finished 
# run_model "qoe-avg" 248 



# run_model "fcfs" 322
# run_model "qoe-avg" 322 
# run_model "fcfs" 248 # -3h
# run_model "qoe-avg" 248 #-3h

run_model "fcfs" 346
run_model "qoe-avg" 346 
# cd /vllm/examples/request_dispatcher/scripts/burstgpt/; bash burstgpt-a100_half.sh 
