#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="meta-llama/Meta-Llama-3.1-70B"
    local arrival="gamma"
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-batched-tokens 100000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --tensor-parallel-size 8 &

    sleep 60
    
# ================================== code ================================ 

#  >>>>>>> 2024-09-24 23:42-meta-llama-Meta-Llama-3.1-70B-code-gamma*249-1.0(0.2)-day--1-fcfs.json (249 requests) <<<<<<<<<
# Avg Qoe: 0.87, Perfect Qoe: 0.77. Throughput: 0.01 req/s. TTFT 92.79 s. Pause frequency: 0.51. Avg response 3988.04. 
# Total time taken: 4019.404539346695. 
#  >>>>>>> 2024-09-25 01:07-meta-llama-Meta-Llama-3.1-70B-code-gamma*249-0.8(0.2)-day--1-fcfs.json (249 requests) <<<<<<<<<
# Avg Qoe: 0.87, Perfect Qoe: 0.77. Throughput: 0.01 req/s. TTFT 139.59 s. Pause frequency: 0.44. Avg response 3779.67. 
# Total time taken: 3619.1349787712097. 


    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.6 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace code

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.4 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace code


# ==================================   arxiv  ================================ 
#  >>>>>>> 2024-09-24 19:31-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.06(0.2)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.92, Perfect Qoe: 0.04. Throughput: 0.00 req/s. TTFT 5.29 s. Pause frequency: 0.21. Avg response 615.00. 
# Total time taken: 5332s. 

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 250 \
    #     --arrival-rate 0.06 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --prompt-trace arxiv 


# ==================================  sharegpt-multi  ================================ 

#  >>>>>>> 2024-09-25 02:07-meta-llama-Meta-Llama-3.1-70B-sharegpt-multi-gamma*499-0.8(0.2)-day--1-fcfs.json (500 requests) <<<<<<<<<
# Avg Qoe: 0.93, Perfect Qoe: 0.43. Throughput: 0.07 req/s. TTFT 1.61 s. Pause frequency: 0.10. Avg response 414.91. 
# Total time taken: 725.2598924636841. 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.8 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi 

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 


run_model "fcfs"

sleep 10

run_model "qoe"
