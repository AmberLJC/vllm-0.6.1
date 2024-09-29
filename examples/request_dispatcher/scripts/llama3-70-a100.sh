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

    sleep 100
    
# ================================================================   arxiv  ======================================================================== 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 500 \
        --arrival-rate 0.08 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.1 \
        --prompt-trace arxiv 

# ==========================================================================  sharegpt-multi  ================================================================================== 
# =====================================  0.8-0.2  ================================ 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 1000 \
        --arrival-rate 1.5 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.1 \
        --prompt-trace sharegpt-multi  

# ========================================================================== code ============================================================== 

#  >>>>>>> 2024-09-24 23:42-meta-llama-Meta-Llama-3.1-70B-code-gamma*249-1.0(0.2)-day--1-fcfs.json (249 requests) <<<<<<<<<
# Avg Qoe: 0.87, Perfect Qoe: 0.77. Throughput: 0.01 req/s. TTFT 92.79 s. Pause frequency: 0.51. Avg response 3988.04. 
# Total time taken: 4019.404539346695. 
#  >>>>>>> 2024-09-25 01:07-meta-llama-Meta-Llama-3.1-70B-code-gamma*249-0.8(0.2)-day--1-fcfs.json (249 requests) <<<<<<<<<
# Avg Qoe: 0.87, Perfect Qoe: 0.77. Throughput: 0.01 req/s. TTFT 139.59 s. Pause frequency: 0.44. Avg response 3779.67. 
# Total time taken: 3619.1349787712097. 
#  >>>>>>> ./2024-09-25 03:57-meta-llama-Meta-Llama-3.1-70B-code-gamma*249-0.4(0.2)-day--1-fcfs.json (247 requests) <<<<<<<<<
# Avg Qoe: 0.88, Perfect Qoe: 0.80. Throughput: 0.01 req/s. TTFT 73.95 s. Pause frequency: 0.40. Avg response 3292.70. 
#  >>>>>>> 2024-09-25 09:55-meta-llama-Meta-Llama-3.1-70B-code-gamma*249-0.3(0.2)-day--1-fcfs.json (249 requests) <<<<<<<<<
# Avg Qoe: 0.87, Perfect Qoe: 0.74. Throughput: 0.01 req/s. TTFT 61.45 s. Pause frequency: 0.37. Avg response 3673.43. 
# Total time taken: 3781.2018673419952. 
#  >>>>>>> 2024-09-26 01:36-meta-llama-Meta-Llama-3.1-70B-code-gamma*249-0.3(0.5)-day--1-qoe-avg.json (249 requests) <<<<<<<<<
# Avg Qoe: 0.95, Perfect Qoe: 0.73. Throughput: 0.01 req/s. TTFT 0.59 s. Pause frequency: 0.71. Avg response 3777.62. 
# Total time taken: 4325.185839176178. 


    python request_dispatcher.py --model "$model_name" \
        --num-requests 500 \
        --arrival-rate 0.5 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.2 \
        --prompt-trace code
        
    python request_dispatcher.py --model "$model_name" \
        --num-requests 500 \
        --arrival-rate 0.5 \
        --max-tokens 30000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.05 \
        --prompt-trace code


    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 
 

run_model "fcfs"
sleep 10
run_model "qoe-avg"
# sleep 10

