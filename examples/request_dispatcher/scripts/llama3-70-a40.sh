#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3-mini-128k-instruct"
    local arrival="gamma"
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --preemption-mode swap \
        --swap-space 30 \
        --tensor-parallel-size 2 &

    sleep 45
    
# ================================== code: short-long  ================================ 

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 200 \
    #     --arrival-rate 3 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --prompt-trace code


# ================================== short-long  ================================ 

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 500 \
    #     --arrival-rate 2 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --prompt-trace short-long

# ==================================   arxiv  ================================ 

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 100 \
    #     --arrival-rate 0.15 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --prompt-trace arxiv 


# ==================================  sharegpt-multi  ================================ 
# throughput: 1k token / s.
# Response length: < 500 tokens
# 2 requests / s
# in fact, 2 req/s w/ gamma cause contention
# 1 req/s w/ gamma(0.2) is fine, queue can reach 75 in the end 

#  >>>>>>> 2024-09-24 21:43-microsoft-Phi-3-mini-128k-instruct-sharegpt-multi-gamma*249-0.8(0.4)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.93, Perfect Qoe: 0.61. Throughput: 0.06 req/s. TTFT 1.80 s. Pause frequency: 0.12. Avg response 414.61. 
# Total time taken: 391. 
#  >>>>>>> 2024-09-24 19:33-microsoft-Phi-3-mini-128k-instruct-sharegpt-multi-gamma*249-0.8(0.2)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.93, Perfect Qoe: 0.53. Throughput: 0.06 req/s. TTFT 1.98 s. Pause frequency: 0.12. Avg response 421.80. 
# Total time taken: 415. 
#  >>>>>>> 2024-09-24 21:37-microsoft-Phi-3-mini-128k-instruct-sharegpt-multi-gamma*249-0.8(0.1)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.86, Perfect Qoe: 0.33. Throughput: 0.07 req/s. TTFT 4.64 s. Pause frequency: 0.09. Avg response 415.65. 
# Total time taken: 368. 

# --------------- Start qoe-avg for microsoft/Phi-3-mini-128k-instruct ----------
#  >>>>>>> 2024-09-25 01:23-microsoft-Phi-3-mini-128k-instruct-sharegpt-multi-gamma*249-0.8(0.4)-day--1-qoe-avg.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.94, Perfect Qoe: 0.62. Throughput: 0.06 req/s. TTFT 1.78 s. Pause frequency: 0.19. Avg response 416.72. 
# Total time taken: 391.74069571495056. 
#  >>>>>>> 2024-09-25 01:10-microsoft-Phi-3-mini-128k-instruct-sharegpt-multi-gamma*249-0.8(0.2)-day--1-qoe-avg.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.95, Perfect Qoe: 0.60. Throughput: 0.06 req/s. TTFT 1.11 s. Pause frequency: 0.07. Avg response 419.06. 
# Total time taken: 415.3263506889343. 
#  >>>>>>> 2024-09-25 01:17-microsoft-Phi-3-mini-128k-instruct-sharegpt-multi-gamma*249-0.8(0.1)-day--1-qoe-avg.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.90, Perfect Qoe: 0.43. Throughput: 0.07 req/s. TTFT 2.35 s. Pause frequency: 0.16. Avg response 419.98. 
# Total time taken: 368.96756410598755. 


    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.8 \
        --max-tokens 25000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi \
        --burst 0.2

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.8 \
        --max-tokens 25000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi \
        --burst 0.1 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.8 \
        --max-tokens 25000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi \
        --burst 0.4

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 

# run_model "fcfs"
# 
run_model "qoe-avg"

sleep 10

run_model "qoe-min"
