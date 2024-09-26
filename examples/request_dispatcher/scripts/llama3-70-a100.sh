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
        --preemption-mode swap \
        --swap-space 30 \
        --tensor-parallel-size 8 &

    sleep 100
    
# ==================================   arxiv  ================================ 
#  >>>>>>> 2024-09-24 19:31-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.06(0.2)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.92, Perfect Qoe: 0.04. Throughput: 0.00 req/s. TTFT 5.29 s. Pause frequency: 0.21. Avg response 615.00. 
# Total time taken: 5332s. 
#  >>>>>>> 2024-09-25 23:37-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.05(0.2)-day--1-qoe-avg.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.92, Perfect Qoe: 0.04. Throughput: 0.00 req/s. TTFT 4.99 s. Pause frequency: 0.27. Avg response 614.71. 
# Total time taken: 6397.741274595261. 

# --------------- Start fcfs for meta-llama/Meta-Llama-3.1-70B ----------
#  >>>>>>> 2024-09-26 06:01-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.2(1.0)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.67, Perfect Qoe: 0.02. Throughput: 0.02 req/s. TTFT 39.98 s. Pause frequency: 0.40. Avg response 612.35. 
# Total time taken: 1257.3921506404877. 
#  >>>>>>> 2024-09-26 06:22-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.2(0.6)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.85, Perfect Qoe: 0.04. Throughput: 0.02 req/s. TTFT 13.88 s. Pause frequency: 0.34. Avg response 615.17. 
# Total time taken: 1478.2505400180817. 
#  >>>>>>> 2024-09-26 06:47-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.2(0.2)-day--1-fcfs.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.79, Perfect Qoe: 0.03. Throughput: 0.02 req/s. TTFT 21.98 s. Pause frequency: 0.49. Avg response 614.41. 
# Total time taken: 1623.7522420883179. 
# --------------- Start qoe-avg for meta-llama/Meta-Llama-3.1-70B ----------
#  >>>>>>> 2024-09-26 07:16-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.2(1.0)-day--1-qoe-avg.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.44, Perfect Qoe: 0.01. Throughput: 0.02 req/s. TTFT 13.56 s. Pause frequency: 4.93. Avg response 612.93. 
# Total time taken: 2461.975613117218. 
#  >>>>>>> 2024-09-26 07:57-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.2(0.6)-day--1-qoe-avg.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.52, Perfect Qoe: 0.00. Throughput: 0.01 req/s. TTFT 13.61 s. Pause frequency: 4.92. Avg response 615.37. 
# Total time taken: 2328.709089756012. 
#  >>>>>>> 2024-09-26 08:37-meta-llama-Meta-Llama-3.1-70B-arxiv-gamma*249-0.2(0.2)-day--1-qoe-avg.json (250 requests) <<<<<<<<<
# Avg Qoe: 0.48, Perfect Qoe: 0.00. Throughput: 0.01 req/s. TTFT 12.80 s. Pause frequency: 4.92. Avg response 614.51. 
# Total time taken: 2715.1321806907654. 


    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.2 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 1 \
        --prompt-trace arxiv 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.2 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.6 \
        --prompt-trace arxiv 

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 0.2 \
        --max-tokens 50000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --burst 0.2 \
        --prompt-trace arxiv 
# ==================================  sharegpt-multi  ================================ 
# =====================================  0.8-0.2  ================================ 
#  >>>>>>> 2024-09-25 02:07-meta-llama-Meta-Llama-3.1-70B-sharegpt-multi-gamma*499-0.8(0.2)-day--1-fcfs.json (500 requests) <<<<<<<<<
# Avg Qoe: 0.93, Perfect Qoe: 0.43. Throughput: 0.07 req/s. TTFT 1.61 s. Pause frequency: 0.10. Avg response 414.91. 
# Total time taken: 725.
#  >>>>>>> 2024-09-26 01:24-meta-llama-Meta-Llama-3.1-70B-sharegpt-multi-gamma*499-0.8(0.2)-day--1-qoe-avg.json (500 requests) <<<<<<<<<
# Avg Qoe: 0.93, Perfect Qoe: 0.44. Throughput: 0.07 req/s. TTFT 1.50 s. Pause frequency: 0.23. Avg response 404.86. 
# Total time taken: 717. 

# =====================================  0.8-0.4  ================================ 
#  >>>>>>> 2024-09-26 03:44-meta-llama-Meta-Llama-3.1-70B-sharegpt-multi-gamma*499-0.8(0.4)-day--1-fcfs.json (500 requests) <<<<<<<<<
# Avg Qoe: 0.94, Perfect Qoe: 0.51. Throughput: 0.07 req/s. TTFT 1.57 s. Pause frequency: 0.19. Avg response 416.32. 
#  >>>>>>> 2024-09-26 04:46-meta-llama-Meta-Llama-3.1-70B-sharegpt-multi-gamma*499-0.8(0.4)-day--1-qoe-avg.json (500 requests) <<<<<<<<<
# Avg Qoe: 0.94, Perfect Qoe: 0.53. Throughput: 0.07 req/s. TTFT 1.52 s. Pause frequency: 0.17. Avg response 416.32. 

# =====================================  0.8-0.6  ================================ 
#  >>>>>>> 2024-09-26 03:56-meta-llama-Meta-Llama-3.1-70B-sharegpt-multi-gamma*499-0.8(0.6)-day--1-fcfs.json (500 requests) <<<<<<<<<
# Avg Qoe: 0.95, Perfect Qoe: 0.58. Throughput: 0.07 req/s. TTFT 0.97 s. Pause frequency: 0.11. Avg response 415.45. 
#  >>>>>>> 2024-09-26 05:02-meta-llama-Meta-Llama-3.1-70B-sharegpt-multi-gamma*499-0.8(0.6)-day--1-qoe-avg.json (500 requests) <<<<<<<<<
# Avg Qoe: 0.95, Perfect Qoe: 0.60. Throughput: 0.07 req/s. TTFT 0.95 s. Pause frequency: 0.16. Avg response 415.65. 


    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 500 \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.4 \
    #     --prompt-trace sharegpt-multi 

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 500 \
    #     --arrival-rate 0.8 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.6 \
    #     --prompt-trace sharegpt-multi 

# ================================== code ================================ 

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

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 250 \
    #     --arrival-rate 0.3 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.6 \
    #     --prompt-trace code

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 250 \
    #     --arrival-rate 0.3 \
    #     --max-tokens 50000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 1 \
    #     --prompt-trace code

    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 
 

run_model "fcfs"
sleep 10
run_model "qoe-avg"
# sleep 10

