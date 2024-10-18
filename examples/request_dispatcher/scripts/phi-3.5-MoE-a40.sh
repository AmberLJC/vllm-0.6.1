#!/bin/bash
ps aux | grep python | awk '{print $2}' | xargs -r kill -9
# # GPU blocks: 24377, # CPU blocks: 61440
export PYTHONPATH=/vllm:$PYTHONPATH

run_model() {
    cd /vllm/examples/request_dispatcher 
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    local model_name="microsoft/Phi-3.5-MoE-instruct" # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    local arrival="gamma"
    echo "--------------- Start $SCHEDULE for $model_name ----------" >> results.log

    # Start serving the model with the input scheduling strategy
    vllm serve "$model_name" \
        --max-num-seqs 512 \
        --max-num-batched-tokens 1000000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --trust-remote-code \
        --preemption-mode swap \
        --swap-space 10 \
        --preemption_freq 0.2 \
        --tensor-parallel-size 4 &

    sleep 70
    
# ==================================   arxiv  ================================  
 
#  >>>>>>> 2024-09-26 17:06-microsoft-Phi-3.5-MoE-instruct-arxiv-gamma*149-0.15(1.0)-day--1-fcfs.json (150 requests) <<<<<<<<<
# Avg Qoe: 0.91, Perfect Qoe: 0.10. Throughput: 0.02 req/s. TTFT 4.66 s. Pause frequency: 0.55. Avg response 604.62. 
# Total time taken: 955.2411255836487. 
#  >>>>>>> 2024-09-26 17:22-microsoft-Phi-3.5-MoE-instruct-arxiv-gamma*149-0.15(0.2)-day--1-fcfs.json (150 requests) <<<<<<<<<
# Avg Qoe: 0.87, Perfect Qoe: 0.03. Throughput: 0.01 req/s. TTFT 8.02 s. Pause frequency: 0.53. Avg response 612.97. 
# Total time taken: 1221.8857214450836. 

#  >>>>>>> 2024-09-27 03:51-microsoft-Phi-3.5-MoE-instruct-arxiv-gamma*149-0.15(1.0)-day--1-qoe-avg.json (150 requests) <<<<<<<<<
# Avg Qoe: 0.91, Perfect Qoe: 0.11. Throughput: 0.02 req/s. TTFT 4.69 s. Pause frequency: 0.62. Avg response 604.64. 
# Total time taken: 948.7534158229828. 
#  >>>>>>> 2024-09-27 04:07-microsoft-Phi-3.5-MoE-instruct-arxiv-gamma*149-0.15(0.2)-day--1-qoe-avg.json (150 requests) <<<<<<<<<
# Avg Qoe: 0.86, Perfect Qoe: 0.03. Throughput: 0.01 req/s. TTFT 8.05 s. Pause frequency: 0.52. Avg response 608.88. 
# Total time taken: 1223.5619299411774. 


    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 250 \
    #     --arrival-rate 0.05 \
    #     --max-tokens 25000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 0.2 \
    #     --prompt-trace arxiv  
 

# ==================================  sharegpt-multi  ================================ 
  

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

    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 1 \
        --max-tokens 25000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi \
        --burst 0.1
 
    python request_dispatcher.py --model "$model_name" \
        --num-requests 250 \
        --arrival-rate 1 \
        --max-tokens 25000 \
        --arrival-trace "$arrival" \
        --scheduling "$SCHEDULE" \
        --prompt-trace sharegpt-multi \
        --burst 10
 
 
# ================================== code ================================ 

    # python request_dispatcher.py --model "$model_name" \
    #     --num-requests 250 \
    #     --arrival-rate 3 \
    #     --max-tokens 10000 \
    #     --arrival-trace "$arrival" \
    #     --scheduling "$SCHEDULE" \
    #     --burst 10 \
    #     --prompt-trace code

    # Terminate python processes 
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
    cd /vllm/examples/request_dispatcher/scripts    
    python send.py "Done running $SCHEDULE for $model_name"
} 


# run_model "fcfs"
# sleep 10
run_model "qoe-avg"
