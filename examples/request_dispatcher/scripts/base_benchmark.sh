# A40 - INFO 10-13 23:43:48 distributed_gpu_executor.py:57] # GPU blocks: 19941, # CPU blocks: 6826
# ============================================================================
# ================================= fcfs  ====================================
# ============================================================================ 
  
# ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# vllm serve microsoft/Phi-3-mini-128k-instruct  --scheduling-strategy fcfs --load-format dummy  --tensor-parallel-size 4 --trust-remote-code --max-num-batched-tokens 200000  & 
# sleep 55

# cd /vllm/examples/request_dispatcher
# python request_dispatcher.py --model  microsoft/Phi-3-mini-128k-instruct  --num-requests 50 --arrival-rate 0.5 --scheduling fcfs  --max-tokens 30000 --prompt-trace arxiv
# ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# ============================================================================
# ======================== fcfs + chunked-prefill  ===========================
# ============================================================================ 
  
# ps aux | grep python | awk '{print $2}' | xargs -r kill -9
# vllm serve microsoft/Phi-3-mini-128k-instruct  --scheduling-strategy fcfs --load-format dummy  --tensor-parallel-size 4 --trust-remote-code --max-num-batched-tokens 200000 --enable-chunked-prefill --disable-sliding-window & 
# sleep 55

# cd /vllm/examples/request_dispatcher
# python request_dispatcher.py --model  microsoft/Phi-3-mini-128k-instruct  --num-requests 50 --arrival-rate 0.5 --scheduling fcfs  --max-tokens 30000 --prompt-trace arxiv
# ps aux | grep python | awk '{print $2}' | xargs -r kill -9


# ============================================================================
# ================================== qoe-avg  ================================
# ============================================================================

# ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# vllm serve microsoft/Phi-3-mini-128k-instruct  --scheduling-strategy qoe-avg --load-format dummy  --tensor-parallel-size 4 --trust-remote-code --max-num-batched-tokens 200000  --preemption-mode swap --swap-space 10 --preemption_freq 0.2 &
# sleep 65
# # 
# cd /vllm/examples/request_dispatcher
# python request_dispatcher.py --model  microsoft/Phi-3-mini-128k-instruct  --num-requests 50 --arrival-rate 0.5 --scheduling qoe-avg  --max-tokens 30000  --prompt-trace arxiv
# ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# ============================================================================
# ================================== qoe-min  ================================
# ============================================================================

ps aux | grep python | awk '{print $2}' | xargs -r kill -9

vllm serve microsoft/Phi-3-mini-128k-instruct  --scheduling-strategy qoe-min --load-format dummy  --tensor-parallel-size 4 --trust-remote-code --max-num-batched-tokens 200000  --preemption-mode swap --swap-space 10 --preemption_freq 0.5 &
sleep 60
# 
cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model  microsoft/Phi-3-mini-128k-instruct   --num-requests 50 --arrival-rate 0.5 --scheduling qoe-min  --max-tokens 30000  --prompt-trace arxiv
ps aux | grep python | awk '{print $2}' | xargs -r kill -9



# ============================================================================
# Sarahti
#  >>>>>>> 2024-10-25 21:26-microsoft-Phi-3-mini-128k-instruct-arxiv-gamma*49-0.5(10)-fcfs.json (50 requests) <<<<<<<<<
# Avg Qoe: 0.71, Perfect Qoe: 0.31. Throughput: 0.02450 req/s. TTFT 35.88 s. Pause frequency: 0.00. Avg response 614.67. 
# Total time taken: 206.93260169029236. 
# vLLM
#  >>>>>>> 2024-10-25 21:33-microsoft-Phi-3-mini-128k-instruct-arxiv-gamma*49-0.5(10)-fcfs.json (50 requests) <<<<<<<<<
# Avg Qoe: 0.76, Perfect Qoe: 0.31. Throughput: 0.03062 req/s. TTFT 25.91 s. Pause frequency: 0.00. Avg response 608.06. 
# Total time taken: 183.61004424095154. 
# Andes
#  >>>>>>> 2024-10-25 21:39-microsoft-Phi-3-mini-128k-instruct-arxiv-gamma*49-0.5(10)-qoe-avg-0.1.json (50 requests) <<<<<<<<<
# Avg Qoe: 0.89, Perfect Qoe: 0.43. Throughput: 0.02722 req/s. TTFT 15.70 s. Pause frequency: 0.06. Avg response 619.53. 
# Total time taken: 185.97827410697937. 