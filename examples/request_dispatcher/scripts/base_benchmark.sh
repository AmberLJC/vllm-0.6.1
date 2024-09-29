#  A100 - INFO 09-27 15:16:36 distributed_gpu_executor.py:57] # GPU blocks: 54126, # CPU blocks: 16384
# ============================================================================
# ==================================== fcfs  =================================
# ============================================================================ 
  
# ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# vllm serve microsoft/Phi-3.5-MoE-instruct --scheduling-strategy fcfs --load-format dummy  --tensor-parallel-size 8 --trust-remote-code --max-num-batched-tokens 200000  & 
# sleep 58
# # 
# cd /vllm/examples/request_dispatcher
# python request_dispatcher.py --model  microsoft/Phi-3.5-MoE-instruct --num-requests 150 --arrival-rate 1 --scheduling fcfs  --max-tokens 30000 --prompt-trace arxiv 
# ps aux | grep python | awk '{print $2}' | xargs -r kill -9


# ============================================================================
# ================================== qoe-avg  ================================
# ============================================================================


#  >>>>>>> 2024-09-27 00:43-microsoft-Phi-3.5-MoE-instruct-arxiv-poisson*50-0.3(0.2)-day--1-qoe-avg.json (51 requests) <<<<<<<<<
# Avg Qoe: 0.78, Perfect Qoe: 0.04. Throughput: 0.03 req/s. TTFT 19.13 s. Pause frequency: 0.80. Avg response 607.02. 


ps aux | grep python | awk '{print $2}' | xargs -r kill -9
vllm serve microsoft/Phi-3.5-MoE-instruct --scheduling-strategy qoe-avg --load-format dummy --preemption-mode swap --swap-space 20  --tensor-parallel-size 8 --trust-remote-code --max-num-batched-tokens 200000  & 
sleep 77
# 
cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model  microsoft/Phi-3.5-MoE-instruct --num-requests 150 --arrival-rate 1 --scheduling qoe-avg   --max-tokens 30000 --prompt-trace arxiv 
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

