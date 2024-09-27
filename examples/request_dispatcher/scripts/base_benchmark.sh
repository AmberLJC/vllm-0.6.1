ps aux | grep python | awk '{print $2}' | xargs -r kill -9
# ============================================================================
# ==================================== fcfs  =================================
# ============================================================================ 
 
#  >>>>>>> 2024-09-26 20:03-microsoft-Phi-3.5-MoE-instruct-arxiv-poisson*50-0.4(0.2)-day--1-fcfs.json (51 requests) <<<<<<<<<
# Avg Qoe: 0.68, Perfect Qoe: 0.02. Throughput: 0.03 req/s. TTFT 33.15 s. Pause frequency: 0.61. Avg response 610.46. 

#  >>>>>>> 2024-09-26 23:05-microsoft-Phi-3.5-MoE-instruct-arxiv-poisson*50-0.3(0.2)-day--1-fcfs.json (51 requests) <<<<<<<<<
# Avg Qoe: 0.75, Perfect Qoe: 0.04. Throughput: 0.03 req/s. TTFT 21.47 s. Pause frequency: 0.59. Avg response 608.94. 

# sleep 55

# cd /vllm/examples/request_dispatcher
# python request_dispatcher.py --model  microsoft/Phi-3.5-MoE-instruct --num-requests 50 --arrival-rate 0.3 --scheduling fcfs  --max-tokens 30000 --prompt-trace arxiv 


# ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# ============================================================================
# ================================== qoe-avg  ================================
# ============================================================================


#  >>>>>>> 2024-09-27 00:43-microsoft-Phi-3.5-MoE-instruct-arxiv-poisson*50-0.3(0.2)-day--1-qoe-avg.json (51 requests) <<<<<<<<<
# Avg Qoe: 0.78, Perfect Qoe: 0.04. Throughput: 0.03 req/s. TTFT 19.13 s. Pause frequency: 0.80. Avg response 607.02. 


ps aux | grep python | awk '{print $2}' | xargs -r kill -9
vllm serve microsoft/Phi-3.5-MoE-instruct --scheduling-strategy qoe-avg --load-format dummy --preemption-mode swap --swap-space 20  --tensor-parallel-size 4 --trust-remote-code --max-num-batched-tokens 200000  & 
sleep 65
# 
cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model  microsoft/Phi-3.5-MoE-instruct --num-requests 50 --arrival-rate 0.3 --scheduling qoe-avg   --max-tokens 30000 --prompt-trace arxiv 
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

