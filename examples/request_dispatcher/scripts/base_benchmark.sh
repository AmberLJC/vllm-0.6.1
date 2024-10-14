# A40 - INFO 10-13 23:43:48 distributed_gpu_executor.py:57] # GPU blocks: 19941, # CPU blocks: 6826
# ============================================================================
# ==================================== fcfs  =================================
# ============================================================================ 
  
ps aux | grep python | awk '{print $2}' | xargs -r kill -9

vllm serve microsoft/Phi-3-mini-128k-instruct  --scheduling-strategy fcfs --load-format dummy  --tensor-parallel-size 4 --trust-remote-code --max-num-batched-tokens 200000  & 
sleep 55

cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model  microsoft/Phi-3-mini-128k-instruct   --num-requests 50 --arrival-rate 0.5 --scheduling fcfs  --max-tokens 30000 --prompt-trace arxiv
ps aux | grep python | awk '{print $2}' | xargs -r kill -9


# ============================================================================
# ================================== qoe-avg  ================================
# ============================================================================

ps aux | grep python | awk '{print $2}' | xargs -r kill -9

vllm serve microsoft/Phi-3-mini-128k-instruct  --scheduling-strategy qoe-avg --load-format dummy  --tensor-parallel-size 4 --trust-remote-code --max-num-batched-tokens 200000  --preemption-mode swap --swap-space 10 --preemption_freq 0.3 &
sleep 55
# 
cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model  microsoft/Phi-3-mini-128k-instruct  --num-requests 50 --arrival-rate 0.5 --scheduling qoe-avg-0.1  --max-tokens 30000  --prompt-trace arxiv
ps aux | grep python | awk '{print $2}' | xargs -r kill -9


# ============================================================================
# ================================== qoe-min  ================================
# ============================================================================

ps aux | grep python | awk '{print $2}' | xargs -r kill -9

vllm serve microsoft/Phi-3-mini-128k-instruct  --scheduling-strategy qoe-min --load-format dummy  --tensor-parallel-size 4 --trust-remote-code --max-num-batched-tokens 200000  --preemption-mode swap --swap-space 10 --preemption_freq 0.2 &
sleep 55
# 
cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model  microsoft/Phi-3-mini-128k-instruct   --num-requests 50 --arrival-rate 0.5 --scheduling qoe-min  --max-tokens 30000  --prompt-trace arxiv
ps aux | grep python | awk '{print $2}' | xargs -r kill -9
