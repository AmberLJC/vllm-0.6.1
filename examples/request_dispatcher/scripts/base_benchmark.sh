ps aux | grep python | awk '{print $2}' | xargs -r kill -9
# ============================================================================
# ==================================== fcfs  =================================
# ============================================================================
#  >>>>>>> 2024-09-25 04:37-facebook-opt-13b-sharegpt-poisson*150-8.0(0.2)-day--1-fcfs.json (142 requests) <<<<<<<<<
# Avg Qoe: 0.83, Perfect Qoe: 0.54. Throughput: 0.13 req/s. TTFT 6.06 s. Pause frequency: 0.46. Avg response 331.08. 
# Total time taken: 105.  


# vllm serve facebook/opt-13b --scheduling-strategy fcfs --load-format dummy --preemption-mode swap --swap-space 30  & 
# sleep 40

# cd /vllm/examples/request_dispatcher
# python request_dispatcher.py --model facebook/opt-13b --num-requests 150 --arrival-rate 8 --scheduling fcfs


# ps aux | grep python | awk '{print $2}' | xargs -r kill -9

# ============================================================================
# ================================== qoe-avg  ================================
# ============================================================================
# >>>>>>>>>> 2024-09-25 19:54-facebook-opt-13b-sharegpt-poisson*150-8.0(0.2)-day--1-qoe-avg. (142 requests) <<<<<<<<<< 
# Avg response 326.96 
# Avg Qoe: 0.98, Perfect Qoe: 0.87. Throughput: 0.13 req/s. Avg TTFT: 0.31.  Pause frequency: 0.18. 
ps aux | grep python | awk '{print $2}' | xargs -r kill -9
vllm serve facebook/opt-13b --scheduling-strategy qoe-avg --load-format dummy --preemption-mode swap --swap-space 30  & 
sleep 40

cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model facebook/opt-13b --num-requests 150 --arrival-rate 8 --scheduling qoe-avg

