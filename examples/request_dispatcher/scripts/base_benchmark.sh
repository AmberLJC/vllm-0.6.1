
#  >>>>>>> 2024-09-25 04:37-facebook-opt-13b-sharegpt-poisson*150-8.0(0.2)-day--1-fcfs.json (142 requests) <<<<<<<<<
# Avg Qoe: 0.83, Perfect Qoe: 0.54. Throughput: 0.13 req/s. TTFT 6.06 s. Pause frequency: 0.46. Avg response 331.08. 
# Total time taken: 105.80011034011841. 
#  >>>>>>> 2024-09-25 04:43-facebook-opt-13b-sharegpt-poisson*150-8.0(0.2)-day--1-qoe-avg.json (142 requests) <<<<<<<<<
# Avg Qoe: 0.95, Perfect Qoe: 0.67. Throughput: 0.13 req/s. TTFT 0.31 s. Pause frequency: 0.33. Avg response 331.72. 
# Total time taken: 109.37019205093384. 

vllm serve facebook/opt-13b --scheduling-strategy fcfs --load-format dummy --preemption-mode swap --swap-space 50  & 
sleep 30

cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model facebook/opt-13b --num-requests 150 --arrival-rate 8 --scheduling fcfs

pkill python
ps aux | grep python | awk '{print $2}' | xargs -r kill -9


vllm serve facebook/opt-13b --scheduling-strategy qoe-avg --load-format dummy --preemption-mode swap --swap-space 30  & 
sleep 50

cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model facebook/opt-13b --num-requests 150 --arrival-rate 8 --scheduling qoe-avg

ps aux | grep python | awk '{print $2}' | xargs -r kill -9
