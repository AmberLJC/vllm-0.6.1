vllm serve facebook/opt-13b --scheduling-strategy fcfs --load-format dummy  & 
sleep 60

cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --time-index 394

sleep 120
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --time-index 857

sleep 120
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --time-index 1363

cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --prompt-trace sharegpt-multi --time-index 394

sleep 120
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --prompt-trace sharegpt-multi --time-index 857

sleep 120
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --prompt-trace sharegpt-multi --time-index 1363

pkill python
ps aux | grep python | awk '{print $2}' | xargs -r kill -9
