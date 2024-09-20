vllm serve facebook/opt-13b --scheduling-strategy fcfs --load-format dummy --preemption-mode swap --swap-space 100 & 
sleep 60

cd /vllm/examples/request_dispatcher
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --time-index 400

sleep 120
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --time-index 548


sleep 120
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --time-index 804


sleep 120
python request_dispatcher.py --model facebook/opt-13b --time-range 'hour' --arrival-trace burstgpt --time-index 1020


cd /vllm/examples/request_dispatcher/scripts
python send.py fcfs-profile-one-hour
pkill python
ps aux | grep python | awk '{print $2}' | xargs -r kill -9
