#  
echo ">>>>>>>>>> Start FCFS for llama3-70B <<<<<<<<<" >> results.log

vllm serve meta-llama/Meta-Llama-3.1-70B --scheduling-strategy fcfs --load-format dummy  --tensor-parallel-size 8 
sleep 90

# python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B --num-requests 50 --arrival-rate 0.5 --max-tokens 128000 --arrival-trace periodic_poisson --scheduling fcfs --prompt-trace short-long
python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B --num-requests 200 --arrival-rate 0.1 --max-tokens 128000 --arrival-trace gamma --scheduling fcfs --prompt-trace short-long

python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B --num-requests 100 --arrival-rate 1 --max-tokens 128000 --arrival-trace periodic_poisson  --scheduling fcfs --prompt-trace sharegpt-multi

python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B --num-requests 100 --arrival-rate 1 --max-tokens 128000 --arrival-trace periodic_poisson  --scheduling fcfs --prompt-trace arxiv

python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B --num-requests 100 --arrival-rate 1 --max-tokens 128000 --arrival-trace periodic_poisson  --scheduling fcfs --prompt-trace sharegpt



pkill python
ps aux | grep python | awk '{print $2}' | xargs -r kill -9
