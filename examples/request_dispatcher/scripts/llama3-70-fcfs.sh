SCHEUDLING='qoe'
echo "--------------- Start $SCHEUDLING for llama3-70B ----------" >> results.log
#!/bin/bash

# Function to run the VLLM model with the specified scheduling strategy
run_model() {
    local SCHEDULE=$1  # Accepts scheduling strategy as an argument
    
    # Start serving the model with the input scheduling strategy
    vllm serve meta-llama/Meta-Llama-3.1-70B \
        --max-num-batched-tokens 100000 \
        --scheduling-strategy "$SCHEDULE" \
        --load-format dummy \
        --tensor-parallel-size 8 &

    sleep 80 

    cd /vllm/examples/request_dispatcher 
    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 50 \
        --arrival-rate 0.5 \
        --max-tokens 4096 \
        --arrival-trace periodic_poisson \
        --scheduling "$SCHEDULE" \
        --prompt-trace short-long
    
    python request_dispatcher.py --model meta-llama/Meta-Llama-3.1-70B \
        --num-requests 200 \
        --arrival-rate 2 \
        --max-tokens 4096 \
        --arrival-trace gamma \
        --scheduling "$SCHEDULE" \
        --prompt-trace short-long
 
    # Terminate python processes
    pkill python
    ps aux | grep python | awk '{print $2}' | xargs -r kill -9
} 

run_model "qoe"
