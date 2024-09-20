import argparse
import json
import yaml
import asyncio
from typing import Iterable, List, Generator
from datetime import datetime
import pandas as pd
import numpy as np
import requests, time
from typing import AsyncGenerator
from collections import defaultdict
import subprocess
import httpx
import aiohttp
import datetime


from analyze_perf import analyze_one_trace
# SYSTEM_PROMPT = "You are an artificial intelligence assistant that gives helpful answers to the user's questions or instructions."
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=3 * 3600)

np.random.seed(42)
 
async def single_request(prompt_config: dict,  
                   model_config: dict,
                   url: str = "http://localhost:8000/v1/completions",
                   file_name: str = "results.json"
                  ): 
    # Define the headers
    headers = {"Content-Type": "application/json"}

    # Define the data payload
    config = {
        "model": model_config['model'],
        "prompt": prompt_config['prompt'],
        "max_tokens": model_config['max_tokens'],
        "stream": model_config['stream'],
        "qoe_required": {
            "ttft": prompt_config['ttft'],
            "latency": prompt_config['latency'],
            "output_len": min(model_config['max_tokens'], prompt_config['output_len']),
        }
    }
    time_list = [time.monotonic()]
    async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as session: 
        async with session.post(url, headers=headers, json=config) as response:
            # Request failed
            if response.status >= 300:
                print(f"Request failed: {await response.text()}") 
                return
            chunks = []
            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk) 
                time_list.append(time.monotonic())
    log_result(file_name, time_list, config['qoe_required'], config['prompt'])

def log_result(file_name: str,
               time_list: list, 
               qoe_required: dict,
               prompt: str,):
    with open(file_name, 'a') as file:
        res = {}
        try:   
            res['time_list'] = time_list 
            res['input'] = prompt
            res['qoe'] = qoe_required

        except Exception as error:
            print(f"Encountered an error while processing results: {error}")
            res['error'] = error
        json.dump(res, file)
        file.write('\n')
        print(f"{file_name}")

def read_arrival_trace(args: argparse.Namespace
                       ) -> List[float]:
    arrival_trace = args.arrival_trace
    time_range = args.time_range

    if arrival_trace == 'burstgpt':
        file_name = 'arrival_trace/BurstGPT_1.csv'
        df = pd.read_csv(file_name)
        time_list = df['Timestamp']

        hourly_arrival = defaultdict(list)
        daily_arrival = defaultdict(list)

        for t in time_list:
            hourly_arrival[t//3600].append(t)
            daily_arrival[t//86400].append(t)
        
        the_day_index = 9 # day 10
        if args.time_index != -1: 
            the_hour_index = args.time_index
            
        else: 
            # the_hour_index = 249 
            the_hour_index = 400 # request rate > 1, up and down
        if time_range == 'day':
            arrival_ts = daily_arrival[the_day_index]
        elif time_range == 'hour':
            arrival_ts = hourly_arrival[the_hour_index]
        return np.diff(np.array(arrival_ts))

    elif arrival_trace == 'poisson':
        arrival_rate = args.arrival_rate 
        return np.random.exponential(scale=1/arrival_rate, size=args.num_requests)

def read_prompt_trace(args: argparse.Namespace):
    prompt_trace = args.prompt_trace
    if prompt_trace == 'sharegpt':
        prompt_trace_file = "prompt_trace/sharegpt_qoe_trace.json" 
    elif prompt_trace == 'sharegpt-multi':
        prompt_trace_file = "prompt_trace/sharegpt_multi_qoe_trace.json"
    elif prompt_trace == 'arxiv':
        prompt_trace_file = "prompt_trace/arvix_qoe_trace.json"
    
    with open(prompt_trace_file, 'r') as file:
        data = json.load(file)
    return data

async def main(args):
    start_time = time.time()
    model_file = args.model.replace("/", "-")
    date = datetime.datetime.fromtimestamp(time.time())

    formatted_date = date.strftime('%Y-%m-%d %H:%M')
    result_file = f'{formatted_date}-{model_file}-{args.arrival_trace}*{args.num_requests}-{args.arrival_rate}-{args.time_range}-{args.time_index}.json'
    prompt_trace = read_prompt_trace(args)
    num_prompts = len(prompt_trace)
    print(f'>>>>>Start {result_file}<<<<<<')
    arrival_intervals = read_arrival_trace(args)
    model_config = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "stream": "True",
    }
    tasks = []
    with open(result_file, 'w') as file:
        # write arrival_intervals to file
        json.dump(arrival_intervals.tolist(), file)
    
    for i, arr_int in enumerate(arrival_intervals): 
        print(f"Progress: {i}/{len(arrival_intervals)} ")
        task = asyncio.create_task(single_request(prompt_trace[i % num_prompts], model_config, args.url, result_file))
        tasks.append(task)  
        await asyncio.sleep(max(arr_int, 0.01))

    # wait for all requests to finish
    await asyncio.gather(*tasks)
    print(f"Total time taken: {time.time()-start_time}")
    analyze_one_trace(result_file)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/completions") 
    parser.add_argument("--stream", action="store_true") 
    parser.add_argument("--prompt-trace", type=str, default='sharegpt') 
    parser.add_argument("--arrival-trace", type=str, default='poisson', choices=['burstgpt', 'poisson']) 
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--time-range", type=str, default='day', choices=['day', 'hour'])
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    asyncio.run(main(args))
