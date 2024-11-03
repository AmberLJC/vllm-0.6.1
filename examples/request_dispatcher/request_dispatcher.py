import argparse
import json 
import asyncio
from typing import Iterable, List, Generator
from datetime import datetime
import pandas as pd
import numpy as np
import requests, time
from typing import AsyncGenerator
from collections import defaultdict 
import aiohttp
import datetime
import subprocess

from read_sys_log import visualize_system_stats, list_files_by_creation_time
from analyze_perf import analyze_one_trace, plot_cdf_together
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

def gamma_arrival_times(shape, scale, num_arrivals):
    # Generate inter-arrival times from a Gamma distribution
    inter_arrival_times = np.random.gamma(shape, scale, num_arrivals)
    arrival_times = np.cumsum(inter_arrival_times)
    return np.diff(arrival_times) 

def generate_duty_cycle_poisson_arrival(arrival_rate, height, width, duration=1800, num_cycle=1):
  peak_rate = arrival_rate * height
  non_peak_rate = (arrival_rate - peak_rate * width) / (1-width)
  print(f'peak rate: {peak_rate}, non-peak rate: {non_peak_rate}')
  num_peak_requests = int(peak_rate * duration * width)
  num_non_peak_requests = max(0, int(non_peak_rate * duration * (1-width)))
  print(f'# requests in peak: {num_peak_requests}, non-peak: {num_non_peak_requests}')
  interval_list = np.array([])
  for c in range(num_cycle):
    peak_interval = np.random.exponential(1/peak_rate, num_peak_requests)
    if num_non_peak_requests > 0:
      non_peak_interval = np.random.exponential(1/non_peak_rate, num_non_peak_requests)
    else:
      non_peak_interval = np.array([])
    interval_list = np.concatenate((interval_list, peak_interval, non_peak_interval))
  return interval_list


def read_arrival_trace(args: argparse.Namespace
                       ) -> List[float]:
    arrival_trace = args.arrival_trace
    time_range = args.time_range

    if arrival_trace == 'burstgpt':
        file_name = 'arrival_trace/BurstGPT_2.csv'
        import os
        if not os.path.exists(file_name): 
            command = ["wget", "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_2.csv"] 
            subprocess.run(command)

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
            the_hour_index = 400 # request rate > 1, up and down
        if time_range == 'day':
            arrival_ts = daily_arrival[the_day_index]
        elif time_range == 'hour':
            arrival_ts = hourly_arrival[the_hour_index]
        return np.diff(np.array(arrival_ts))
    elif arrival_trace == 'poisson':
        arrival_rate = args.arrival_rate 
        return np.random.exponential(scale=1/arrival_rate, size=args.num_requests)
    elif arrival_trace == 'periodic_poisson':
        arrival_rate = args.arrival_rate
        interval_list = np.array([])
        num_duty_cycles = 5
        for i in range(num_duty_cycles):
            inv = np.random.exponential(scale=1/arrival_rate, size=args.num_requests)
            interval_list = np.concatenate((interval_list, inv))
            if i < num_duty_cycles - 1:
                interval_list = np.append(interval_list, args.num_requests / args.arrival_rate)
        return interval_list
    elif arrival_trace == 'gamma':
        arrival_inv = 1 / args.arrival_rate
        shape = args.burst   # burstness: smaller value means more bursty
        scale = arrival_inv / shape  # avg arrival interval 
        return gamma_arrival_times(shape, scale, args.num_requests )
    elif arrival_trace == 'duty':
        return generate_duty_cycle_poisson_arrival(args.arrival_rate, args.height, args.width, args.duration)
    else:
        raise ValueError(f"Unknown arrival trace: {arrival_trace}")
    
def read_prompt_trace(args: argparse.Namespace):
    prompt_trace = args.prompt_trace
    if prompt_trace == 'sharegpt':
        prompt_trace_file = "prompt_trace/sharegpt_qoe_trace.json" 
    elif prompt_trace == 'sharegpt-multi':
        prompt_trace_file = "prompt_trace/sharegpt_multi_qoe_trace.json"
        # prompt_trace_file = "prompt_trace/sharegpt_multi_50k_qoe_trace.json"
    elif prompt_trace == 'arxiv':
        prompt_trace_file = "prompt_trace/arvix_qoe_trace.json"
    elif prompt_trace == 'short-long':
        prompt_trace_file = "prompt_trace/short_long_qoe_trace.json"
    elif prompt_trace == 'code':
        prompt_trace_file = "prompt_trace/code_qoe_trace.json"
    elif prompt_trace == 'prefill':
        # Just for profiling recompute latency
        prompt_trace_file = "prompt_trace/test_prefill_trace.json"
    else:
        raise ValueError(f"Unknown prompt trace: {prompt_trace}")
    
    with open(prompt_trace_file, 'r') as file:
        data = json.load(file)
    return data

async def main(args):
    start_time = time.time()
    model_file = args.model.replace("/", "-")
    now = time.time()
    date = datetime.datetime.fromtimestamp(now)

    formatted_date = date.strftime('%Y-%m-%d %H:%M')
    prompt_trace = read_prompt_trace(args)
    num_prompts = len(prompt_trace)
    arrival_intervals = read_arrival_trace(args)
    if args.arrival_trace == 'burstgpt':
        result_file = f'{formatted_date}-{model_file}-{args.prompt_trace}-{args.arrival_trace}*{len(arrival_intervals)}-{args.time_range}-{args.time_index}-{args.scheduling}.json'
    elif args.arrival_trace == 'gamma':
        result_file = f'{formatted_date}-{model_file}-{args.prompt_trace}-{args.arrival_trace}*{len(arrival_intervals)}-{args.arrival_rate}({args.burst})-{args.scheduling}.json'
    elif args.arrival_trace == 'duty':
        result_file = f'{formatted_date}-{model_file}-{args.prompt_trace}-{args.arrival_trace}*{len(arrival_intervals)}-{args.arrival_rate}(h={args.height},w={args.width})-{args.scheduling}.json'
    else:
        result_file = f'{formatted_date}-{model_file}-{args.prompt_trace}-{args.arrival_trace}*{len(arrival_intervals)}-{args.arrival_rate}-{args.scheduling}.json'
    print(f'>>>>>Start {result_file}<<<<<<')
    model_config = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "stream": "True",
    }
    tasks = []
    with open(result_file, 'w') as file:
        # write arrival_intervals to file
        json.dump(arrival_intervals.tolist(), file)
        file.write('\n')
    
    for i, arr_int in enumerate(arrival_intervals): 
        print(f"Progress: {i}/{len(arrival_intervals)} ") 
        task = asyncio.create_task(single_request(prompt_trace[i % num_prompts], model_config, args.url, result_file))
        tasks.append(task)  
        await asyncio.sleep(max(arr_int, 0.01))

    # wait for all requests to finish
    await asyncio.gather(*tasks)
    total_duration = f"Total time taken: {time.time()-start_time}. \n"
    print(total_duration)
    metric_dict = analyze_one_trace(result_file)
    # plot_cdf_together({result_file: metric_dict}, result_file)
    with open('results.log', 'a') as file:
        file.write(total_duration)
    
    visualize_system_stats(list_files_by_creation_time('system_logs'), arrival_intervals.tolist(), now)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/completions") 
    parser.add_argument("--stream", action="store_true") 
    parser.add_argument("--prompt-trace", type=str, default='sharegpt-multi') 
    parser.add_argument("--arrival-trace", type=str, default='gamma', choices=['burstgpt', 'poisson', 'periodic_poisson', 'gamma', 'duty']) 
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    # for burstGPT
    parser.add_argument("--time-range", type=str, default='day', choices=['day', 'hour'])
    parser.add_argument("--time-index", type=int, default=-1)
    # for Gamma
    parser.add_argument("--burst", type=float, default=10)
    # for duty cycle, each duty cycle is 20 minutes
    parser.add_argument("--height", type=float, default=2, help="height of the bursty period compared to avg.")
    parser.add_argument("--width", type=float, default=0.35, help="duration ratio of the bursty period.")
    parser.add_argument("--duration", type=int, default=1200, help="duration of the trace.")
    # for the serving system
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--scheduling", type=str, default="unknown", help="Notes for the server")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    asyncio.run(main(args))
