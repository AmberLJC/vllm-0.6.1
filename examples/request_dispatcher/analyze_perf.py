import json
import os
import matplotlib.pyplot as plt
# from datetime import datetime
from collections import defaultdict
import numpy as np
# Initialize empty lists to store timestamps and latency_per_total_token values
 
from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

from vllm.core.andes_utils.qoe_tracker import QoETracker

# Read the log file and extract the data


FIG_DIR = 'fig'
def read_log(filename, log_data):
	with open(filename, 'r') as file:
		for line in file:
			try:
				log_data.append(json.loads(line))
			except json.decoder.JSONDecodeError:
				# print(line)
				continue

def cal_avg_std(latency_list, alpha = 1.5):
	# avg = sum(latency_list) / len(latency_list)
	med = np.median(latency_list)
	std = (sum([(x - med) ** 2 for x in latency_list]) / len(latency_list)) ** 0.5
	data = np.array(latency_list)
	Q1 = np.percentile(data, 25)
	Q3 = np.percentile(data, 75)
	IQR = Q3 - Q1
	pause_threshold = Q3 + 1.5 * IQR
	pause_list = [l for l in latency_list if l > pause_threshold]
	pause_duration = sum(pause_list) - len(pause_list) * med
	pause_times = len(pause_list)
	return med, std, pause_duration, pause_times

def get_token_latency(time_list): 
	latency_list = []
	for i in range(len(time_list) - 2):
		latency_list.append(time_list[i+2] - time_list[i+1])
	return latency_list

def cal_input_len(input, tokenizer):
	return  len(tokenizer.tokenize(input))

def find_first_positive(my_list):
	first_positive = 0
	for number in my_list:
		if number > 0:
			first_positive = number
			break  # Exit the loop as soon as a positive number is found
	return first_positive


def cal_pause_duration(latency_list):
	# sum up all pause > 1s
	pause_duration = 0
	pause_times = 0
	anchor_time = anchor_len = 0 
	for i in range(len(latency_list) - 1):
		if latency_list[i] > 5:
			pause_times += 1
		if latency_list[i] > 1:  
			p = sum(latency_list[:i+1]) - anchor_time - (i-anchor_len) * 0.25
			if p > 0:
				pause_duration += p
				anchor_time = sum(latency_list[:i+1])
				anchor_len = i
	return pause_duration, pause_times

def process_results(log_data, alpha = 3):
	
	avg_token_latency_list = [] 
	pause_duration_list = []
	pause_times_list = []
	# first_token_latency_list = [] 
	first_chunk_latency_list = []
	input_len_list = []
	output_len_list = []
	total_latency_list = []
	thpt_list = []
	
	error_rate = 0 

	for entry in log_data: 
		if "time_list" in entry: 
			# timestamp_str = entry["timestamp"]
			output_len = len(entry["time_list"])

			if 'input_len' not in entry:
				if 'input' not in entry:
					continue
				input_len = cal_input_len(entry['input'], tokenizer)

			else:
				input_len = entry["input_len"]
			input_len_list.append(input_len)
			output_len_list.append(output_len)
			
			if 'latency' not in entry:
				latency = get_token_latency(entry["time_list"])
			else:
				latency = entry["latency"]
			
			if len(latency) < 2:
				continue
			ttft = entry["time_list"][1] - entry["time_list"][0]
			chunk_size = min(100, input_len)
			first_chunk_latency_list.append(sum([ttft]+latency[:chunk_size]))

			total_latency = entry["time_list"][-1] - entry["time_list"][0]
			if total_latency > 0: # TODO: remove this
				total_latency_list.append(total_latency)
				thpt_list.append(total_latency / output_len)
				
			avg_token_latency, std_token_latency, pause_duration, pause_times = cal_avg_std(latency[1:], alpha)
			avg_token_latency_list.append(avg_token_latency) 
			pause_times_list.append(pause_times)
			
			# if ttft > 0:
			# 	first_token_latency_list.append(ttft)
			# else:
			# 	first_token_latency_list.append(find_first_positive(latency))

			pause_duration, pause_times = cal_pause_duration(latency)
			pause_duration_list.append(pause_duration)
			# timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
			# hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
			# hourly_averages[hour_key].append( output_tokens/ sum(latency))
		elif 'error' in entry:
			error_rate += 1


	metric_dict = {}
	metric_dict['avg_token_latency'] = avg_token_latency_list
	# metric_dict['cv_token_latency'] = cv_token_latency_list
	metric_dict['pause_duration'] = pause_duration_list
	metric_dict['pause_times'] = pause_times_list
	# metric_dict['first_token_latency'] = first_token_latency_list
	# metric_dict['error_rate'] = error_rate / len(log_data)
	metric_dict['input_len'] = input_len_list
	metric_dict['output_len'] = output_len_list
	metric_dict['total_latency'] = total_latency_list
	metric_dict['avg_thpt'] = thpt_list
	metric_dict['first_chunk_latency'] = first_chunk_latency_list
	short_results = f'avg thpt: {np.mean(thpt_list) :2f}'
	# with open('thpt.log', 'a') as f: 
	# 	f.write(short_results + '\n')

	return metric_dict
 
def plot_over_hour(hours, averages):
	plt.figure(figsize=(10, 5))
	plt.plot(hours, averages, marker='o', linestyle='-')
	plt.title("Average Latency per Output Token Over Time")
	plt.xlabel("Time (Hourly)")
	plt.ylabel("Average Latency per Output Token")
	plt.grid(True) 
	plt.show()

def plot_cdf_per_request(metric_list, file_name, metric_name ):
	# Sort the data
	data_sorted = np.sort(metric_list)
	cdf = np.linspace(0, 1, len(data_sorted))
	plt.figure(figsize=(8, 4))
	plt.plot(data_sorted, cdf, marker='.', linestyle='none')

	top90 = np.percentile(data_sorted, 90)
	top50 = np.percentile(data_sorted, 50)
	top10 = np.percentile(data_sorted, 10)

	plt.axvline(top90, color='red', linestyle='--', label=f'Top 90%: {top90:.5f}')
	plt.axvline(top50, color='green', linestyle='--', label=f'Top 50%: {top50:.5f}')
	plt.axvline(top10, color='blue', linestyle='--', label=f'Top 10%: {top10:.5f}')
	plt.title(f'{file_name}')
	plt.xlabel(f"{metric_name}")
	plt.ylabel("CDF")
	plt.grid(True)
	plt.xlim(xmin=0)
	plt.ylim((0, 1))
	plt.legend()
	plt.savefig(f'{FIG_DIR}/{file_name}-{metric_name}.png')
	plt.show()

def plot_pdf_per_request(metric_list, file_name ):
	plt.figure(figsize=(8, 4))
	num_req = 0
	for k in metric_list: 
		if k == 'input_len' or k == 'output_len':
			data = metric_list[k] 
			plt.hist(data, bins=40, alpha=0.7, label=k) 
			avg = round( sum(data) / len(data), 2)
			plt.axvline(avg, color='red', linestyle='--', label=f'{k} Avg: {avg:.5f}')
			num_req = len(data) 

	plt.xlabel(f'Length - {num_req} request')
	plt.ylabel('PDF ')
	plt.legend()
	plt.title(f'{file_name}')
	plt.savefig(f'{FIG_DIR}/{file_name}-len-distribution.png')

class ThptTracker:
	def __init__(self) -> None:
		self.time_token_thpt = defaultdict(int)
		self.granularity = 10
		# self.output_len = 0
		self.req_counter = 0
	
	def add_tokens(self, timelist):
		# self.output_len += len(timelist)
		self.req_counter += 1
		for t in timelist:
			self.time_token_thpt[t//self.granularity] += 1
	
	def plt_thpt_time(self, file_name):
		def find_saturate_thpt(thpt_list):
			
			start_id = end_id = -1
			for i in range(1, len(thpt_list)):
				if thpt_list[i] < thpt_list[i-1]:
					start_id = i-1
					break
			for i in range(1, len(thpt_list)):
				if thpt_list[-1-i] < thpt_list[-i] - 1:
					end_id = len(thpt_list) - i + 1
					break

			res = [i for i in thpt_list[start_id:end_id] if i > 100]
			
			return res[:len(res)]
				 
		x = list(self.time_token_thpt.keys())
		x = (np.array(x) - x[0]) * self.granularity
		y = np.array(list(self.time_token_thpt.values())) / self.granularity
		saturate_thpt = find_saturate_thpt(y)
		avg_thpt = sum(saturate_thpt) / max(1, len(saturate_thpt)) 

		x = list(self.time_token_thpt.keys())
		x = (np.array(x) - x[0]) * self.granularity

		makespan = (x[-1] - x[0]) * self.granularity
		return self.req_counter / makespan
	
def plot_scatter(x_list, y_list, file_name):
	plt.figure()
	# sample 5% data points
	x_list = np.array(x_list)
	y_list = np.array(y_list)
	idx = np.random.choice(len(x_list), int(len(x_list) * 0.08), replace=False)
	x_list = x_list[idx]
	y_list = y_list[idx]	
	plt.scatter(x_list, y_list)
	plt.xlabel('Total Length')
	plt.ylabel('QoE')	
	plt.savefig(f'{FIG_DIR}/{file_name}-len-qoe.png')

def plot_arrival_histogram(time_list, file_name):
	plt.figure()
	# sample 5% data points
	time_list = np.cumsum(time_list)
	plt.hist(time_list, bins=40, alpha=0.7, label='Arrival Time') 
	plt.xlabel('Time')
	plt.ylabel('Frequency')
	plt.legend()
	plt.title(f'{file_name}')
	plt.savefig(f'{FIG_DIR}/{file_name}-arrival-histogram.png')


def plt_accumulate_token_over_time(log_data, file_name):
	plt.figure() 
	total_tokens = 0 
	qoe_list = []
	ttft_list = []
	pause_list = []
	thpt_tracker = ThptTracker()
	total_pause = 0
	total_len_list = []

	for i, entry in enumerate(log_data): 
		if "time_list" in entry:
			time_list = entry["time_list"]
			latency = get_token_latency(time_list)
			pause_duration, pause_times = cal_pause_duration(latency) 
			pause_list.append(pause_duration)
			total_pause += pause_times

			total_tokens += len(time_list)

			request = QoETracker(entry['qoe'])
			qoe = request.analyze_QoE( list(np.array(time_list) - time_list[0])[1:] )
			qoe_list.append(qoe)
			ttft_list.append(time_list[1] - time_list[0])
			thpt_tracker.add_tokens(time_list[1:])
			input_len = cal_input_len(entry['input'], tokenizer)
			total_len_list.append( len(time_list) + input_len ) 

			if  len(time_list) % 1 == 0:
				group_time = np.array(time_list) - time_list[0]
				y = np.arange(1, len(time_list)+1)
				plt.plot(group_time, y, linestyle='-', markersize=1)  
			
			# debug
			if ttft_list[-1] > 10:
				print(f'input_len={input_len}, output_len={len(time_list)}, ttft={ttft_list[-1]}, qoe={qoe_list[-1]}, required_qoe={entry["qoe"]}')

	avg_qoe = sum(qoe_list) / len(qoe_list)
	# error = np.std(qoe_list, ddof=1)  
	# qoe_25th = np.percentile(qoe_list, 25)
	# qoe_75th = np.percentile(qoe_list, 75)

	perfect_qoe = sum([1 for q in qoe_list if q >= 0.99]) / len(qoe_list)
	# avg_pause = sum(pause_list) / len(pause_list)
	avg_thpt = thpt_tracker.plt_thpt_time(file_name)

	log_results =	f'>>>>>>>>>> {file_name[:-5]}. ({len(log_data)} requests) <<<<<<<<<< \n' + \
					f'Avg response {total_tokens/len(qoe_list):.2f} \n' + \
					f'Avg Qoe: {avg_qoe:.2f}, Perfect Qoe: {perfect_qoe:.2f}. ' + \
					f'Throughput: {avg_thpt:.5f} req/s. Avg TTFT: {sum(ttft_list) / len(ttft_list) :.2f}.  Pause frequency: {total_pause/len(log_data):.2f}. '
					
	short_results = f'Avg Qoe: {avg_qoe:.2f}, Perfect Qoe: {perfect_qoe:.2f}. Throughput: {avg_thpt:.5f} req/s. TTFT {sum(ttft_list) / len(ttft_list) :.2f} s. Pause frequency: {total_pause/len(log_data) :.2f}. Avg response {total_tokens/len(qoe_list):.2f}. '
	with open('results.log', 'a') as f: 
		f.write(f' >>>>>>> {file_name} ({len(log_data)} requests) <<<<<<<<<\n')
		f.write(short_results + '\n')

	x = np.linspace(0, group_time[-1], 100)
	xmin = 0
	plt.plot(x + xmin, 5.1 * (x - 1), linestyle='--', markersize=2)
	# plt.plot(x, 3.78 * (x - 1), linestyle='--', markersize=2)
	
	plt.xlabel('Time (s)')
	plt.ylabel('Accumulated Tokens')
	plt.title(log_results)  

	plt.savefig(f'{FIG_DIR}/{file_name}-accumulated-token.png')
	plt.close()
	# plot_arrival_histogram(log_data[0], file_name)
	# plot_scatter(total_len_list, qoe_list, file_name)
	print(log_results)
	# print(f'ttft={ttft_list}')
	# print(f'qoe={qoe_list}')
	return {'qoe': qoe_list, 'ttft': ttft_list}
		 
def plot_cdf_together(log_dict, file_name ):
	# Sort the data
	
	metric_names = []
	error_rate = {}
	for key in log_dict:
		metric_names = log_dict[key].keys()
		break 
	
	if 'input_len' in metric_names:
		for log in log_dict:
			plot_pdf_per_request(log_dict[log], file_name)
			break

	for metric in metric_names: 
		if metric == 'error_rate':
			for log in log_dict:  
				error_rate[log] = log_dict[log][metric]
			continue

		if 'len' in metric:
			continue 
		plt.figure(figsize=(5.4, 4), constrained_layout=True)
		xmin = 0
		xmax = 10e6

		res = []
		for log in log_dict: # different log file  
			data_sorted = np.sort(log_dict[log][metric])

			avg_data = sum(data_sorted) / len(data_sorted) 
			xmin = max(xmin, np.percentile(data_sorted, 0.1))
			xmax = min(xmax, np.percentile(data_sorted, 99.9))
			cdf = np.linspace(0, 1, len(data_sorted))
			date_time =  (log.split('-')[5] + '-' + log.split('-')[-5]  )
			plt.plot(data_sorted, cdf, label=f'{date_time}, avg: {avg_data:.2f}')
			res.append(list(data_sorted)) 

		plt.title(f'{file_name}')
		plt.xlabel(f"{metric}")
		plt.ylabel("CDF")
		plt.grid(True)
		plt.xlim( (xmin, xmax))
		plt.ylim((0, 1))
		plt.legend()
		plt.savefig(f'{FIG_DIR}/{file_name}-{metric}.png')
		  

def read_all_files(directory = '.'): 
# List to hold the contents of each text file
	file_list = [ 	]

	# Iterate over all files in the directory
	for filename in os.listdir(directory):
		# Check if the file ends with '.txt'
		if filename.endswith(".json"):
			# Construct the full path to the file
			file_path = os.path.join(directory, filename)
			file_list.append(file_path)
			
	# text_files_contents now contains the contents of all the text files in the directory
	# print(file_list)
	return file_list

def analyze_one_trace(file_name):
	log_data = []
	read_log(file_name, log_data) 
	qoe_dict = plt_accumulate_token_over_time(log_data, file_name)
	metric_dict = process_results(log_data)
	metric_dict.update(qoe_dict) 
	return metric_dict

if __name__ == "__main__":
	dir = './' # 'past/'
	file_list = [
'2024-10-12 18:43-microsoft-Phi-3.5-MoE-instruct-sharegpt-multi-gamma*249-1.2(10.0)-day--1-fcfs.json',
'2024-10-12 18:50-microsoft-Phi-3.5-MoE-instruct-sharegpt-multi-gamma*249-1.2(10.0)-day--1-qoe-avg.json'	]
	if not file_list:
		file_list = read_all_files()

	log_dict = {}
	for file in file_list:
		file = dir + file
		metric_dict = analyze_one_trace(file)
		log_dict[file] = metric_dict 

	plot_cdf_together(log_dict, ''.join(file_list[0].split('-')[3:9]))#


