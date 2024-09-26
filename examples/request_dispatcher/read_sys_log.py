import re
import os
import matplotlib.pyplot as plt
import sys
# Function to extract the necessary information from the log file
def extract_system_stats_from_log(file_path):
    # Lists to store extracted values
    timestamps = []
    running = []
    swapped = []
    waiting = []
    gpu_cache_usage = []
    
    # Regular expression to match the desired parts of the log lines
    pattern = re.compile(
        r'\[(\d+)\] System Stats: Running: (\d+),\s+- Swapped: (\d+),\s+- Waiting: (\d+),\s+- GPU Cache Usage: ([0-9.]+)'
    )
    
    # Read the file and extract data
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamps.append(int(match.group(1)))
                running.append(int(match.group(2)))
                swapped.append(int(match.group(3)))
                waiting.append(int(match.group(4)))
                gpu_cache_usage.append(float(match.group(5)))

    return timestamps, running, swapped, waiting, gpu_cache_usage



def plot_system_stats(timestamps, running, swapped, waiting, gpu_cache_usage, file_name):
    # Plotting the data with different scales for GPU Cache Usage and other metrics
    fig, ax1 = plt.subplots(figsize=(20, 6))
    timestamps = [t - timestamps[0] for t in timestamps]
    # Plot running, swapped, and waiting on the left y-axis (scale of 0 to 100)
    ax1.plot(timestamps, running, label='Running', color='blue' )
    ax1.plot(timestamps, swapped, label='Swapped', color='red' )
    ax1.plot(timestamps, waiting, label='Waiting', color='green' )

    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Queue Length', color='black')
    ymax = max(max(running), max(swapped), max(waiting))
    ax1.set_ylim([0, ymax])  # Scaling for running, swapped, and waiting

    # Add a second y-axis for GPU Cache Usage (scale of 0 to 1)
    ax2 = ax1.twinx()
    ax2.plot(timestamps, gpu_cache_usage, label='GPU Cache Usage', color='purple' )
    ax2.set_ylabel('GPU Cache Usage (%)', color='purple')
    ax2.set_ylim([0, 1])  # Scaling for GPU cache usage

    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Title and grid
    plt.title('System Stats Over Time')
    plt.grid(True)

    plt.tight_layout() 

    plt.savefig(f'fig/{file_name}-system_stats.png')


def visualize_system_stats(file_name: str): 
    log_file_path = f'{file_name}'
    timestamps, running, swapped, waiting, gpu_cache_usage = extract_system_stats_from_log(log_file_path)
    # print(timestamps, running, swapped, waiting, gpu_cache_usage)

    plot_system_stats(timestamps, running, swapped, waiting, gpu_cache_usage, file_name.split('/')[-1])
 

# Function to list files in a directory sorted by creation time
def list_files_by_creation_time(directory):
    # Get the list of files and directories with their full paths
    entries = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort the entries by creation time
    sorted_entries = sorted(entries, key=lambda x: os.path.getctime(x))

    return sorted_entries[-1]
