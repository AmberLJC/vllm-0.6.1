import time
from functools import lru_cache
from knapsack_solver import KnapSack
import numpy as np
np.random.seed(3)

# Mock request class
class MockRequest:
    def __init__(self, id, length, slack, value):
        self.id = id
        self.length = length
        self.slack = slack
        self.value = value
    
    def get_len(self):
        return self.length

    def get_slack(self, now):
        return self.slack

    def get_value(self, now, token_latency, delta_t, is_running):
        # Simplified value calculation for testing purposes
        return self.value

# Test case for `pick_requests`
def test_pick_requests():
    # Set up the mock environment
    self = type('', (), {})()  # Creating a simple instance with attributes
    self.block_size = 1
    self.unit_overhead = 1
    self.token_latency = 2
    self.delta_t = 1
    self.total_available_blocks = 250

    knapsack = KnapSack(self.block_size, self.total_available_blocks, 'greedy', self.delta_t)
    knapsack.percentile_to_sacrifice = 0.2

    size_list = [50,30,20]
    # Create lists of mock requests
    length = np.random.randint(1, 10, sum(size_list))
    slacks = np.random.randint(-1, 10, sum(size_list))
    values = np.random.randint(0, 5, sum(size_list))
    print(f'Capacity: {self.total_available_blocks}. Used length: {sum(length[:50])} / {sum(length)}')
    print(f'Slacks: {list(slacks)}')
    print(f'Length: {list(length)}')
    running = [MockRequest(id=i, length=length[i], slack=slacks[i], value = values[i]) for i in range(size_list[0])]
    waiting = [MockRequest(id=i+50, length=length[i+50], slack=slacks[i+50], value = values[i+50]) for i in range(size_list[1])]
    swapped = [MockRequest(id=i+80, length=length[i+80], slack=slacks[i+80], value = values[i+80]) for i in range(size_list[2])]
    print(f'========== Start Solver ==========')
    # Measure correctness by checking output structure
    result = knapsack.pick_requests( running, waiting, swapped)
    print(f'KnapSack max_num_preempt: {knapsack.max_num_preempt}')
    print(f'Pick: {[(res.id, res.length) for res in result]}')
    print(f'Num of Requests: {len(result)}.  \
          Used Length:  {sum([res.length for res in result])+len(result)}.  \
          Newly admitted: {len([res for res in result if res.id > 50])}.')

    ground_truth = [(10, 1), (27, 1), (35, 1), (54, 1), (61, 1), (79, 1), (87, 1), (90, 1), (4, 1), (52, 1), (55, 1), (59, 1), (36, 1), (53, 1), (71, 1), (80, 1), (86, 1), (22, 2), (37, 3), (69, 1), (46, 2), (57, 3), (65, 3), (73, 2), (97, 3), (99, 2), (14, 2), (16, 3), (18, 2), (26, 2), (62, 2), (64, 2), (66, 3), (75, 2), (81, 2), (95, 2), (6, 4), (29, 5), (32, 5), (19, 4), (50, 5), (85, 5), (98, 5), (31, 6), (39, 6), (44, 2), (45, 2), (56, 3), (70, 7), (94, 3), (1, 4), (11, 5), (38, 5), (67, 5), (89, 4), (28, 6), (2, 9), (24, 8)]    # # Measure speed
    assert  ground_truth == [(res.id, res.length) for res in result]
    start_time = time.time()
    for _ in range(1000):  # Run multiple times to test speed
        knapsack.pick_requests( running, waiting, swapped)
    duration = time.time() - start_time
    print(f"Execution time for 1000 runs: {duration:.4f} seconds")

# Run the test case
test_pick_requests()

# Num of Requests: 58.            Used Length:  226.            Newly admitted: 31.
# Execution time for 1000 runs: 0.3621 seconds
# Execution time for 1000 runs: 0.3570 seconds
# Execution time for 1000 runs: 0.3067 seconds

# DP
# Execution time for 1000 runs: 8.9303 seconds