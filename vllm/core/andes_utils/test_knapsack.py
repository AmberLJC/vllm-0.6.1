import time
from typing import Optional, Literal
from collections import deque
import random
from knapsack_solver import KnapSack

# Assuming KnapSack class is imported here

def test_knap_sack_correctness():
    weights = [1, 2, 3]
    # weights = [2, 2, 2]
    values = [60, 100, 120]
    capacity = 5
    knap_sack_greedy = KnapSack(block_size=10, total_available_blocks=5, solver='greedy')
    result = knap_sack_greedy.solver_func(capacity, weights, values)
    print(f"Greedy solver: {result[0]}, Selected items: {result[1]}")

def test_knap_sack_efficiency():
    # Initialize test case parameters
    small_capacity = 50
    large_capacity = 1000
    small_item_count = 10
    large_item_count = 1000

    # Create random weights and values for small test case
    small_weights = [random.randint(1, 10) for _ in range(small_item_count)]
    small_values = [random.randint(1, 100) for _ in range(small_item_count)]
    
    # Create random weights and values for large test case
    large_weights = [random.randint(1, 100) for _ in range(large_item_count)]
    large_values = [random.randint(1, 500) for _ in range(large_item_count)]

    # Initialize the KnapSack object
    knap_sack_greedy = KnapSack(block_size=10, total_available_blocks=100, solver='greedy')

    # Measure time for small test case
    start_time = time.time()
    result_small = knap_sack_greedy.solver_func(small_capacity, small_weights, small_values)
    small_duration = time.time() - start_time
    print(f"Greedy solver (small test): {result_small[0]}, Time taken: {small_duration:.6f} seconds")

    # Measure time for large test case
    start_time = time.time()
    result_large = knap_sack_greedy.solver_func(large_capacity, large_weights, large_values)
    large_duration = time.time() - start_time
    print(f"Greedy solver (large test): {result_large[0]}, Time taken: {large_duration:.6f} seconds")

    knap_sack_greedy = KnapSack(block_size=10, total_available_blocks=100, solver='dp')

    # Measure time for small test case
    start_time = time.time()
    result_small = knap_sack_greedy.solver_func(small_capacity, small_weights, small_values)
    small_duration = time.time() - start_time
    print(f"DP solver (small test): {result_small[0]}, Time taken: {small_duration:.6f} seconds")

    # Measure time for large test case
    start_time = time.time()
    result_large = knap_sack_greedy.solver_func(large_capacity, large_weights, large_values)
    large_duration = time.time() - start_time
    print(f"DP solver (large test): {result_large[0]}, Time taken: {large_duration:.6f} seconds")


import unittest
from unittest.mock import MagicMock
import time
from knapsack_solver import KnapSack
class TestPickRequests(unittest.TestCase):

    def setUp(self):
        # Initialize the object containing pick_requests function
        self.system = KnapSack(16, 1, 'greedy')  
        self.system.unit_overhead = 0.1
        self.system.token_latency = 0.05
        self.system.delta_t = 1
        self.system.solver_func = MagicMock(return_value=(None, [0, 1, 2]))  # Mock solver function

    def create_mock_request(self, req_len, slack_value, running_value, waiting_value):
        """Helper function to create a mock request object."""
        request = MagicMock()
        request.get_len.return_value = req_len
        request.get_slack.return_value = slack_value
        request.get_value.side_effect = lambda now, token_latency, delta_t, is_running: running_value if is_running else waiting_value
        return request
    
    @staticmethod
    def print_request_details(request):
        print(f"Request length: {request.get_len()}") 

    def test_preemption_case(self):
        now = time.monotonic()

        # Create mock requests
        running_request1 = self.create_mock_request(req_len=10, slack_value=10, running_value=.5, waiting_value=.5)
        waiting_request1 = self.create_mock_request(req_len=12, slack_value=6, running_value=1, waiting_value=1)
        swapped_request1 = swapped_request2 = self.create_mock_request(req_len=20, slack_value=6, running_value=1, waiting_value=1)
        
        # Test the pick_requests function
        selected_requests = self.system.pick_requests(
            running=[running_request1],
            waiting=[waiting_request1],
            swapped=[swapped_request1, swapped_request2]
        )
 
        for request in selected_requests:
            self.print_request_details(request)

if __name__ == '__main__':
    unittest.main()
 
    # test_knap_sack_efficiency()
    # test_knap_sack_correctness()
