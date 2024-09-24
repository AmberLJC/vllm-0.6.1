import time
from typing import Optional, Literal
from collections import deque
import random
from knapsack_solver import KnapSack

# Assuming KnapSack class is imported here

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
    result_small = knap_sack_greedy.knap_sack(small_capacity, small_item_count, small_weights, small_values)
    small_duration = time.time() - start_time
    print(f"Greedy solver (small test): {result_small[0]}, Time taken: {small_duration:.6f} seconds")

    # Measure time for large test case
    start_time = time.time()
    result_large = knap_sack_greedy.knap_sack(large_capacity, large_item_count, large_weights, large_values)
    large_duration = time.time() - start_time
    print(f"Greedy solver (large test): {result_large[0]}, Time taken: {large_duration:.6f} seconds")



    knap_sack_greedy = KnapSack(block_size=10, total_available_blocks=100, solver='dp')

    # Measure time for small test case
    start_time = time.time()
    result_small = knap_sack_greedy.knap_sack(small_capacity, small_item_count, small_weights, small_values)
    small_duration = time.time() - start_time
    print(f"DP solver (small test): {result_small[0]}, Time taken: {small_duration:.6f} seconds")

    # Measure time for large test case
    start_time = time.time()
    result_large = knap_sack_greedy.knap_sack(large_capacity, large_item_count, large_weights, large_values)
    large_duration = time.time() - start_time
    print(f"DP solver (large test): {result_large[0]}, Time taken: {large_duration:.6f} seconds")

    # Optionally: Add similar tests for 'dp' solver if implemented in future
    
if __name__ == "__main__":
    test_knap_sack_efficiency()
