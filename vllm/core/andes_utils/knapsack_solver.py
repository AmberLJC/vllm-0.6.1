from typing import Optional, Literal
from collections import deque
import time
import heapq

class KnapSack():
    def __init__(self, 
                 block_size = int, 
                 total_available_blocks = int, 
                 solver: Literal['greedy', 'dp'] = 'greedy',
                 ) -> None:
        
        if solver == 'dp':
            self.solver_func = KnapSack._dp_knapsack
        elif solver == 'greedy':
            self.solver_func = KnapSack._greedy_knapsack
        else:
            raise ValueError(f"Solver {self.solver} is not supported")
        # TODO: move this to somewhere else
        self.delta_t = 50
        self.block_size = block_size 
        self.total_available_blocks = total_available_blocks 
        self.token_latency = 0.05 
        self.max_num_preempt = 2
    import time

    def pick_requests(self, running, waiting, swapped): 
        # Helper functions to get context block for list and individual requests
        def get_context_block_list(requests):
            return [round((r.get_len() + 1) / self.block_size + 0.5) for r in requests]
        
        # Convert input lists to mutable lists (if not already)
        running = list(running)
        waiting = list(waiting)
        swapped = list(swapped)

        now = time.monotonic()

        # Precompute run and pause values for running, waiting, and swapped
        run_value_list = [r.get_value(now, self.token_latency, self.delta_t, True) for r in running] 
        pause_value_list = [r.get_value(now, self.token_latency, self.delta_t, False) for r in waiting + swapped]

        # Precompute max value for preemption decisions
        max_pause_value = max(pause_value_list, default=1)

        # Determine items that could be preempted (those with value <= max_pause_value)
        maybe_preempt = [(i, r) for i, r in enumerate(running) if run_value_list[i] <= max_pause_value][:self.max_num_preempt]

        # Use set for efficient index checking when keeping non-preempted running items
        preempt_indices = {i for i, _ in maybe_preempt}
        keep_running = [r for i, r in enumerate(running) if i not in preempt_indices]

        # Update running with preempted items only
        running = [r for _, r in maybe_preempt]

        # Calculate available blocks after keeping some running requests
        available_blocks = self.total_available_blocks - sum(get_context_block_list(keep_running))

        # Prepare context blocks and values for the knapsack problem
        context_block_list = get_context_block_list(running + waiting + swapped)
        value_list = [run_value_list[i] for i in preempt_indices] + pause_value_list

        # Solve the knapsack problem to get the best plan
        _, best_plan = self.solver_func(available_blocks, context_block_list, value_list)

        # Return the best plan along with items that were kept running
        return [(running + waiting + swapped)[i] for i in best_plan] + keep_running

        # context_block_list = get_context_block_list(running + waiting + swapped)
        # value_list = [r.get_value(now, self.token_latency, self.delta_t, True) for r in running] 
        # value_list += [r.get_value(now, self.token_latency, self.delta_t, False) for r in waiting + swapped] 
        
        # max_value, best_plan = self.solver_func(self.total_available_blocks, context_block_list, value_list )
        # return [(running + waiting + swapped)[i] for i in best_plan]  

    def schedule_requests(self, budget, running, waiting, swapped, utilization, latency_function):
        # TODO: consider budget
        if utilization < 0.9:
            return deque(waiting), deque(swapped), deque() 
        
        new_running = self.pick_requests(running, waiting, swapped)
        seq_to_admit = [r for r in new_running if r in waiting]
        seq_to_swap_in = [r for r in new_running if r in swapped]
        seq_to_evict = [r for r in running if r not in new_running]
        
        return deque(seq_to_admit), deque(seq_to_swap_in), deque(seq_to_evict)

    @staticmethod
    def _greedy_knapsack(capacity: int,  weights: list, values: list) -> tuple:
        """
        Greedy algorithm for the fractional knapsack problem.

        Args:
        capacity (int): Maximum weight the knapsack can carry.
        weights (list): List of weights of the items.
        values (list): List of values of the items.

        Returns:
        tuple: Total value of the picked items and the list of indices of the picked items.
        """

        # Calculate value-to-weight ratio and sort items by this ratio in descending order
        items = [(v / w, w, v, i) for i, (v, w) in enumerate(zip(values, weights))]
        items.sort(reverse=True, key=lambda x: x[0]) 

        current_weight = 0
        current_value = 0
        picked_items = []

        for _, weight, value, index in items:
            if current_weight + weight <= capacity:  
                current_weight += weight
                current_value += value
                picked_items.append(index)
            else:
                if current_weight / capacity < 0.9:
                    continue
                else:
                    break

        return current_value, picked_items

    @staticmethod
    def _dp_knapsack(M, B, deltaQ):
        l = N = len(deltaQ) 
        dp = [[[-float('inf') for _ in range(M+1)] for _ in range(B+1)] for _ in range(N+1)]
        choice = [[[0 for _ in range(M+1)] for _ in range(B+1)] for _ in range(N+1)]
        dp[0][0][0] = 0  # Base case

        # Main DP loop
        for i in range(1, N+1):
            for b in range(min(i, B)+1):  # +1 because range is exclusive
                for m in range(M+1):
                    # Case 1: Do not select the i-th request
                    if dp[i][b][m] < dp[i-1][b][m]:
                        dp[i][b][m] = dp[i-1][b][m]
                        choice[i][b][m] = 0

                    # Case 2: Select the i-th request, if it does not exceed the constraints
                    if b > 0 and m + l[i-1] <= M and dp[i][b][m + l[i-1]] < dp[i-1][b-1][m] + deltaQ[i-1]:
                        dp[i][b][m + l[i-1]] = dp[i-1][b-1][m] + deltaQ[i-1]
                        choice[i][b][m + l[i-1]] = 1

        # Find the optimal solution and corresponding memory
        max_dealtQ, current_m = max((value, index) for index, value in enumerate(dp[N][B]))

        # Backtrack to find the decision array x_i
        x = [0] * (N+1)  # Initialize decision array with zeros
        current_b = B
        for i in range(N, 0, -1):
            x[i] = choice[i][current_b][current_m]
            if x[i] == 1:
                current_m -= l[i-1]
                current_b -= 1

        picked_requests = [i for i in range(1, N+1) if x[i] == 1] 
        return  max_dealtQ, picked_requests
