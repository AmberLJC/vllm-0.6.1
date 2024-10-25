import time
from collections import deque
from vllm.core.andes_utils.knapsack_solver import KnapSack
from functools import lru_cache


class LeastQoEFirst(KnapSack):
    def __init__(self, 
                 block_size = int, 
                 total_available_blocks = int, 
                 solver = 'greedy',
                 delta_t: int = 10,
                 ) -> None:
        super().__init__(block_size, total_available_blocks, solver, delta_t)

    def pick_requests(self, running, waiting, swapped): 
        # Helper functions to get context block for list and individual requests
        def get_context_block_list(requests):
            return [get_context_block(r) for r in requests]
         
        @lru_cache(maxsize=100)
        def get_context_block(request):
            return round((request.get_len() + 1) / self.block_size + 0.5)
        
        # Convert input lists to mutable lists (if not already)
        running = list(running)
        waiting = list(waiting)
        swapped = list(swapped)

        now = time.monotonic()

        # Precompute run and pause values for running, waiting, and swapped
        run_value_list = [r.get_value(now, self.token_latency, self.delta_t, True)  for r in running] 
        pause_value_list = [r.get_value(now, self.token_latency, self.delta_t, False) for r in waiting + swapped]
        maybe_preempt = [(i, r) for i, r in enumerate(running) if run_value_list[i] <= max(pause_value_list, default=1)][:self.max_num_preempt]

        # Use set for efficient index checking when keeping non-preempted running items
        preempt_indices = {i for i, _ in maybe_preempt}
        keep_running = [r for i, r in enumerate(running) if i not in preempt_indices]
        running = [r for _, r in maybe_preempt]
 
        available_blocks = self.total_available_blocks - sum(get_context_block_list(keep_running))
        context_block_list = [1 for _ in range(len(running + waiting + swapped))]
        value_list = [run_value_list[i] for i in preempt_indices] + pause_value_list 
        _, best_plan = self.solver_func(available_blocks, context_block_list, value_list)
        return [(running + waiting + swapped)[i] for i in best_plan] + keep_running 
    
    def schedule_requests(self, budget, running, waiting, swapped, utilization, latency_function):
        if utilization < 0.9:
            return deque(waiting), deque(swapped), deque()
        
        new_running = self.pick_requests(running, waiting, swapped)
        seq_to_admit = [r for r in new_running if r in waiting]
        seq_to_swap_in = [r for r in new_running if r in swapped]
        seq_to_evict = [r for r in running if r not in new_running]
        
        return deque(seq_to_admit), deque(seq_to_swap_in), deque(seq_to_evict)
