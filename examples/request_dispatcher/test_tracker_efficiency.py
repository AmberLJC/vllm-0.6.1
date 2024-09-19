import time

class AvgQoEOptimizerTest:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def run_efficiency_test(self, test_cases, num_iterations=1000):
        total_time = 0
        for i, test_case in enumerate(test_cases):
            token_timestamp, cur_time, token_delivery_speed, delta_t = test_case
            start_time = time.time()
            for _ in range(num_iterations):
                self.optimizer.get_value(token_timestamp, cur_time, token_delivery_speed, delta_t)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            print(f"Test Case {i + 1}: Time for {num_iterations} iterations: {elapsed_time:.6f} seconds")
        avg_time = total_time / len(test_cases)
        print(f"\nAverage time per test case: {avg_time:.6f} seconds")

# Test Setup
if __name__ == "__main__":
    # Dummy QoEOptimizer class definition for illustration
    class QoEOptimizer:
        def __init__(self):
            self.qoe_required = {'ttft': 0.5, 'latency': 0.2}
            self.display_rate = 10.0
            self.value = 0 
            
    class AvgQoEOptimizer(QoEOptimizer):
 
        def get_value(self, token_timestamp, cur_time: float, token_latency: float, delta_t: float, running: bool = True) -> float:
            expected_response_len =( cur_time + delta_t) * self.display_rate
            length = len(token_timestamp) 

            if length > 0 and token_timestamp[-1] < self.qoe_required['ttft'] + len(token_timestamp) * self.qoe_required['latency'] and \
                token_latency <= self.qoe_required['latency']: 
        
                qoe_serve = 1
            else:
                actual_response_len = length + delta_t / token_latency
                qoe_serve = 1 - ((expected_response_len - actual_response_len) / expected_response_len) ** 2

            if length == 0:
                qoe_preempt = cur_time + delta_t <= self.qoe_required['ttft']  
            else:
                qoe_preempt =  1 - ((expected_response_len-length) / expected_response_len) ** 2
            
            return qoe_serve - qoe_preempt


    optimizer = AvgQoEOptimizer()

    # Define test cases
    test_cases = [
        ([*range(100)], 1.0, 1.5, 0.5),  # Sample test case 1
        ([*range(200)], 1.0, 1.5, 0.5),  # Sample test case 1 
        ([*range(300)], 1.0, 1.5, 0.5),  # Sample test case 1
    ]

    # Initialize the test class
    qoe_test = AvgQoEOptimizerTest(optimizer)

    # Run efficiency test
    qoe_test.run_efficiency_test(test_cases, num_iterations=10000)  # Increase iterations for better accuracy

