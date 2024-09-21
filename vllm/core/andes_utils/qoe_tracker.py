class ServiceTracker():
    def __init__(self, qoe_required: dict = None):
        pass

    def add(self, time_stamp):
        pass

    def get_priority(self, time_stamp: float, running: bool) -> float:
        pass

    def preempt_signal(self):
        pass

    def get_QoE(self, token_timestamp: list) -> float:
        pass

    def activate_status(self, time_stamp: float):
        pass
    
    def can_be_preempted(self) -> bool:
        pass

class QoETracker(ServiceTracker):
    def __init__(self, qoe_required: dict,  prompt_len: int = 0):
        self.buffer_size = 0
        self.qoe_required = qoe_required
        self.prompt_len = prompt_len
        self.display_rate = 1 / self.qoe_required['latency']
        self.buffer_size_list = [0]
        self.last_time = self.qoe_required['ttft']
        self.token_timestamp = [] # offset to 0  
        self.ttft_penalty = 1
        self.value = None
        self.obj_func = AvgQoEOptimizer(qoe_required)
        self.preemption_times = 0
            
    def add(self, time_stamp):   
        self.token_timestamp.append(time_stamp)
 
    def preempt_signal(self):
        self.preemption_times += 1
 
    def get_value(self, cur_time: float, token_latency: float, delta_t: float, running: bool = True) -> float:
        # return 10 if self.preemption_times and running or len(self.token_timestamp) > 500 \
        #          else self.obj_func.get_value(self.token_timestamp, cur_time, token_latency, delta_t, running)
        # return 10 if len(self.token_timestamp) > 500 \
        #          else self.obj_func.get_value(self.token_timestamp, cur_time, token_latency, delta_t, running)

        return self.obj_func.get_value(self.token_timestamp, cur_time, token_latency, delta_t, running)

    
    def get_QoE(self, token_timestamp: tuple = None, buffer_size_list: tuple = None, predict: bool = False) -> float:
        # TODO: add current time
        if not token_timestamp:
            token_timestamp = self.token_timestamp
        
        def _add(time_stamp): 
            self.buffer_size = max(0, self.buffer_size + 1 - (time_stamp - self.last_time) * self.display_rate)
            self.last_time = time_stamp
            self.buffer_size_list.append(self.buffer_size)

        if not buffer_size_list:
            for t in token_timestamp:
                _add(t)
            buffer_size_list = self.buffer_size_list
        
        if len(token_timestamp) == 0:
            return 1
    
        token_timestamp = list(token_timestamp)
        token_timestamp.insert(0, self.qoe_required['ttft'])
        s_actual = 0
 
        for i in range(2, len(token_timestamp)):
            delta_t = self.qoe_required['latency'] if buffer_size_list[i] > 0  else token_timestamp[i] - token_timestamp[i-1]
            s_actual += (2 * i -1) * delta_t / 2

        N = len(token_timestamp) - 1
        if predict:
            s_target = (token_timestamp[-1] - self.qoe_required['ttft']) ** 2 * self.display_rate / 2
        elif token_timestamp[-1] < self.qoe_required['ttft'] + N / self.display_rate:
            s_target = N * N * self.qoe_required['latency'] / 2 
        else:
            ttlt = token_timestamp[-1] + buffer_size_list[-1] * self.qoe_required['latency']
            s_target = N * (2 * ttlt - 2 * self.qoe_required['ttft'] - N * self.qoe_required['latency']) / 2  
        return min(1, s_actual / s_target) * self.ttft_penalty ** max(token_timestamp[1]-token_timestamp[0], 0) if s_target > 0 else 0

    def analyze_QoE(self, token_timestamp: list) -> float:
        # token_timestamp: start from TTFT
        # for post processing 
        # TODO: not sensitive to long TTFT under 
        for t in token_timestamp:
            self.add(t)
        
        return self.get_QoE(tuple(token_timestamp))
    
class QoEOptimizer():
    def __init__(self, qoe_required) -> None:
        self.qoe_required = qoe_required 
        self.display_rate = 1 / self.qoe_required['latency']
        self.record = -1
    
    def cal_qoe_serve(self, token_timestamp, cur_time: float, token_latency: float, delta_t: float) -> float:
        pass
                      

    def cal_qoe_preempt(self, token_timestamp, cur_time: float,  delta_t: float = None) -> float:
        pass

    def get_value(self, token_timestamp, cur_time: float, token_latency: float, delta_t: float, running: bool = True ) -> float:
        pass

 
class AvgQoEOptimizer(QoEOptimizer):  
    def get_value(self, token_timestamp, cur_time: float, token_latency: float, delta_t: float, running: bool = True) -> float:
        expected_response_len =( cur_time + delta_t) * self.display_rate
        length = len(token_timestamp) 

        if length > 0 and token_timestamp[-1] < self.qoe_required['ttft'] + length * self.qoe_required['latency'] and \
            token_latency <= self.qoe_required['latency']: 
            qoe_serve = 1
        else:
            actual_response_len = length + delta_t / token_latency
            qoe_serve = 1 - ((expected_response_len - actual_response_len) / expected_response_len) ** 2

        if length == 0:
            qoe_preempt = cur_time + delta_t <= self.qoe_required['ttft']  
        else:
            qoe_preempt =  1 - ((expected_response_len-length) / expected_response_len) ** 2
        
        return (qoe_serve - qoe_preempt) * (1 + 0.5 * running)
