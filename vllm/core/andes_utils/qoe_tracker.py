class ServiceTracker():
    def __init__(self, qoe_required: dict = None):
        pass

    def add(self, time_stamp):
        pass

    def get_priority(self, time_stamp: float, running: bool) -> float:
        pass

    def get_QoE(self, token_timestamp: list) -> float:
        pass

    def activate_status(self, time_stamp: float):
        pass
    
    def can_be_preempted(self) -> bool:
        pass

class QoETracker(ServiceTracker):
    def __init__(self, qoe_required: dict,  prompt_len: int = 0, obj_func: str = 'avg'):
        self.buffer_size = 0
        self.qoe_required = qoe_required
        self.prompt_len = prompt_len
        self.display_rate = 1 / self.qoe_required['latency']
        self.buffer_size_list = [0]
        self.last_time = self.qoe_required['ttft']
        self.token_timestamp = [] # offset to 0  
        self.ttft_penalty = 1
        self.value = None
        if obj_func == 'avg':
            self.obj_func = AvgQoEOptimizer(qoe_required)
        elif obj_func == 'min':
            self.obj_func = MinQoEOptimizer(qoe_required)
        else:
            raise ValueError(f"Unsupported objective function: {obj_func}") 
            
    def add(self, time_stamp):   
        self.token_timestamp.append(time_stamp) 
 
    def get_value(self, cur_time: float, token_latency: float, delta_t: float, running: bool = False) -> float:
        # return 10 if self.preemption_times and running or len(self.token_timestamp) > 500 \
        #          else self.obj_func.get_value(self.token_timestamp, cur_time, token_latency, delta_t, running)
        # return 10 if len(self.token_timestamp) > 500 \
        #          else self.obj_func.get_value(self.token_timestamp, cur_time, token_latency, delta_t, running)

        return self.obj_func.get_value(self.token_timestamp, cur_time, token_latency, delta_t, running)

    def get_QoE(self, token_timestamp: list = None, buffer_size_list: tuple = None, predict: bool = False) -> float:
        # TODO: add current time
        if not token_timestamp:
            token_timestamp = self.token_timestamp
        
        s_gap = 0
        y = list(range(len(token_timestamp)))
        expected_timeline = [yi * self.qoe_required['latency'] + self.qoe_required['ttft'] for yi in y]
        
        user_ts = token_timestamp[0]

        for i in range( len(token_timestamp)):
            s_gap += max(user_ts - expected_timeline[i], 0)
            user_ts = max(token_timestamp[i], user_ts + self.qoe_required['latency'])
        s_target = (2 * user_ts - 2*self.qoe_required['ttft'] - len(token_timestamp) * self.qoe_required['latency']) * len(token_timestamp)  / 2
        return min(1, 1- s_gap / s_target)

    def analyze_QoE(self, token_timestamp: list) -> float: 
        return self.get_QoE(token_timestamp)
    
    def get_slack(self, cur_time: float) -> float:
        response_len = len(self.token_timestamp)  
        expected_ts = response_len * self.qoe_required['latency'] + self.qoe_required['ttft']
        return expected_ts - cur_time
    
class QoEOptimizer():
    def __init__(self, qoe_required) -> None:
        self.qoe_required = qoe_required 
        self.display_rate = 1 / self.qoe_required['latency']
        self.recorded_value = 1
    
    def cal_qoe_serve(self, token_timestamp, cur_time: float, token_latency: float, delta_t: float) -> float:
        pass
                      

    def cal_qoe_preempt(self, token_timestamp, cur_time: float,  delta_t: float = None) -> float:
        pass

    def get_value(self, token_timestamp, cur_time: float, token_latency: float, delta_t: float, running: bool = True ) -> float:
        pass

 
class AvgQoEOptimizer(QoEOptimizer):  
    def get_value(self, token_timestamp, cur_time: float, token_latency: float, delta_t: float, running: bool = False) -> float:
        expected_response_len = (cur_time + delta_t) * self.display_rate
        length = len(token_timestamp) 

        if length > 0 and token_timestamp[-1] < self.qoe_required['ttft'] + length * self.qoe_required['latency']:
            qoe_serve = 1
        else:
            actual_response_len = length + delta_t / max(self.qoe_required['latency'], token_latency)
            qoe_serve = min(1, (actual_response_len / expected_response_len) ** 2)

        if length == 0:
            qoe_preempt = (cur_time + delta_t ) <= self.qoe_required['ttft']  
        elif expected_response_len <= length:
            qoe_preempt = 1
        else:
            qoe_preempt =  1 - ((expected_response_len - length) / expected_response_len) ** 2
        # print(f'qoe_serve: {qoe_serve}, qoe_preempt: {qoe_preempt}')
        return (qoe_serve - qoe_preempt) * (1+running*0.1)

 
class MinQoEOptimizer(QoEOptimizer):  
    def get_value(self, token_timestamp, cur_time: float, token_latency: float, delta_t: float, running: bool = False) -> float:
        if self.recorded_value == 0:
            return 0

        length = len(token_timestamp) 
        # check if there is hope
        if length == 0 and cur_time > self.qoe_required['ttft']:
            self.recorded_value = 0
            return 0
        elif length > 0 and token_timestamp[-1] > self.qoe_required['ttft'] + length * self.qoe_required['latency']:
            self.recorded_value = 0
            return 0

        expected_response_len = (cur_time + delta_t) * self.display_rate
        # still have hope
        if length == 0:
            qoe_preempt = (cur_time + delta_t ) <= self.qoe_required['ttft']  
        elif expected_response_len <= length:
            qoe_preempt = 1
        else:
            qoe_preempt =  1 - ((expected_response_len - length) / expected_response_len) ** 2

        return 1 - qoe_preempt
