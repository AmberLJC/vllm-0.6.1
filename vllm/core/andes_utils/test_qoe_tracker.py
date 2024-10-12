import pytest
import numpy as np
from vllm.core.andes_utils.qoe_tracker import QoETracker

qoe_required = {
    'ttft': 1,
    'latency': 1,
}
time_list = [
            #   list(np.arange(0, 10, 0.2)), # 1
            #   list(np.arange(0, 10, 0.5)), # 1
            #  list(np.arange(5, 10, 0.2)), # 0.75
            #  list(np.arange(5, 20, 0.2)), # 0.75
            #  list(np.arange(10, 20, 0.2)),   #
            #  list(np.arange(10, 15, 0.1)),    #
              list(np.arange(0, 1, 0.2)) + list(np.arange(10, 11, 0.2)), # 0.85
              list(np.arange(0, 1, 0.2)) + list(np.arange(10, 12.5, 0.5)), # 0.85
             ] 
 
@pytest.mark.parametrize("time_list", time_list)
@pytest.mark.parametrize("qoe_required", [qoe_required])
def test_qoe_calculation(time_list, qoe_required):
    rt = QoETracker(qoe_required)
    qoe = rt.analyze_QoE(time_list)
    print(f'qoe: {qoe}. [{time_list}] ')    
    return qoe  

# time_list = [
#         [],
#         list(np.arange(0, 10, 0.2)),
#         list(np.arange(0, 100, 0.2)),
#         ]
# curtime_list = [1, 10,100]
# @pytest.mark.parametrize("time_list", time_list)
# @pytest.mark.parametrize("cur_time", curtime_list)
# def test_qoe_serve_preempt(time_list, cur_time):  
#     rt = QoETracker(qoe_required)
#     for time in time_list:
#         rt.add(time) 

#     value = rt.get_value( cur_time, 0.2, 100)
#     print(f"value ({len(time_list)}, {cur_time}) : {value}" ) 
