import pytest
from vllm.core.andes_utils.qoe_tracker import QoETracker

qoe_required = {
    'ttft': 1,
    'latency': 1,
}
time_list = [
             [0, 1, 2, 3],  # 1
            [],
            #  [10, 11, 12, 13], # 
            #  [1,2,3,7,8,9,10,11,12 ], # .8
            #  [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,5.5,6,6.5,7,7.5,8], # 1
            #  [ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            #  [ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            #  [ 8,9,10, 11, 12, 13, 14, 15, 16, 17],
            #  [ 6,7,8,9,10, 11, 12, 13, 14, 15],
            # [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,5.5,6,6.5],
            # [*range(20,120,1)],
            # [*range(30,130,1)],
            # [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16,16.5],
            # [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, ],
            # [10, 10.5, 11, 11.5, 12, 12.5 ], # 17.5/66
             ] 


# @pytest.mark.parametrize("time_list", time_list)
# @pytest.mark.parametrize("qoe_required", [qoe_required])
# def test_qoe_calculation(time_list, qoe_required):
#     rt = QoETracker(qoe_required)
#     for time in time_list:
#         rt.add(time)
#     qoe = rt.get_QoE(tuple(time_list))
#     print(qoe)    
#     return qoe  

curtime_list = [3 , 8, 16, 32, 64]

@pytest.mark.parametrize("time_list", time_list)
@pytest.mark.parametrize("cur_time", curtime_list)
def test_qoe_serve_preempt(time_list, cur_time): 
    # if cur_time < time_list[-1]:
    #     return
    rt = QoETracker(qoe_required)
    for time in time_list:
        rt.add(time) 

    value = rt.get_value(cur_time, 1, 10)
    print(f"value ({time_list}, {cur_time}) : {value}" )
    # value = rt.get_value(cur_time, 1, 5)
    # print(f"value ({time_list}, {cur_time}) : {value}" )

