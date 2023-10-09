# first list: Q (num_of_orders)
# from second line, info_of_orders are given

# url is composed of <domain>/<question_id>

from collections import defaultdict
import heapq
from heapq import heappop, heappush

order_queue = []
heapq.heapify(order_queue)

# (idle/run: 0/1, number: 1~N)
device_dict = {}

idle_queue = []
heapq.heapify(idle_queue)

processing_domain_cnt = defaultdict(int)
domain_time_record = dict()


# device_dict
# k: device_number, v: tuple (priority, insert_time, domain, question_id)
def preparing(N, u0):
    """
    1) preparing devices
    100 N u0
    N devices and initial problem url is u0
    """
    global device_dict
    for key in range(1, N+1):
        device_dict[key] = (None, None)
        heappush(idle_queue, key)

    domain, question_id = u0.split("/")
    item = (1, 0, domain, question_id)
    heappush(order_queue, item)

    return

def requesting(t, p, u):
    """
    2) request scoring
    200 t p u
    at t-sec, 
    there is a request whose priority and url are p and u
    """

    # check duplicated url in queue
    is_duplicated = False
    for _, _, domain, qid in order_queue:
        if tuple(u.split("/")) == (domain, qid):
            is_duplicated = True
            break
    
    if not is_duplicated:
        domain, qid = u.split("/")
        item = (p, t, domain, qid)
        heappush(order_queue, item)
    
    return
    

def trying(t):
    """
    3) try scoring
    300 t
    at t-sec, if scoring is possible, 
    take task with highest priority and do scoring
    """
    global order_queue, device_dict

    fail_list = []

    best_task = None
    for _ in range(len(order_queue)):
        item = heappop(order_queue)
        _, _, domain, _ = item

        if processing_domain_cnt[domain] > 0:
            fail_list.append(item)
        elif domain not in domain_time_record.keys(): # very_first_domain
            best_task = item
            break
        else: # 
            latest_start, latest_end = domain_time_record[domain]
            threshold = latest_start + 3 * (latest_end - latest_start)

            if t < threshold: # inappropriate
                fail_list.append(item)
            else: # appropriate
                best_task = item
                break

    # restore order_queue
    for item in fail_list:
        heappush(order_queue, item)
    
    if best_task == None:
        return
    
    # idle_device_list = [key for key in device_dict.keys() \
    #     if device_dict[key] == None
    # ]
    if len(idle_queue) == 0:
        heappush(order_queue, best_task)
        return
    
    best_device = heappop(idle_queue)

    # if len(idle_device_list) == 0:
    #     heappush(device_queue, best_device)
    #     heappush(order_queue, best_task)
    #     return

    # idle_device_list.sort()
    target_device = best_device
    target_item = best_task

    device_dict[target_device] = (t, target_item[2])

    # updated_item = list(target_item)
    # updated_item[1] = t

    # heappush(device_queue, (1, best_device[1]))
    # device_dict[target_device] = tuple(updated_item)
    processing_domain_cnt[target_item[2]] += 1
    
    return

def terminating(t, J_id):
    """
    4) terminate scoring
    400 t J_id
    at t-sec, scoring process of J_id is terminated
    """
    # global device_dict

    if J_id in idle_queue:
        return

    task_start_time, task_domain = device_dict[J_id]
    # if device_dict[J_id] != None:
        # task_p, task_t, task_domain, task_qid = device_dict[J_id]
    processing_domain_cnt[task_domain] -= 1
    device_dict[J_id] = (t, None)

    heappush(idle_queue, J_id)
        
    domain_time_record[task_domain] = (task_start_time, t)

    return

def searching(t):
    """
    5) search queue
    500 t
    at t-sec, print num_of_tasks in queue
    """
    print(len(order_queue))

Q = int(input())
for _ in range(Q):
    info_of_order = input().split()
    order_number = int(info_of_order[0])
    if order_number == 100:
        N = int(info_of_order[1])
        u0 = info_of_order[2]
        preparing(N, u0)
    elif order_number == 200:
        t = int(info_of_order[1])
        p = int(info_of_order[2])
        u = info_of_order[3]
        requesting(t, p, u)
    elif order_number == 300:
        t = int(info_of_order[1])
        trying(t)
    elif order_number == 400:
        t = int(info_of_order[1])
        J_id = int(info_of_order[2])
        terminating(t, J_id)
    elif order_number == 500:
        t = int(info_of_order[1])
        searching(t)
    else:
        raise ValueError("Wrong order_number.")

        