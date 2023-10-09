# first list: Q (num_of_orders)
# from second line, info_of_orders are given

# url is composed of <domain>/<question_id>

from collections import deque, defaultdict

order_queue = deque([])
# .popleft() # left out
# .append() # right in
device_dict = dict()

domain_time_record = dict()

# device_dict
# k: device_number, v: tuple (insert_time, priority, domain, question_id)
def preparing(N, u0):
    """
    1) preparing devices
    100 N u0
    N devices and initial problem url is u0
    """
    global device_dict
    # device
    for key in range(1, N+1):
        device_dict[key] = None

    domain, question_id = u0.split("/")
    item = (0, 1, domain, question_id)
    order_queue.append(item)

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
        order_queue.append((t, p, domain, qid))
    
    return
    

def trying(t):
    """
    3) try scoring
    300 t
    at t-sec, if scoring is possible, 
    take task with highest priority and do scoring
    """
    global order_queue, device_dict
    # 1) check possibility
    # 2) if possible, take ...
    initial_cands = list(order_queue.copy())
    rm_dup_domain_cands = []

    processing_domain_list = [value[2] for value in device_dict.values() \
        if value != None
    ]

    for item in initial_cands:
        _, _, domain, _ = item
        if domain in processing_domain_list:
            continue
        if domain not in domain_time_record.keys():
            pass
        else:
            latest_start, latest_end = domain_time_record[domain]
            threshold = latest_start + 3 * (latest_end - latest_start)

            if t < threshold:
                continue
        rm_dup_domain_cands.append(item)
    
    if len(rm_dup_domain_cands) == 0:
        return
    rm_dup_domain_cands.sort(key=lambda x: (x[1], x[0]))
    
    idle_device_list = [key for key in device_dict.keys() \
        if device_dict[key] == None
    ]

    if len(idle_device_list) == 0:
        return
    idle_device_list.sort()
    target_device = idle_device_list[0]
    target_item = rm_dup_domain_cands[0]

    updated_item = list(target_item)
    updated_item[0] = t

    device_dict[target_device] = tuple(updated_item)
    
    # update queue!
    updated_queue = [item for item in order_queue if item != target_item]
    order_queue = deque(updated_queue)
    return

def terminating(t, J_id):
    """
    4) terminate scoring
    400 t J_id
    at t-sec, scoring process of J_id is terminated
    """
    global device_dict
    if device_dict[J_id] != None:
        task_t, task_p, task_domain, task_qid = device_dict[J_id]
        device_dict[J_id] = None
        
        domain_time_record[task_domain] = (task_t, t)

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

        
