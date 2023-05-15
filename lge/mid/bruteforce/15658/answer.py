from itertools import combinations, permutations

N = int(input())
A = list(map(int, input().split()))

# +, -, x, /
op_cnt = list(map(int, input().split()))

# operators = []
# for i in range(len(op_cnt)):
#     operators += [i for _ in range(op_cnt[i])]

minval = 1e10
maxval = -1e10

def recursion(l, idx, container):
    global minval, maxval
    if idx == len(A)-1:
        lo = A[0]
        for i, op in enumerate(container):
            ro = A[i+1]
            if op == 0: lo += ro
            elif op == 1: lo -= ro
            elif op == 2: lo *= ro
            else:
                if lo < 0:
                    q = (-lo)//ro
                    lo = -q
                else:
                    lo = lo//ro            
        minval = min(minval, lo)
        maxval = max(maxval, lo)
        return
    
    for i in range(len(l)):
        if l[i] > 0:
            l[i] -= 1
            recursion(l, idx+1, container + [i])
            l[i] += 1

recursion(op_cnt, 0, [])

# for item in combinations(operators, len(A)-1):
#     item = list(item)
#     for ordered_item in permutations(item, len(item)):
#         lo = A[0]
#         for i in range(len(ordered_item)):
#             ro = A[i+1]
#             op = ordered_item[i]
#             if op == 0: lo += ro
#             elif op == 1: lo -= ro
#             elif op == 2: lo *= ro
#             else:
#                 if lo < 0:
#                     q = (-lo)//ro
#                     lo = -q
#                 else:
#                     lo = lo//ro
#         minval = min(minval, lo)
#         maxval = max(maxval, lo)

print(maxval)
print(minval)