from itertools import combinations, permutations

N = int(input())
A = list(map(int, input().split()))

# +, -, x, /
op_cnt = list(map(int, input().split()))

operators = []
for i in range(len(op_cnt)):
    operators += [i for _ in range(op_cnt[i])]

minval = 1e10
maxval = -1e10

for item in combinations(operators, len(A)-1):
    item = list(item)
    for ordered_item in permutations(item, len(item)):
        lo = A[0]
        for i in range(len(ordered_item)):
            ro = A[i+1]
            op = ordered_item[i]
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

print(maxval)
print(minval)

