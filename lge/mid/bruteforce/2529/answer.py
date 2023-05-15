from itertools import permutations

a = int(input())
l = input().split()

min_val = 1e11
max_val = -1

for item in permutations([i for i in range(10)], a+1):
    flag = True
    for i in range(len(l)):
        ineq_sign = l[i]
        if ineq_sign == "<" and item[i] >= item[i+1]:
            flag = False
            break
        if ineq_sign == ">" and item[i] <= item[i+1]:
            flag = False
            break
    
    if flag:
        item = list(map(str, item))
        min_val = min(min_val, int("".join(item)))
        max_val = max(max_val, int("".join(item)))

max_val = str(max_val)
min_val = str(min_val)

if len(max_val) != a+1:
    max_val = "0" * (a+1-len(max_val)) + max_val
if len(min_val) != a+1:
    min_val = "0" * (a+1-len(min_val)) + min_val

print(max_val)
print(min_val)