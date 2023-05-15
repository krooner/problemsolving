from itertools import permutations

N = int(input())

def operation(option, a, b):
    if option == 0:
        return a+b
    elif option == 1:
        return a-b
    elif option == 2:
        return a*b
    else:
        if a < 0:
            return ((-a)//b) * -1
        else:
            return a//b


numbers = list(map(int, input().split()))
ops = list(map(int, input().split()))

l = []
for i in range(len(ops)):
    l += [i for j in range(ops[i])]

maxval = -1e10
minval = 1e10

for case in permutations(l, len(l)):
    left_operand = numbers[0]
    for i in range(1, len(numbers)):
        left_operand = operation(case[i-1], left_operand, numbers[i])
    maxval = max(maxval, left_operand)
    minval = min(minval, left_operand)

print(maxval)
print(minval)
        

