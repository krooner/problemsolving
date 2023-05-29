N = int(input())

weights = list(map(int, input().split()))

maxpower = -1

def solve(weights, power):
    global maxpower
    if len(weights) == 2:
        maxpower = max(maxpower, power)
    
    length = len(weights)

    for i in range(1, len(weights)-1):
        gathering = [weights[j] for j in range(len(weights)) if j != i]

        solve(gathering, power+weights[i-1]*weights[i+1])
        


solve(weights, 0)
print(maxpower)