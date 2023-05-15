N, S = list(map(int, input().split()))

sequence = list(map(int, input().split()))

answer = 0
def recursion(index, summation):
    global answer
    if index == N:
        if summation == S:
            answer += 1 
        return

    recursion(index+1, summation+sequence[index])
    recursion(index+1, summation)

recursion(0, 0)
if S == 0: answer -= 1
print(answer)