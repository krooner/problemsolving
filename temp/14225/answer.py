N = int(input())
S = list(map(int, input().split()))


answers = []
def recursion(idx, partial_sum, cnt):
    if idx == len(S):
        if cnt != 0:
            answers.append(partial_sum)
        return
    
    # If choose
    recursion(idx+1, partial_sum+S[idx], cnt+1)
    # Otherwise
    recursion(idx+1, partial_sum, cnt)

recursion(0, 0, 0)

answers = sorted(set(answers))

returned = False
for i, item in enumerate(answers):
    if i == 0:
        if item != 1:
            returned = True
            print(1)
            break
    else:
        if item-1 != answers[i-1]:
            returned = True
            print(answers[i-1]+1)
            break

if returned == False:
    print(answers[-1]+1)
            
