from itertools import permutations

N = int(input())
unique_alpha_set = set()
l = []
for _ in range(N):
    ele = input()
    unique_alpha_set |= set(list(ele))
    l.append(ele)

unique_length = len(unique_alpha_set)

alpha_list = list(unique_alpha_set)

def convert_strlist(origin, table):
    answer = []
    for item in origin:
        conversion = "".join([str(table[e]) for e in list(item)])
        answer.append(int(conversion))
    return sum(answer)

maxval = -1

for item in permutations([i for i in range(9, 9-unique_length, -1)], unique_length):
    table = {alpha_list[j]: digit for j, digit in enumerate(item)}

    maxval = max(maxval, convert_strlist(l, table))

print(maxval)


    