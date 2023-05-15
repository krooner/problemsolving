from itertools import combinations

N = int(input())

table = []
for _ in range(N):
    table.append(list(map(int, input().split())))


min_diff = 1e10

for item in combinations(list(range(N)), N//2):
    
    team_a = list(item)
    stat_a = 0
    for a_pair in combinations(team_a, 2):
        i, j = tuple(a_pair)
        stat_a += table[i][j] + table[j][i]

    team_b = [i for i in range(N) if i not in team_a]
    stat_b = 0
    for b_pair in combinations(team_b, 2):
        i, j = tuple(b_pair)
        stat_b += table[i][j] + table[j][i]

    min_diff = min(min_diff, abs(stat_a-stat_b))

print(min_diff)    

