from itertools import combinations
testcases = []

while True:
    tc = input()
    if tc == "0":
        break
    S = list(map(int, tc.split()))[1:]
    testcases.append(S)
    
for i, element in enumerate(testcases):
    for item in combinations(element, 6):
        item = sorted(item)
        print(" ".join(list(map(str, item))))
    if i != len(testcases)-1:
        print()