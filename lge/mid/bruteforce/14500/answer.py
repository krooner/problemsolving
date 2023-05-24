N, M = list(map(int, input().split()))

board = []
for _ in range(N):
    board.append(
        list(map(int, input().split()))
    )

# rotation
# symmetry

row = len(board)
col = len(board[0])

def check_validity(r, c, dr, dc):
    global row, col
    answer = True

    for i in range(4):
        rr = r + dr[i]
        cc = c + dc[i]
        if rr < 0 or rr >= row or cc < 0 or cc >= col:
            answer = False
            break
    
    return answer

def check_poly(drs, dcs, r, c, value):
    for i in range(len(drs)):
        dr, dc = drs[i], dcs[i]

        if check_validity(r, c, dr, dc):
            summation = sum([board[r+dr[j]][c+dc[j]] for j in range(4)])
            value = max(value, summation)

    return value

maxval = -1

for r in range(row):
    for c in range(col):

        # 1. stick
        drs = [
            [0, 0, 0, 0],
            [0, 1, 2, 3]
        ]
        dcs = [
            [0, 1, 2, 3],
            [0, 0, 0, 0]
        ]

        maxval = check_poly(drs, dcs, r, c, maxval)

        # 2. rectangle
        drs = [
            [0, 0, 1, 1]
        ]
        dcs = [
            [0, 1, 0, 1]
        ]
        
        maxval = check_poly(drs, dcs, r, c, maxval)

        # 3. gun
        drs = [
            [0, 1, 2, 2],
            [0, 0, 0, 1],
            [0, -1, -2, -2],
            [0, 0, 0, -1],

            [0, 1, 2, 2],
            [0, 0, 0, 1],
            [0, -1, -2, -2],
            [0, 0, 0, -1]
        ]
        dcs = [
            [0, 0, 0, 1],
            [0, -1, -2, -2],
            [0, 0, 0, -1],
            [0, 1, 2, 2],

            [0, 0, 0, -1],
            [0, 1, 2, 2],
            [0, 0, 0, 1],
            [0, -1, -2, -2]
        ]

        maxval = check_poly(drs, dcs, r, c, maxval)

        # 4. z
        drs = [
            [0, 1, 1, 2],
            [0, 0, -1, -1],
            [0, 1, 1, 2],
            [0, 0, 1, 1]
        ]
        dcs = [
            [0, 0, 1, 1],
            [0, 1, 1, 2],
            [0, 0, -1, -1],
            [0, 1, 1, 2]
        ]

        maxval = check_poly(drs, dcs, r, c, maxval)

        # 5. f-word
        drs = [
            [0, 0, 0, -1],
            [0, 1, 2, 1],
            [0, 0, 0, 1],
            [0, 1, 2, 1]
        ]
        dcs = [
            [0, 1, 2, 1],
            [0, 0, 0, 1],
            [0, 1, 2, 1],
            [0, 0, 0, -1]
        ]

        maxval = check_poly(drs, dcs, r, c, maxval)

print(maxval)


        
        