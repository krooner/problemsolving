dr = [-1, 0, 1, 0]; dc = [0, 1, 0, -1]

N, Q = list(map(int, input().split()))
length = pow(2, N)
board = [list(map(int, input().split())) for _ in range(length)]
visited = [[False for _ in range(length)] for _ in range(length)]

L = list(map(int, input().split()))

def find_cluster(pos, acc):
    r, c = pos
    for i in range(4):
        rr, cc = r + dr[i], c + dc[i]
        if 0<=rr<length and 0<=cc<length and visited[rr][cc] == False and board[rr][cc] > 0:
            visited[rr][cc] = True
            cnt = find_cluster((rr, cc), acc+1)
            acc = max(cnt, acc)
    return acc

for l in L:
    if l > 0:
        interval = pow(2, l)
        backup = [[0 for _ in range(interval)] for _ in range(interval)]
        for i in range(0, length, interval):
            for j in range(0, length, interval):
                for a in range(interval):
                    for b in range(interval):
                        backup[a][b] = board[i+interval-b-1][j+a]

                for a in range(interval):
                    for b in range(interval):
                        board[i+a][j+b] = backup[a][b]

    # 3. reduce ice by condition
    backup = [[0 for _ in range(length)] for _ in range(length)]

    for i in range(length):
        for j in range(length):
            if board[i][j] == 0:
                continue
            cnt = 0
            for k in range(4):
                rr, cc = i + dr[k], j + dc[k]
                if 0<=rr<length and 0<=cc<length and board[rr][cc] > 0:
                    cnt += 1
            if cnt < 3:
                backup[i][j] = board[i][j] - 1
            else:
                backup[i][j] = board[i][j]
    board = backup

answer = 0
max_cells = 0

for i in range(length):
    for j in range(length):
        if board[i][j] > 0:
            answer += board[i][j]
            if visited[i][j] == False:
                visited[i][j] = True
                cells = find_cluster((i, j), 1)
                max_cells = max(max_cells, cells)

print(answer)
print(max_cells)

