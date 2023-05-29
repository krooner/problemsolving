N, M = list(map(int, input().split()))

board = []
for _ in range(N):
    board.append(list(input()))

r1 = c1 = r2 = c2 = None

for r in range(N):
    for c in range(M):
        if board[r][c] == "o":
            if r1 == None:
                r1 = r
                c1 = c
            else:
                r2 = r
                c2 = c

dr = [-1, 0, 1, 0]
dc = [0, 1, 0, -1]

answer = 11

def move(cnt, r1, c1, r2, c2):
    global answer
    if cnt == 11:
        return

    for i in range(4):
    
        fall1 = fall2 = True

        nr1 = r1 + dr[i]
        nc1 = c1 + dc[i]
        nr2 = r2 + dr[i]
        nc2 = c2 + dc[i]

        # if nr1<0 or nr1==N or nc1<0 or nc1==M:
        if 0<=nr1<N and 0<=nc1<M:
            fall1 = False
            if board[nr1][nc1] == "#":
                nr1 = r1
                nc1 = c1

        # if nr2<0 or nr2==N or nc2<0 or nc2==M:
        if 0<=nr2<N and 0<=nc2<M:
            fall2 = False
            if board[nr2][nc2] == "#":
                nr2 = r2
                nc2 = c2

        if fall1 and fall2: continue
        elif fall1+fall2 == 1:
            answer = min(answer, cnt+1)
        else:
            move(cnt+1, nr1, nc1, nr2, nc2)

move(0, r1, c1, r2, c2)

print(answer if answer < 11 else -1)