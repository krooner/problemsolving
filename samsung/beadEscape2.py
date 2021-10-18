
n, m = map(int, input().split())

board = [list(input()) for _ in range(n)]

hole = blue = red = None

for ver in range(n):
    for hor in range(m):
        if board[ver][hor]=="B":
            blue = (ver, hor)
        if board[ver][hor]=="R":
            red = (ver, hor)
        if board[ver][hor]=="O":
            hole = (ver, hor)