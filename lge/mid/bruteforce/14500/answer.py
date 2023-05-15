N, M = list(map(int, input().split()))

board = []
for _ in range(N):
    board.append(
        list(map(int, input().split()))
    )

# 1. Set center point
# 2. Expand it 