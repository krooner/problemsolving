N = int(input())

# diagonal
# anti-diagonal
# col

check_diagonal = [False] * (2 * N)
check_anti_diagonal = [False] * (2 * N)
check_col = [False] * N

board = [[False for _ in range(N)] for _ in range(N)]

def check(row, col):
    # row-wise
    # col-wise
    # diagonal-wise
    # anti-diagonal-wise

    if check_col[col]:
        return False

    if check_diagonal[row+col]:
        return False
    
    if check_anti_diagonal[row-col+N]:
        return False
    
    return True

def cal(row):
    if row == N:
        return 1
    
    cnt = 0
    for col in range(N):
        if check(row, col):
            check_diagonal[row+col] = True
            check_anti_diagonal[row-col+N] = True
            check_col[col] = True

            board[row][col] = True

            cnt += cal(row+1)

            board[row][col] = False

            check_diagonal[row+col] = False
            check_anti_diagonal[row-col+N] = False
            check_col[col] = False

    return cnt

print(cal(0))

        


