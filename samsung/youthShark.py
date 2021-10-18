from copy import deepcopy

dy = [-1, -1, 0, 1, 1, 1, 0, -1]
dx = [0, -1, -1, -1, 0, 1, 1, 1]

N = 4
grid = []
fishes = {}

ret = 0

for r in range(N):
    row = list(map(int, input().split()))
    fish_row = []
    for c in range(0, len(row), 2):
        num, direct = row[c:c+2]
        fish_row.append(num-1)
        fishes[num-1] = [direct-1, r, int(c/2)]
    grid.append(fish_row)

def solve(grid, fishes, shark_x, shark_y, sum_):

    grid_copy = deepcopy(grid)
    fishes_copy = deepcopy(fishes)
    keys = sorted(fishes_.keys())
    
    fish_number = grid[shark_y][shark_x]
    shark_dir = fishes_copy[fish_number][0]
    fishes_copy[fish_number] = [-1]*3
    grid_copy[shark_y][shark_x] = -1

    sum_ += (fish_number + 1)

    if ret < sum_:
        ret = sum_

    for k in keys:
        if fishes_copy[k][1] == -1:
            continue

        cd, cy, cx = fishes_copy[k]

        ny = cy + dy[cd]
        nx = cx + dx[cd]
        nd = cd

        while ny < 0 or ny >= 4 or nx <0 or nx >=4 or (ny == shark_y and nx == shark_x):
            nd = (nd+1) % 8
            ny = cy + dy[nd]
            nx = cx + dx[nd]

        if grid_copy[ny][nx] != -1:
            target = grid_copy[ny][nx]
            fishes_copy[target][1] = cy
            fishes_copy[target][2] = cx
            fishes_copy[k][1] = ny
            fishes_copy[k][2] = nx
            fishes_copy[k][0] = nd

            grid_copy[ny][nx] = k
            grid_copy[cy][cx] = target
        else:
            fishes_copy[k][1] = ny
            fishes_copy[k][2] = nx
            fishes_copy[k][0] = nd

            grid_copy[ny][nx] = k
            grid_copy[cy][cx] = -1

    for step in range(1, 4):
        ny = shark_y + dy[shark_dir] * step
        nx = shark_x + dx[shark_dir] * step

        if ny <0 or ny >=4 or nx < 0 or nx >= 4:
            break

        if grid_copy[ny][nx] != -1:
            solve(grid_copy, fishes_copy, ny, nx, sum_)



solve(grid, fishes, 0, 0, 0) 

print(ret)