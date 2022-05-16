# 벽돌 깨기
---

[URL](https://swexpertacademy.com/main/code/problem/problemDetail.do?contestProbId=AWXRQm6qfL0DFAUo)

### 알아야할 것
- **2차원 배열 복사는 deepcopy** - `array[:]`로 뻘짓했다.

```
from copy import deepcopy

2d_array_copy_wrong = array[:]
2d_array_copy_right = deepcopy(array)
```

- 중간 과정에 조건을 집어넣어서 최대한 최적화를 시도하다가는 오히려 꼬여버리는 수가 있다
    - 벽돌이 존재하지 않는 column을 체크하는 과정을 넣는 등..
- **입력값의 범위를 보고 크지 않으면 일단 Raw하게 구현을 하자!**
---

N 개의 벽돌을 떨어트려 **최대한 많은 벽돌을 제거**... $\rightarrow$ DFS

- 돌을 다 쓸 때까지 실행되도록 `Recursion`을 활용한다.
- **최대한 많은 벽돌 제거 = 벽돌을 최소한으로 남긴다**
    - 함수의 argument로 벽돌을 제거시킨 후의 board의 잔여 벽돌 갯수를 갖고 간다: `count`
    - `answer = min(answer, count)`
- 벽돌이 부서지자마자 바로 해당 column을 update하지 않고, 연쇄 작용이 모두 끝날 때 한꺼번에 update한다
    1. 새로운 board를 initialize한다.
    2. 기존 board의 column에서 남은 벽돌을 확인한다.
    3. 새로운 board를 update한다.






