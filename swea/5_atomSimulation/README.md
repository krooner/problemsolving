# 원자 소멸 시뮬레이션
---

[URL](https://swexpertacademy.com/main/code/problem/problemDetail.do?contestProbId=AWXRFInKex8DFAUo)


### Idea
**최초 위치의 범위 밖에서 충돌할 수 없다**

**0.5초 단위에서 충돌할 수 있다**

---
1. 0.5초마다 원자들에 대해 업데이트를 하기 위해 `deque` 사용
    - `atoms = deque([])`
    - 좌표 `(x, y)`, 방향, 에너지를 저장
2. 충돌 여부를 확인하기 위해 좌표를 key로, 원자들의 (방향, 에너지)의 list를 value로 하는 `dictionary` 생성
    - `atoms.popleft()` 하면서 원자 정보 체크
    - 최초 위치 범위 `-1000~1000`를 초과한 경우에는 버린다.
    - 범위 안에 있는 경우에는 dictionary에 넣는다.
3. 해당 dictionary의 key (좌표)의 value list가 2개 이상일 경우 **충돌 발생**
4. value list에 1개 element만 있는 경우
    - 방향에 따라 좌표 업데이트 하여 `deque`에 다시 넣는다.
5. `while`문을 사용
    - 원자가 `atoms`에 하나도 없는 경우에 종료함
