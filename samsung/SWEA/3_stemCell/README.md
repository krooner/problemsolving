# 줄기세포배양
---

[URL](https://swexpertacademy.com/main/code/problem/problemDetail.do?contestProbId=AWXRJ8EKe48DFAUo)

1. 현재 생존 중인 세포의 상태를 저장하고 업데이트 해야한다.
    - `queue` 기능을 사용하고자 `deque`를 활용한다
    - 세포가 활성화되는 경우
    - 세포가 죽어버린 경우
2. 죽은 세포가 있는 공간에는 접근 불가하게 만들어야 한다.
    - `board`을 선언하여 cell마다의 상태를 표시한다.
    - `-1 (죽음), 0 (비어있음), 1 (살아있음)`
3. 빈 공간에 여러 세포가 증식할 경우를 고려하여 생명력 정보를 저장해야 한다.
    - 좌표를 key, 생명력 list를 value로 하는 `dictionary`를 선언한다.
    - 생명력이 가장 높은 값을 골라서 증식시킨다.

    