# 핀볼 게임
---

[URL](https://swexpertacademy.com/main/code/problem/problemDetail.do?contestProbId=AWXRF8s6ezEDFAUo)


### Variable name은 항상 체크하자
`answer`를 출력해야 하는데 `score`를 출력하고 있는..

---

1. 블록의 모양에 따라 달라지는 방향
    - 규칙을 신박하게 풀어버릴 아이디어를 단숨에 생각해내면 좋겠지만, 바로 안 떠오르는 경우에는 변환 테이블을 만들자.
    - `(block_number, before_dir)`를 key로 하고, `after_dir` (converted dir)를 value로 하는 `dictionary`
2. 웜홀이 쌍으로 발생한다
    - `wormhole_number`를 key로 하고, `(x, y)`의 리스트를 value로 하는 `dictionary`를 활용했다.