# 보물상자 비밀번호
---

[URL](https://swexpertacademy.com/main/code/problem/problemDetail.do?contestProbId=AWXRUN9KfZ8DFAUo)

비밀번호를 만드는 과정 - **3중 for-loop**
1. 회전 횟수 `q = N//4`
    - 한 칸씩 회전시키면서, 각 변의 element 구성이 처음과 동일하기 전까지의 경우의 수
    - 변의 element 수와 동일하다 
2. 4개의 변에서의 시작 index `range(0, N, q)`
    - 변에 있는 element 수 `q` 만큼 interval을 두어 시작 index를 정한다
    - 해당 index에서 `q`개의 element로 숫자를 만든다.
3. 각 변의 element
    - hexa-symbol을 변환하는 table을 미리 만들어서 숫자로 변환한다
    - 16진법을 10진법으로 변환한다.

비밀번호를 만들었으면,
1. 발생할 수 있는 모든 수를 담는다. 
    - 중복 수를 없애기 위해 `set()`을 사용했다.
2. `set`을 `list`로 변환한다
3. `list`를 sorting한다
4. `K`번째 원소를 고른다