# 2022 KAKAO BLIND RECRUITMENT > 양과 늑대

from collections import deque

def solution(info, edges):
    answer = [0]
    
    edges.sort(key=lambda x: x[0])
    
    # dict
    # key: Int (parent node)
    # value: List[Int] (list of child node)
    
    graph = {}
    for k, v in edges:
        if k not in graph:
            graph[k] = [v]
        else:
            graph[k].append(v)
    
    def dfs(candidates, n_sheep, n_wolf):
        
        for i in range(len(candidates)):
            n_sheep_copy = n_sheep
            n_wolf_copy = n_wolf
            cand = candidates[i]
            
            if info[cand] == 0:
                n_sheep_copy += 1
            else:
                n_wolf_copy += 1
            
            if n_sheep_copy <= n_wolf_copy:
                continue

            answer[0] = max(answer[0], n_sheep_copy)
            
            leftover = candidates[:i] + candidates[i+1:]
            
            if cand in graph:
                dfs(leftover + graph[cand], n_sheep_copy, n_wolf_copy)
            else:
                dfs(leftover, n_sheep_copy, n_wolf_copy)
        
    dfs([0], 0, 0)
        
    
    
    
    
    return answer[0]
