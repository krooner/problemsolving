from collections import defaultdict, deque

def solution(n, edge):
    answer = 0
    
    graph = defaultdict(set)
    for a, b in edge:
        graph[a].add(b)
        graph[b].add(a)
        
    visited = [0] * n
    distance = [1e9] * n
    
    candidates = deque([])
    
#     def bfs(curr_node):
        
#         print(curr_node, graph[curr_node], distance, visited)
        
#         for next_node in graph[curr_node]:
#             if not visited[next_node-1]:
#                 distance[next_node-1] = \
#                     min(distance[next_node-1], distance[curr_node-1]+1)
#                 candidates.append(next_node)
#                 visited[next_node-1] = 1
        
#         for next_node in candidates:
#             bfs(next_node)
    
    curr_node = 1
    distance[curr_node-1] = 0
    visited[curr_node-1] = 1
    candidates.append(curr_node)
    
    while len(candidates) != 0:
        # print(candidates)
        for _ in range(len(candidates)):
            curr_node = candidates.popleft()
            for next_node in graph[curr_node]:
                if not visited[next_node-1]:
                    distance[next_node-1] = \
                        min(distance[next_node-1], distance[curr_node-1]+1)
                    candidates.append(next_node)
                    visited[next_node-1] = 1
            
    # bfs(curr_node)
    # print(distance)
    
    distance.sort(reverse=True)
    
    idx = 0
    max_value = distance[idx]
    while True:
        if distance[idx] == max_value:
            answer += 1
            idx += 1
        else:
            break
    
    return answer
