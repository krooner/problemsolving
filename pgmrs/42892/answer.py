import sys
sys.setrecursionlimit(10**3)

def solution(nodeinfo):
        
    for i in range(len(nodeinfo)):
        nodeinfo[i].append(i)

    preorder = []
    postorder = []
    
    def find_root(info, is_pre=True):
        if len(info) == 0:
            return

        info.sort(key=lambda x: x[1], reverse=True)
        
        root_node = info[0]
        root_node_x, root_node_y, root_node_idx = root_node
        
        left_subtrees = []
        right_subtrees = []
        
        for i in range(1, len(info)):
            item = info[i]
            if item[0] < root_node_x: 
                left_subtrees.append(item)
            else:
                right_subtrees.append(item)
            
        if is_pre:
            preorder.append(root_node_idx+1)
            find_root(left_subtrees, is_pre)
            find_root(right_subtrees, is_pre)
        else: # post
            find_root(left_subtrees, is_pre)
            find_root(right_subtrees, is_pre)
            postorder.append(root_node_idx+1)

        return 
    
    find_root(nodeinfo, is_pre=True)
    find_root(nodeinfo, is_pre=False)
    
    return [preorder, postorder]
