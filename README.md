Leetcode algo code template:

0,LEETCODE COMMON TEMPLATES & COMMON CODE PROBLEMS
https://cheatsheet.dennyzhang.com/cheatsheet-leetcode-a4 (python)

1,Binary search   
--while(left<right){mid-if-right=mid-else-left=mid+1}

2,DFS template                                                                                                            
--dfs{if-return-for-dfs}

3,Two Pointers code template                                           
--opposite two pointers--for{while(left<right){left++right--}}                                  
--same-direction two pointers--for-end-{while-begin+-if-end-begin}
        
4,sorting

5,⼆叉树分治 Binary Tree Divide & Conquer

6,BST Iterator                                                 
--while(!cur!||!stack)){while(!cur){add,cur=let}pop,cur=right}

7,BFS                                                       
--queue.offer-distance0-while(!queue){queue.poll-if-continue-for{if-continue-queue.offer-distance+}}
    
8, Dynamic Programming                                                                            
8.1 Matching type ----dp[n+1][m+1],dp[i][0],dp[0][j],for-i-for-j-dp[i][j]                                                                                                    
8.3 Solitaire type --dp[j]=max(dp[j],dp[i]+1)
                                                                                                            
9 堆 Heap 
                                                                       
10, 并查集 Union Find 
                                                                                                            
11, 字典树 Trie                                      
                                      
12, LRU 缓存
