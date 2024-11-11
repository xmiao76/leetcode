Leetcode algo code template:

0,LEETCODE COMMON TEMPLATES & COMMON CODE PROBLEMS
[https://cheatsheet.dennyzhang.com/cheatsheet-leetcode-a4](https://github.com/dennyzhang/cheatsheet.dennyzhang.com/tree/master/cheatsheet-leetcode-A4) (python)

1,Binary search   
--while(left<right){mid-if-right=mid-else-left=mid+1}

2,DFS template                                                                                                            
--dfs{if-return-for-dfs}

3,Two Pointers code template                                           
--opposite two pointers--for{while(left<right){left++right--}}                                  
--same-direction two pointers--for-end-{while-begin+-if-end-begin} or --for-begin-{while-end+-if-begin-end}
        
4,sorting
Arrays.sort(myArr, (a, b) -> a[0] - b[0]);-->Arrays.sort(myArr, (a, b) -> Integer.compare(b[0], a[0]));(avoid potential integer overflow)

5,⼆叉树分治 Binary Tree Divide & Conquer

6,BST Iterator                                                 
--while(!cur||!stack)){while(!cur){add,cur=let}pop,cur=right}

7,BFS                                                       
--queue.offer-distance0-while(!queue){queue.poll-if-continue-for{if-continue-queue.offer-distance+}}
    
8, Dynamic Programming                                                                            
8.1 Matching type ----dp[n+1][m+1],dp[i][0],dp[0][j],for-i-for-j-dp[i][j]                                                                                
8.2 Partition type  ----dp[len+1]-for-for-dp[i] or dp[m+1][n+1]-for-for-dp[i][j]                                                                                
8.3 Solitaire type --dp[j]=max(dp[j],dp[i]+1)                                                               
8.4 1/0 knapsack problem ----dp[n+1][sum+1]--for-for-dp[i][j] = dp[i][j] or || dp[i-1][j-nums[i-1]]                                             
8.5                                                                                                                                                     
8.6 Interval type --for-interval-for-i-for-j+ineterval-dp[i][j] = max/min/sum/or(dp[smaller interval inside i,j])
                                                                                                            
9 堆 Heap                                                         
--PriorityQueue-for/while-poll-if-offer                                        
                                                                       
10, 并查集 Union Find                                                                                               
--UnionFind(parent,size)-union(merge root)-find(return root)                                                       
                                                                                                            
11, 字典树 Trie                                                      
--Trie{Trie array[] = new Trie[n];boolean}
                                      
12, LRU 缓存                                                             
--new LinkedHashMap(capacity, 0.75f, true){boolean removeEldestEntry(Map.Entry eldest) {return size() > capacity}}

other
--reverse listNode: next= -> current.next= -> prev= -> current=
