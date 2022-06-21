Leetcode algo code template:

0,LEETCODE COMMON TEMPLATES & COMMON CODE PROBLEMS
https://cheatsheet.dennyzhang.com/cheatsheet-leetcode-a4 (python)

1,Binary search 

2,DFS template
使⽤条件 
• 找满⾜某个条件的所有⽅案 (99%) 
• ⼆叉树 Binary Tree 的问题 (90%) 
• 组合问题(95%) ◦ 问题模型：求出所有满⾜条件的“组合” ◦ 判断条件：组合中的元素是顺序⽆关的 
• 排列问题 (95%) ◦ 问题模型：求出所有满⾜条件的“排列” ◦ 判断条件：组合中的元素是顺序“相关”的。 
不要⽤ DFS 的场景 
1. 连通块问题（⼀定要⽤ BFS，否则 StackOverflow） 
2. 拓扑排序（⼀定要⽤ BFS，否则 StackOverflow） 
3. ⼀切 BFS 可以解决的问题 
复杂度 
• 时间复杂度：O(⽅案个数 * 构造每个⽅案的时间) 
◦ 树的遍历 ： O(n) 
◦ 排列问题 ： O(n! * n) 
◦ 组合问题 ： O(2^n * n)
https://leetcode.com/problems/number-of-islands/discuss/1089975/java-dfs-1ms-general-matrix-traversing-dfs-template?msclkid=61b9991cce9011ec8ac5370ae07b7016

https://leetcode.com/problems/binary-tree-inorder-traversal/
https://leetcode.com/problems/factor-combinations/
public ReturnType dfs(参数列表) { 
    if (递归出⼝) { 
        记录答案; 
        return; 
    } 
    for (所有的拆解可能性) { 
        修改所有的参数 
        dfs(参数列表); 
        还原所有被修改过的参数 
    } 
    return something 如果需要的话，很多时候不需要 return 值除了分治的写法 
}

3,Two Points code template
滑动窗⼝ (90%) 
时间复杂度要求 O(n) (80%是双指针) 
要求原地操作，只可以使⽤交换，不能使⽤额外空间 (80%) 
有⼦数组 subarray /⼦字符串 substring 的关键词 (50%) 
有回⽂ Palindrome 关键词(50%) 
时间复杂度：O(n) 时间复杂度与最内层循环主体的执⾏次数有关 与有多少重循环⽆关 
空间复杂度：O(1) 只需要分配两个指针的额外内存
3.1 https://leetcode.com/problems/binary-subarrays-with-sum/discuss/1353347/Java-3-O(N)-Time-Solutions

3.2 https://leetcode.com/problems/minimum-window-substring/discuss/26808/Here-is-a-10-line-template-that-can-solve-most-'substring'-problems
Java template code:
class Solution {
    public String minWindow(String s, String t) {
        int[] arr = new int[128];        
        int counter=t.length(), begin=0, end=0,  head=0;//two pointers, one point to tail and one  head
        int d=Integer.MAX_VALUE;//the length of substring
        for(char c: t.toCharArray()) arr[c-'A']++;/* initialize the hash map here */
        while(end<s.length()){            
            if(arr[s.charAt(end++)-'A']-->0) counter--;  /* modify counter here */
            while(counter==0){ /* counter condition */
                /* update d here if finding minimum*/
                if(end-begin<d) {
                    head=begin;
                    d=end-begin;
                } 
                //increase begin to make it invalid/valid again
                if(arr[s.charAt(begin++)-'A']++==0) counter++;   /*modify counter here*/
            }
            /* update d here if finding maximum*/
        }
        return d==Integer.MAX_VALUE? "":s.substring(head, head+d);    
    }
}
https://leetcode.com/problems/binary-subarrays-with-sum/submissions/ , code follow similar tempalte:
    public int numSubarraysWithSum(int[] nums, int goal) {
        int ans=0;
        if(goal==0){
            ...
        }
        int begin=0,end=0,sum=0;
        while(end<nums.length){
            sum+=nums[end];/* modify counter here */
            end++;
            if(sum==goal){/* counter condition */
                int count=1;
                while(begin<end && nums[begin]==0) {count++;begin++;}
                while(end<nums.length && nums[end]==0){ ans+=count;end++;}
                ans+=count;
                //increase begin to make it invalid/valid again
                if(begin<nums.length && nums[begin]==1){sum--; begin++;}
            }
        }
        return ans;
        
    }
        
4,sorting
时间复杂度： 
快速排序(期望复杂度) ： O(nlogn) 
归并排序(最坏复杂度) ： O(nlogn) 
空间复杂度： 
快速排序 ： O(1) 
归并排序 ： O(n)
4.1, 7 Sorting Algorithms (quick sort, top-down/bottom-up merge sort, heap sort, etc.)
https://leetcode.com/problems/sort-an-array/discuss/492042/7-Sorting-Algorithms-(quick-sort-top-downbottom-up-merge-sort-heap-sort-etc.)

5,⼆叉树分治 Binary Tree Divide & Conquer
⼆叉树相关的问题 (99%) 
可以⼀分为⼆去分别处理之后再合并结果 (100%) 
数组相关的问题 (10%) 
时间复杂度 O(n) 
空间复杂度 O(n) (含递归调⽤的栈空间最⼤耗费)
5.1,
https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/
https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/discuss/1494821/Simplest-Java-solution-with-explanation.-Inorder-traversal-in-place-no-dummy-node-needed
5.2,
https://leetcode.com/problems/binary-tree-maximum-path-sum/
5.3，
https://leetcode.com/problems/validate-binary-search-tree/


6,BST Iterator
使⽤条件 
• ⽤⾮递归的⽅式（Non-recursion / Iteration）实现⼆叉树的中序遍历 
• 常⽤于 BST 但不仅仅可以⽤于 BST 
复杂度 
时间复杂度 O(n) 
空间复杂度 O(n)
https://leetcode.com/problems/binary-tree-inorder-traversal/

public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<Integer>();

        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode cur = root;

        while(cur!=null || !stack.empty()){
            while(cur!=null){
                stack.add(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            list.add(cur.val);
            cur = cur.right;
        }

        return list;
    }
    
https://leetcode.com/problems/kth-smallest-element-in-a-bst/    
    public int kthSmallest(TreeNode root, int k) {
      Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode cur = root;

        while(cur!=null || !stack.empty()){
            while(cur!=null){
                stack.add(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            k--;
            if(k==0) break;
            cur = cur.right;
        }
        
        if(cur==null) return -1;
        else return cur.val;
    }
    
    
7,宽度优先搜索 BFS
    拓扑排序(100%) 
    出现连通块的关键词(100%) 
    分层遍历(100%) 
    简单图最短路径(100%) 
    给定⼀个变换规则，从初始状态变到终⽌状态最少⼏步(100%) 
    时间复杂度：O(n + m) n 是点数, m 是边数 
    空间复杂度：O(n)
7.1,
https://leetcode.com/problems/01-matrix/
https://leetcode.com/problems/01-matrix/discuss/1499453/JAVA-%2B-BFS-SOLUTION-WITH-BETTER-INTUITION
    
ReturnType bfs(Node startNode) { 
    // BFS 必须要⽤队列 queue，别⽤栈 stack！ 
    Queue<Node> queue = new ArrayDeque<>(); 
    // hashmap 有两个作⽤，⼀个是记录⼀个点是否被丢进过队列了，避免重复访问 
    // 另外⼀个是记录 startNode 到其他所有节点的最短距离 
    // 如果只求连通性的话，可以换成 HashSet 就⾏ 
    // node 做 key 的时候⽐较的是内存地址 
    Map<Node, Integer> distance = new HashMap<>(); 
    // 把起点放进队列和哈希表⾥，如果有多个起点，都放进去 
    queue.offer(startNode); 
    distance.put(startNode, 0); // or 1 if necessary 
    
    // while 队列不空，不停的从队列⾥拿出⼀个点，拓展邻居节点放到队列中 
    while (!queue.isEmpty()) { 
        Node node = queue.poll(); 
        // 如果有明确的终点可以在这⾥加终点的判断 
        if (node 是终点) { 
        break or return something; 
        } 
        for (Node neighbor : node.getNeighbors()) { 
            if (distance.containsKey(neighbor)) { 
                continue; 
            } 
            queue.offer(neighbor); 
            distance.put(neighbor, distance.get(node) + 1); 
        } 
    } 
    // 如果需要返回所有点离起点的距离，就 return hashmap 
    return distance; 
    // 如果需要返回所有连通的节点, 就 return HashMap ⾥的所有点 
    return distance.keySet(); 
    // 如果需要返回离终点的最短距离 
    return distance.get(endNode); 
}

7.2,
https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph
https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/discuss/77651/Standard-BFS-and-DFS-Solution-JAVA
    
8, 动态规划 Dynamic Programming 
    使⽤条件 • ◦◦
        使⽤场景： 
            求⽅案总数(90%) 
            求最值(80%)
            ◦ 求可⾏性(80%)
    • 不适⽤的场景： 
        ◦ 找所有具体的⽅案（准确率99%） 
        ◦ 输⼊数据⽆序(除了背包问题外，准确率60%~70%) 
        ◦ 暴⼒算法已经是多项式时间复杂度（准确率80%） 
    • 动态规划四要素(对⽐递归的四要素)： 
        ◦ 状态 (State) -- 递归的定义 
        ◦ ⽅程 (Function) -- 递归的拆解 
        ◦ 初始化 (Initialization) -- 递归的出⼝ 
        ◦ 答案 (Answer) -- 递归的调⽤ 
    • ⼏种常⻅的动态规划： 
        • 背包型 
            ◦ 给出 n 个物品及其⼤⼩,问是否能挑选出⼀些物品装满⼤⼩为m的背包 
            ◦ 题⽬中通常有“和”与“差”的概念，数值会被放到状态中 
            ◦ 通常是⼆维的状态数组，前 i 个组成和为 j 状态数组的⼤⼩需要开 (n + 1) * (m + 1) 
            ◦ ⼏种背包类型： 
                ▪ 01背包 
                    • 状态 state 
                        dp[i][j] 表⽰前 i 个数⾥挑若⼲个数是否能组成和为 j 
                    ⽅程 function 
                        dp[i][j] = dp[i - 1][j] or dp[i - 1][j - A[i - 1]] 如果 j >= A[i - 1] dp[i][j] = dp[i - 1][j] 
                        如果 j < A[i - 1] 第 i 个数的下标是 i - 1，所以⽤的是 A[i - 1] ⽽不是 A[i] 
                    初始化 initialization
                        dp[0][0] = true dp[0][1...m] = false 
                    答案 answer 
                        使得 dp[n][v], 0 s <= v <= m 为 true 的最⼤ v 
                ▪ 多重背包 
                    • 状态 state 
                        dp[i][j] 表⽰前i个物品挑出⼀些放到 j 的背包⾥的最⼤价值和 
                    ⽅程 function 
                        dp[i][j] = max(dp[i - 1][j - count * A[i - 1]] + count * V[i - 1]) 其中 0 <= count <= j / A[i - 1] 
                    初始化 initialization
                        dp[0][0..m] = 0 
                    答案 
                        answer dp[n][m] 
                • 区间型 
                    • 题⽬中有 subarray / substring 的信息 
                        ◦ ⼤区间依赖⼩区间 
                        ◦ ⽤ dp[i][j] 表⽰数组/字符串中 i, j 这⼀段区间的最优值/可⾏性/⽅案总数 
                        ◦ 状态 state 
                            dp[i][j] 表⽰数组/字符串中 i,j 这⼀段区间的最优值/可⾏性/⽅案总数 
                         ⽅程 function 
                            dp[i][j] = max/min/sum/or(dp[i,j 之内更⼩的若⼲区间]) 
                • 匹配型 
                    ◦ 通常给出两个字符串 
                    ◦ 两个字符串的匹配值依赖于两个字符串前缀的匹配值 
                    ◦ 字符串⻓度为 n,m 则需要开 (n + 1) x (m + 1) 的状态数组 
                    ◦ 要初始化 dp[i][0] 与 dp[0][i] 
                    ◦ 通常都可以⽤滚动数组进⾏空间优化 
                    ◦ 状态 state 
                        dp[i][j] 表⽰第⼀个字符串的前 i 个字符与第⼆个字符串的前 j 个字符怎么样怎么样 (max/min/sum/or) 
                • 划分型 
                    ◦ 是前缀型动态规划的⼀种, 有前缀的思想 
                    ◦ 如果指定了要划分为⼏个部分： 
                        ▪ dp[i][j] 表⽰前i个数/字符划分为j个 部分的最优值/⽅案数/可⾏性 
                    ◦ 如果没有指定划分为⼏个部分: 
                        ▪ dp[i] 表⽰前i个数/字符划分为若⼲个 部分的最优值/⽅案数/可⾏性 
                    ◦ 状态 state 
                        指定了要划分为⼏个部分: dp[i][j] 表⽰前i个数/字符划分为j个部分的最优值/⽅案数/可⾏ 性
                        没有指定划分为⼏个部分: dp[i] 表⽰前i个数/字符划分为若⼲个部分的最优值/⽅案数/可⾏ 性 
                • 接⻰型 
                    ◦ 通常会给⼀个接⻰规则，问你最⻓的⻰有多⻓ 
                    ◦ 状态表⽰通常为: dp[i] 表⽰以坐标为 i 的元素结尾的最⻓⻰的⻓度 
                    ◦ ⽅程通常是: dp[i] = max{dp[j] + 1}, j 的后⾯可以接上 i 
                    ◦ LIS 的⼆分做法选择性的掌握，但并不是所有的接⻰型DP都可以⽤⼆分来优化 
                    ◦ 状态 state
                        状态表⽰通常为: dp[i] 表⽰以坐标为 i 的元素结尾的最⻓⻰的⻓度 
                        ⽅程 function dp[i] = max{dp[j] + 1}, j 的后⾯可以接上 i
    复杂度 
        • 时间复杂度: 
            ◦ O(状态总数 * 每个状态的处理耗费) 
            ◦ 等于O(状态总数 * 决策数) 
        • 空间复杂度： 
            ◦ O(状态总数) (不使⽤滚动数组优化) 
            ◦ O(状态总数 / n)(使⽤滚动数组优化, n是被滚动掉的那⼀个维度)
8.1 匹配型
https://leetcode.com/problems/wildcard-matching/submissions/
8.2 划分型
https://leetcode.com/problems/word-break/
                                                                                                            
9 堆 Heap 
使⽤条件 
1. 找最⼤值或者最⼩值(60%) 
2. 找第 k ⼤(pop k 次 复杂度O(nlogk))(50%) 
3. 要求 logn 时间对数据进⾏操作(40%)
堆不能解决的问题 
1. 查询⽐某个数⼤的最⼩值/最接近的值（平衡排序⼆叉树 Balanced BST 才可以解决） 
2. 找某段区间的最⼤值最⼩值（线段树 SegmentTree 可以解决） 
3. O(n)找第k⼤ (使⽤快排中的partition操作)
                                                                                                            
https://leetcode.com/problems/find-k-pairs-with-smallest-sums/submissions/
https://leetcode.com/problems/meeting-rooms-ii/submissions/
https://leetcode.com/problems/minimum-cost-to-hire-k-workers/submissions/
                                                                       
10, 并查集 Union Find 
使⽤条件 
    • 需要查询图的连通状况的问题 
    • 需要⽀持快速合并两个集合的问题 复杂度 
    • 时间复杂度 union O(1), find O(1) 
    • 空间复杂度 O(n)
Sample: 
    721. Accounts Merge: https://leetcode.com/problems/accounts-merge/
        class UnionFind {
            int[] parent;
            int[] weight;

            public UnionFind(int num) {
                parent = new int[num];
                weight = new int[num];

                for(int i =  0; i < num; i++) {
                    parent[i] = i;
                    weight[i] = 1;
                }
            }

            public void union(int a, int  b) {
                int rootA = root(a);
                int rootB = root(b);

                if (rootA == rootB) {
                    return;
                }

                if (weight[rootA] > weight[rootB]) {
                    parent[rootB] = rootA;
                    weight[rootA] += weight[rootB];
                } else {
                    parent[rootA] = rootB;
                    weight[rootB] += weight[rootA];
                }
            }

            public int root(int a) {
                if (parent[a] == a) {
                    return a;
                }

                //return root(parent[a]);
                parent[a] = root(parent[a]);
                return parent[a];            
            }
        }                                                                                                        
    803. Bricks Falling When Hit , https://leetcode.com/problems/bricks-falling-when-hit/ , https://www.youtube.com/watch?v=qItIA8m3OnU
        private class UnionFind {

            /**
             * 当前结点的父亲结点
             */
            private int[] parent;
            /**
             * 以当前结点为根结点的子树的结点总数
             */
            private int[] size;

            public UnionFind(int n) {
                parent = new int[n];
                size = new int[n];
                for (int i = 0; i < n; i++) {
                    parent[i] = i;
                    size[i] = 1;
                }
            }

            /**
             * 路径压缩，只要求每个不相交集合的「根结点」的子树包含的结点总数数值正确即可，因此在路径压缩的过程中不用维护数组 size
             *
             * @param x
             * @return
             */
            public int find(int x) {
                if (x != parent[x]) {
                    parent[x] = find(parent[x]);
                }
                return parent[x];
            }

            public void union(int x, int y) {
                int rootX = find(x);
                int rootY = find(y);

                if (rootX == rootY) {
                    return;
                }
                parent[rootX] = rootY;
                // 在合并的时候维护数组 size
                size[rootY] += size[rootX];
            }

            /**
             * @param x
             * @return x 在并查集的根结点的子树包含的结点总数
             */
            public int getSize(int x) {
                int root = find(x);
                return size[root];
            }
        }
                                                                                                            
11, 字典树 Trie 
    使⽤条件
        需要查询包含某个前缀的单词/字符串是否存在 
        字符矩阵中找单词的问题
    复杂度
        时间复杂度 O(L) 增删查改 
        空间复杂度 O(N * L) N 是单词数，L是单词⻓度
                                      
                                      
