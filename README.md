Leetcode algo code template:

0,LEETCODE COMMON TEMPLATES & COMMON CODE PROBLEMS
https://cheatsheet.dennyzhang.com/cheatsheet-leetcode-a4 (python)

1,Binary search 
1.a:Python template code(java code is similar)
class Solution:
    def firstBadVersion(self, n) -> int:
        left, right = 1, n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left
        
Detail explain: https://leetcode.com/problems/first-bad-version/discuss/769685/Python-Clear-explanation-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems
1.b: use Java util method: 
Collections.binarySearch(List slist, T key)

Notice:if a number is not present in the list when we use Collections.binarySearch(List slist, T key), it returns -potentialIndex-1, where the potentialIndex is the location at which the number would have been inserted, so to get the exact index author is doing -index - 1
So, in this case, index = -(-potentialIndex - 1) - 1 = potentialIndex

2,DFS template
https://leetcode.com/problems/number-of-islands/discuss/1089975/java-dfs-1ms-general-matrix-traversing-dfs-template?msclkid=61b9991cce9011ec8ac5370ae07b7016

3,Two Points code template
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
4.1, 7 Sorting Algorithms (quick sort, top-down/bottom-up merge sort, heap sort, etc.)
https://leetcode.com/problems/sort-an-array/discuss/492042/7-Sorting-Algorithms-(quick-sort-top-downbottom-up-merge-sort-heap-sort-etc.)

5,BST Iterator
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
    
    
6,宽度优先搜索 BFS
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
