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
