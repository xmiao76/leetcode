Leetcode algo code template:

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

Notice:
