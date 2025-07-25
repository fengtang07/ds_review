# Arrays, Stacks, and Queues

# What is an Array?

An **array** is a fundamental data structure that stores elements of the same type in a **contiguous block of memory**. This sequential storage is what makes arrays highly efficient for many data science tasks, as it's cache-friendly and allows for direct memory access. In Python, the `list` type is a dynamic array, meaning it can automatically resize itself.

# Key Properties

- **Fixed size** (in most languages)
- **Zero-indexed** (first element at index 0)
- **Contiguous memory** allocation
- **Random access** in O(1) time
- **Cache-friendly** due to spatial locality

# Key Properties & Complexities

| Operation | Average Case | Worst Case | Space Complexity | Data Science Context & Notes |
| --- | --- | --- | --- | --- |
| **Access by Index** | O(1) | O(1) | O(n) | Essential for feature matrix lookups ( |
| **Search (Unsorted)** | O(n) | O(n) | O(n) | A linear scan. Inefficient for large datasets. |
| **Search (Sorted)** | O(log n) | O(log n) | O(n) | Possible with binary search.  |
| **Insertion** | O(n) | O(n) | O(n) | Requires shifting elements, costly for large arrays. |
| **Deletion** | O(n) | O(n) | O(n) | Also requires shifting, a performance bottleneck. |

# Memory Layout

```
Array: [10, 20, 30, 40, 50]
Memory: [1000][1004][1008][1012][1016]
Index:    0     1     2     3     4

Element at index i = base_address + (i * element_size)

```

---

# Essential Patterns & Techniques

## 1. Two Pointers

This is a highly common pattern in array-based interview questions, especially when the array is sorted. By using two pointers, you can often avoid nested loops and reduce time complexity from O(n2) to O(n).

- **Opposite-Direction Pointers**: Start pointers at each end of the array and move them inward.
    - **Use Case**: Finding pairs that sum to a target in a sorted array. You can efficiently discard half of the remaining search space with each comparison.
- **Same-Direction Pointers (Slow & Fast)**: Both pointers start at the beginning. The fast pointer iterates through the array, while the slow pointer only moves when a certain condition is met.
    - **Use Case**: Removing duplicates in-place from a sorted array. The slow pointer marks the end of the unique-element section.

### Example 1: Two Sum II - Input Array Is Sorted

**LEETCODE LINK** [https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

**Solution Strategy**

1. Initialize two pointers at opposite ends of the sorted array:
    - `left`: Points to the smallest element (index 0)
    - `right`: Points to the largest element (last index)
2. Use a while loop to move pointers toward each other until they meet
3. At each step, calculate the sum of elements at current pointer positions
4. Compare the current sum with target:
    - If `current_sum == target`: Found the answer, return 1-indexed positions
    - If `current_sum < target`: Sum too small, move `left` pointer right to get larger numbers
    - If `current_sum > target`: Sum too large, move `right` pointer left to get smaller numbers
5. Continue until pointers meet or solution is found

**Time/Space Complexity**

- Time: O(n) - each element is visited at most once as pointers move toward center
- Space: O(1) - only using two pointer variables, no additional data structures

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        """
        Two Sum II - Input array is sorted
        
        Given a sorted array, find two numbers that add up to target.
        Return their indices (1-indexed).
        
        Example: numbers = [2,7,11,15], target = 9
        Output: [1,2] because numbers[0] + numbers[1] = 2 + 7 = 9
        
        Time Complexity: O(n) - we visit each element at most once
        Space Complexity: O(1) - only using two pointer variables
        """
        
        # STEP 1: Initialize two pointers at opposite ends
        # left points to smallest element (index 0)
        # right points to largest element (last index)
        left = 0
        right = len(numbers) - 1
        
        # Alternative shorthand (same as above):
        # left, right = 0, len(numbers) - 1
        
        # STEP 2: Move pointers toward each other until they meet
        while left < right:
            
            # STEP 3: Calculate sum of elements at current pointer positions
            current_sum = numbers[left] + numbers[right]
            
            # STEP 4: Check if we found the target sum
            if current_sum == target:
                # Found the answer! Return 1-indexed positions
                return [left + 1, right + 1]
            
            # STEP 5a: Sum is too small, need a larger number
            elif current_sum < target:
                # Since array is sorted, numbers[left] is the smallest 
                # unused number
                # All pairs with numbers[left] will be too small
                # So we eliminate numbers[left] and move to next smallest
                left += 1
                
                # What we're eliminating:
                # If left=0, we eliminate pairs: (0,1), (0,2), (0,3), ..., 
                #(0,right)
                # All these pairs would have sum ≤ current_sum < target
            
            # STEP 5b: Sum is too large, need a smaller number  
            else:  # current_sum > target
                # Since array is sorted, numbers[right] is the largest 
                # unused number
                # All pairs with numbers[right] will be too large
                # So we eliminate numbers[right] and move to next largest
                right -= 1
                
                # What we're eliminating:
                # If right=3, we eliminate pairs: (left,3), (left+1,3), ..., 
                # (2,3)
                # All these pairs would have sum ≥ current_sum > target
        
        # STEP 6: If we exit the loop, no solution exists
        # (This shouldn't happen according to problem constraints)
        return []

```

### Example 2: Same Direction Pointers (Slow/Fast)

LEETCODE LINK [https://leetcode.com/problems/remove-duplicates-from-sorted-array/](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

**Solution Strategy**

1. Keep the first element (it's always unique)
2. Use two pointers:
    - `i`: Scans through all elements
    - `k`: Tracks where to place the next unique element
3. Compare current element with the last confirmed unique element
4. Only update when we find a new unique element

**Time/Space Complexity**

- Time: O(n) - single pass through array
- Space: O(1) - in-place modification

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        """
        Remove duplicates from sorted array in-place using two-pointer technique.
        
        Args:
            nums: List[int] - sorted array that may contain duplicates
            
        Returns:
            int - number of unique elements (also length of deduplicated portion)
            
        The array is modified in-place so that the first k elements contain 
        the unique elements in their original order.
        
        Example:
            Input:  nums = [1,1,2,2,3]
            Output: 3
            Result: nums = [1,2,3,2,3] (first 3 elements are unique)
        """
        
        # Edge case: empty array
        if not nums:
            return 0
            
        # Edge case: single element (always unique)
        if len(nums) == 1:
            return 1
        
        # MAIN ALGORITHM - Two Pointer Technique
        
        # k tracks the position where next unique element should be placed
        # Start at 1 because nums[0] is always unique (first element)
        k = 1
        
        # i iterates through all elements starting from index 0
        for i in range(len(nums)):
            
            # KEY COMPARISON: Check if current element is different from 
            # the last confirmed unique element (which is at position k-1)
            if nums[i] != nums[k-1]:
                
                # FOUND A NEW UNIQUE ELEMENT!
                
                # Place the unique element at position k
                nums[k] = nums[i]
                
                # Move k to the next position for future unique elements
                k += 1
                
                # Note: We don't need an 'else' clause because if nums[i] 
                # equals nums[k-1], it's a duplicate and we simply skip it
        
        # Return the count of unique elements
        # The first k elements of nums now contain all unique values
        return k

```

### Example 3: Shuffle the Array

LEETCODE LINK [https://leetcode.com/problems/shuffle-the-array/](https://leetcode.com/problems/shuffle-the-array/)

**Solution Strategy**

1. Identify the two halves in the input array
    - First half: indices `0` to `n-1` (elements x₁, x₂, ..., xₙ)
    - Second half: indices `n` to `2n-1` (elements y₁, y₂, ..., yₙ)
2. Use single loop through first half only
    - `i`: Iterates from `0` to `n-1`
    - For each `i`, access both `nums[i]` and `nums[i + n]`
3. Alternate pattern for each iteration
    - Take element from first half: `nums[i]`
    - Take corresponding element from second half: `nums[i + n]`
    - Add both to result in sequence
4. Key insight: Index mapping
    - Position `i` in first half maps to position `i + n` in second half
    - This creates the interleaved pattern: `[x₁, y₁, x₂, y₂, ..., xₙ, yₙ]`

**Time/Space Complexity**

- **Time**: O(n) - single pass through array, each element accessed exactly once
- **Space**: O(n) - create new result array of size 2n to store shuffled elements

```python
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        """
        Shuffle Array - Interleave first and second half
        
        Problem: Transform [x1,x2,...,xn,y1,y2,...,yn] 
                 Into:    [x1,y1,x2,y2,...,xn,yn]
        
        Strategy: For each position i in first half:
                 - Take nums[i] (from first half)
                 - Take nums[i+n] (corresponding element from second half)
                 - Add both to result in order
        
        Time Complexity: O(n) - visit each element once
        Space Complexity: O(n) - for the result array
        """
        
        # Initialize empty result array to store shuffled elements
        result = []
        
        # Loop through each position in the FIRST HALF only
        # We only need to iterate n times (not 2n times)
        for i in range(n):
            
            # STEP 1: Take element from FIRST HALF
            # nums[i] gives us x1, x2, x3, ... xn
            result.append(nums[i])
            
            # STEP 2: Take corresponding element from SECOND HALF  
            # nums[i + n] gives us y1, y2, y3, ... yn
            # When i=0: nums[0+n] = nums[n] = first element of second half
            # When i=1: nums[1+n] = nums[n+1] = second element of second half
            # When i=2: nums[2+n] = nums[n+2] = third element of second half
            result.append(nums[i + n])
        
        # Return the shuffled array
        return result

```

## 2. In-place Manipulation

Interviewers often look for solutions that modify the array directly without using extra memory. This tests your understanding of memory management and pointer manipulation.

**Use Case**: Moving all negative elements to one side of an array. A common strategy involves using the array itself to store intermediate results, reducing space complexity from

O(n) to O(1).

### Example 1: Move Zeros

LEEDCODE  [https://leetcode.com/problems/move-zeroes/description/](https://leetcode.com/problems/move-zeroes/description/)

**Solution Strategy**

1. Initialize a `write_index` pointer to track where the next non-zero element should be placed
2. **First pass**: Iterate through the entire array:
    - When encountering a non-zero element, place it at `write_index` position
    - Increment `write_index` to point to the next available position
    - Skip zero elements (don't move `write_index`)
3. **Second pass**: Fill all remaining positions from `write_index` to end of array with zeros
4. This approach maintains the relative order of non-zero elements while moving all zeros to the end

**Time/Space Complexity**

- Time: O(n) - two passes through the array, each element processed at most twice
- Space: O(1) - in-place modification using only one additional pointer variable

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        write_index = 0
        
        # First pass: move all non-zeros to front
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[write_index] = nums[i]
                write_index += 1
        
        # Second pass: fill remaining positions with zeros
        while write_index < len(nums):
            nums[write_index] = 0
            write_index += 1
```

## 3. Sliding Window

This pattern involves maintaining a "window" (a sub-array) of a certain size that moves through the main array. It's excellent for problems involving contiguous sub-arrays.

**Use Case**: Finding the contiguous subarray with the largest sum.  As you slide the window, you update the current sum. If the sum becomes negative, you reset the window's start because a negative prefix won't contribute to a larger maximum sum.

### Example 1: Maximum Average Subarray I

LEETCODE : [https://leetcode.com/problems/maximum-average-subarray-i/description/](https://leetcode.com/problems/maximum-average-subarray-i/description/)

**Solution Strategy**

1. Calculate the initial sum of the first `k` elements to establish the first window
2. Set `max_sum` to this initial window sum as our baseline
3. **Slide the window**: Starting from index `k`, iterate through the rest of the array:
    - Add the new element entering the window on the right (`nums[i]`)
    - Remove the element leaving the window on the left (`nums[i-k]`)
    - Update `max_sum` if current window sum is larger
4. Return the maximum sum divided by `k` to get the maximum average

**Time/Space Complexity**

- Time: O(n) - O(k) for initial sum + O(n-k) for sliding window = O(n) total
- Space: O(1) - only using a few variables regardless of input size

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # Initial sum of first window
        window_sum = sum(nums[:k])
        max_sum = window_sum

        # Slide window
        for i in range(k, len(nums)):
            window_sum += nums[i] - nums[i - k]  # add right, remove left
            max_sum = max(max_sum, window_sum)

        # Return maximum average
        return max_sum / k
```

# Stack Overview

## What is a Stack?

A **stack** is a linear data structure adhering to the **LIFO (Last-In, First-Out)** principle. It's restricted to adding (pushing) and removing (popping) elements from the same end, known as the "top."

```python
Initial Stack:        After push(40):       After pop():
┌─────────┐          ┌─────────┐           ┌─────────┐
│   30    │ ← TOP    │   40    │ ← TOP     │   30    │ ← TOP
├─────────┤          ├─────────┤           ├─────────┤
│   20    │          │   30    │           │   20    │
├─────────┤          ├─────────┤           ├─────────┤
│   10    │          │   20    │           │   10    │
└─────────┘          ├─────────┤           └─────────┘
                     │   10    │
                     └─────────┘
```

## Core Operations

- **Push**: Add element to the top
- **Pop**: Remove element from the top
- **Top/Peek**: View the top element without removing
- **isEmpty**: Check if stack is empty
- **Size**: Get number of elements

## Key Properties

- **LIFO**: Last element added is first to be removed
- **Restricted access**: Can only access the top element
- **Dynamic size**: Can grow and shrink during runtime

## Stack Applications in Data Science & Interviews

Stacks are fundamental for handling nested or recursive structures.

1. **Parsing and Evaluating Expressions**: Stacks are perfect for problems like evaluating mathematical formulas or checking for balanced parentheses.  When you see a problem that involves matching pairs or following a specific order of operations, a stack should come to mind.
    - **Example**: To check for balanced parentheses, you push opening brackets onto the stack. When a closing bracket appears, you pop from the stack and check if it’s a match. If the stack is empty at the end, the string is balanced.
2. **Managing Recursion**: Under the hood, programming languages use a "call stack" to manage function calls. Recursive algorithms can often be implemented iteratively using a stack, which can help prevent stack overflow errors for deep recursions.
    - **Example**: Depth-First Search (DFS) on a tree or graph can be implemented recursively or iteratively with a stack.

## Stack Cheat Sheet

```python
# Creation
stack = []

# Operations
stack.append(item)    # Push
item = stack.pop()    # Pop
top = stack[-1]       # Peek
empty = len(stack) == 0  # Check empty

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, n):
        self.stack.append(n)

    def pop(self):
        return self.stack.pop()
```

### Example 1: Baseball Game Operation

LEETCODE LINK  [https://leetcode.com/problems/baseball-game/description/](https://leetcode.com/problems/baseball-game/description/)

**Solution Strategy**

1. Initialize an empty `record` list to act as a stack for storing valid scores
2. Iterate through each operation in the input:
    - If operation is `'+'`: Add the sum of the last two scores to the record
    - If operation is `'D'`: Add double the last score to the record
    - If operation is `'C'`: Remove (cancel) the most recent score from the record
    - Otherwise: The operation is a number, convert to integer and add to record
3. Return the sum of all remaining scores in the record

**Time/Space Complexity**

- Time: O(n) - single pass through operations, each operation processed in O(1) time
- Space: O(n) - record list can store up to n scores in worst case (when no cancellations occur)

```python
class Solution:
    def calPoints(self, operations: List[str]) -> int:
        record=[]
        for op in operations:
            if op == '+':
                record.append(record[-1]+record[-2])
            elif op== 'D':
                record.append(record[-1]*2)
            elif op== 'C':
                record.pop()
            else:
                record.append(int(op))
        return sum(record)
```

### Example 2: Valid Parentheses

Leetcode [https://leetcode.com/problems/valid-parentheses/description/](https://leetcode.com/problems/valid-parentheses/description/)

**Solution Strategy**

1. Initialize an empty `stack` to track opening brackets and create a `mapping` dictionary to match closing brackets with their corresponding opening brackets
2. Iterate through each character in the input string:
    - If character is a **closing bracket** (`')'`, `'}'`, `']'`):
        - Check if stack is empty or if the top element doesn't match the expected opening bracket
        - If either condition is true, return `False` (invalid)
        - Otherwise, the `pop()` operation removes the matched opening bracket
    - If character is an **opening bracket**: Push it onto the stack
3. After processing all characters, return `True` only if stack is empty (all brackets were properly matched)

**Time/Space Complexity**

- Time: O(n) - single pass through the string, each character processed in O(1) time
- Space: O(n) - stack can store up to n/2 opening brackets in worst case scenario

```python
def isValid(self,s: str) -> bool:
    """
    Stack-based solution for Valid Parentheses
    Time: O(n), Space: O(n)
    """
    # Stack to keep track of opening brackets
    stack = []
    
    # Mapping of closing to opening brackets
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        # If it's a closing bracket
        if char in mapping:
            # Check if stack is empty or top doesn't match
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # It's an opening bracket, push to stack
            stack.append(char)
    
    # Valid if all brackets were matched (stack is empty)
    return len(stack) == 0
```

### Example 3: **Min Stack**

LEETCODE [https://leetcode.com/problems/min-stack/](https://leetcode.com/problems/min-stack/)

**Solution Strategy**

1. **Initialize two parallel stacks**:
    - `stack`: Stores all actual values
    - `min_stack`: Stores the minimum value available at each level
2. **Push operation**:
    - Always add the new value to the main stack
    - For min_stack: if new value is ≤ current minimum (or min_stack is empty), add the new value; otherwise, duplicate the current minimum
    - This ensures min_stack[top] always contains the current overall minimum
3. **Pop operation**:
    - Remove from both stacks simultaneously
    - After removal, min_stack automatically reflects the minimum of remaining elements
4. **Top operation**: Return the top element from main stack
5. **GetMin operation**: Return the top element from min_stack (current minimum in O(1) time)

**Time/Space Complexity**

- Time: O(1) - all operations (push, pop, top, getMin) perform constant-time stack operations
- Space: O(n) - two stacks each storing up to n elements, so 2n total space

```python
class MinStack:
    def __init__(self):
        # Create two separate stacks to solve the problem
        self.stack = []      # Main stack: stores all the actual values
        self.min_stack = []  # Helper stack: stores the minimum value at each level
        
        # KEY INSIGHT: min_stack[i] contains the minimum value 
        # among all elements from bottom up to position i in main stack
    
    def push(self, val: int) -> None:
        # Step 1: Always add the new value to main stack
        self.stack.append(val)
        
        # Step 2: Decide what to add to min_stack
        # We need to store what the minimum is after adding this value
        
        if not self.min_stack or val <= self.min_stack[-1]:
            # Case 1: min_stack is empty OR new value is smaller/equal to current min
            # This value becomes the new minimum, so store it
            self.min_stack.append(val)
        else:
            # Case 2: new value is larger than current minimum
            # The minimum doesn't change, so keep the previous minimum
            self.min_stack.append(self.min_stack[-1])
        
        # RESULT: Both stacks are same length, min_stack[top] = current minimum
    
    def pop(self) -> None:
        # Only pop if there are elements to remove
        if self.stack:
            # Remove from both stacks simultaneously
            self.stack.pop()      # Remove the actual value
            self.min_stack.pop()  # Remove the corresponding minimum info
            
            # MAGIC: After popping, min_stack automatically shows 
            # what the minimum was BEFORE we added the removed element!
    
    def top(self) -> int:
        # Return the top element from main stack
        # [-1] means "last element" (top of stack)
        return self.stack[-1]
    
    def getMin(self) -> int:
        # Return the current minimum in O(1) time!
        # The top of min_stack always contains the current minimum
        return self.min_stack[-1]
        
        # WHY THIS WORKS: min_stack[top] stores the minimum among 
        # all elements currently in the main stack
```

## Queue Overview

## What is a Queue?

A **queue** is a linear data structure that follows the **FIFO (First-In, First-Out)** principle. Elements are added (enqueued) to the rear and removed (dequeued) from the front.

```python
FRONT                                    REAR
  ↓                                        ↓
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  10 │  20 │  30 │  40 │  50 │  60 │  70 │  80 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↑                                        ↑
Remove from here              Add new elements here
(dequeue)                         (enqueue)
```

## Core Operations

- **Enqueue**: Add element to the rear
- **Dequeue**: Remove element from the front
- **Front**: View the front element without removing
- **Rear**: View the rear element without removing
- **isEmpty**: Check if queue is empty
- **Size**: Get number of elements

## Key Properties

- **FIFO**: First element added is first to be removed
- **Two-ended access**: Add at rear, remove from front
- **Dynamic size**: Can grow and shrink during runtime

## Queue Applications in Data Science & Interviews

Queues are ideal for processing items in the order they were received.

1. **Breadth-First Search (BFS)**: This is the most common application of queues in interviews. BFS explores a graph or tree layer by layer.
    - **Example**: Finding the shortest path between two nodes in an unweighted graph is a classic BFS problem. You start at the source node, add it to the queue, and then iteratively process each node's neighbors, keeping track of the distance at each step.
2. **Task Scheduling**: In systems like Apache Airflow, tasks are often placed in a queue to be executed by worker nodes. This ensures that jobs are processed in an orderly fashion.
3. **Data Streaming**: Queues can manage incoming data points in a stream, ensuring that they are processed sequentially. This is relevant in real-time analytics and monitoring systems.

### Queue Cheat Sheet

```python
# Creation
from collections import deque
queue = deque()

# Operations
queue.append(item)     # Enqueue
item = queue.popleft() # Dequeue
front = queue[0]       # Front
rear = queue[-1]       # Rear
empty = len(queue) == 0  # Check empty

```

---

### Example 1 Implement Queue using Stacks

LEETCODE [https://leetcode.com/problems/implement-queue-using-stacks/description/](https://leetcode.com/problems/implement-queue-using-stacks/description/)

**Solution Strategy**

1. **Initialize two stacks**:
    - `in_stack`: Handles all incoming elements (push operations)
    - `out_stack`: Handles all outgoing elements (pop/peek operations)
2. **Push operation**: Always add new elements to `in_stack` (maintains insertion order)
3. **Pop/Peek operations**:
    - Call helper method `move_in_to_out()` to ensure `out_stack` has the front element
    - Pop or peek from `out_stack` (which now has the oldest element on top)
4. **Transfer logic** (`move_in_to_out`):
    - Only transfer when `out_stack` is empty (avoids disrupting order)
    - Move all elements from `in_stack` to `out_stack`, which reverses their order
    - Now oldest element (queue front) is on top of `out_stack`
5. **Empty check**: Queue is empty only when both stacks are empty

**Time/Space Complexity**

- Time: O(1) amortized for all operations - each element is moved at most once between stacks
- Space: O(n) - two stacks storing up to n total elements

```python
class MyQueue:
    def __init__(self):
        # Stack for incoming elements (push operations)
        self.in_stack = []
        # Stack for outgoing elements (pop/peek operations)
        self.out_stack = []

    def push(self, x: int) -> None:
        # Always push new elements onto in_stack
        # This is O(1)
        self.in_stack.append(x)

    def pop(self) -> int:
        # Ensure out_stack has the current front element
        self.move_in_to_out()
        # Pop from out_stack (front of queue)
        return self.out_stack.pop()

    def peek(self) -> int:
        # Ensure out_stack has the current front element
        self.move_in_to_out()
        # Peek top of out_stack (front of queue)
        return self.out_stack[-1]

    def empty(self) -> bool:
        # Queue is empty only if BOTH stacks are empty
        return not self.in_stack and not self.out_stack

    def move_in_to_out(self):
        """
        Transfer elements from in_stack to out_stack
        BUT only when out_stack is empty.
        This reverses the order, so the oldest element
        (front of the queue) ends up on top of out_stack.
        """
        if not self.out_stack:
            while self.in_stack:
                # Pop from in_stack and push onto out_stack
                self.out_stack.append(self.in_stack.pop()
```

## Key Differences between Stack and Queue

| Aspect | Stack | Queue |
| --- | --- | --- |
| **Principle** | LIFO (Last In, First Out) | FIFO (First In, First Out) |
| **Access Points** | One end (top) | Two ends (front & rear) |
| **Add Operation** | Push (to top) | Enqueue (to rear) |
| **Remove Operation** | Pop (from top) | Dequeue (from front) |
| **Real-world Analogy** | Stack of plates | Line at store |
| **Use Cases** | Function calls, undo operations | Task scheduling, printing |

---

# Practice questions

## **1. Arrays (In-place Manipulation & Two Pointers)**

*(Different from Move Zeroes / Remove Duplicates)*

- [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
    
    *In-place merge of two sorted arrays.*
    
- [Sort Array By Parity](https://leetcode.com/problems/sort-array-by-parity/)
    
    *Reorder array to put even numbers first using two pointers.*
    
- [Height Checker](https://leetcode.com/problems/height-checker/)
    
    *Count how many are out of order compared to sorted array.*
    
- [Relative Sort Array](https://leetcode.com/problems/relative-sort-array/)
    
    *Sort one array based on order of another.*
    

---

## **2. Sliding Window (Different Average/Window Problems)**

- [Maximum Sum of Distinct Subarrays With Length K](https://leetcode.com/problems/maximum-sum-of-distinct-subarrays-with-length-k/)
    
    *Sliding window + set for distinct sum.*
    
- [Longest Subarray of 1’s After Deleting One Element](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/)
    
    *Window expansion and shrinkage logic.*
    
- [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
    
    *Find smallest subarray with sum ≥ target.*
    
- [Grumpy Bookstore Owner](https://leetcode.com/problems/grumpy-bookstore-owner/)
    
    *Window sum optimization for customer satisfaction.*
    

---

## **3. Stacks (Different Stack Use Cases)**

- [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
    
    *Stack to find next greater numbers.*
    
- [Crawler Log Folder](https://leetcode.com/problems/crawler-log-folder/)
    
    *Stack simulation for folder navigation.*
    
- [Remove Outermost Parentheses](https://leetcode.com/problems/remove-outermost-parentheses/)
    
    *Track depth of parentheses with stack.*
    
- [Make The String Great](https://leetcode.com/problems/make-the-string-great/)
    
    *Use stack to remove adjacent opposite-case letters.*