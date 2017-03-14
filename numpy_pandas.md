# NumPy
Note that unlike MATLAB, * is elementwise multiplication, not matrix multiplication. We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices.

```
>>> v = np.array([9,10])
>>> w = np.array([11, 12])
// 元素相乘
>>> v * w
array([ 99, 120])

// 矩陣向量相乘
>>> v.dot(w)
219
>>> 
```

```
>>> a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
>>> a
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
       
// 第一排
>>> a[0]
array([1, 2, 3, 4])
>>> a[0,:]
array([1, 2, 3, 4])

// 第一列
>>> a[:,0]
array([1, 5, 9])

// 前兩列
>>> a[:,0:2]
array([[ 1,  2],
       [ 5,  6],
       [ 9, 10]])
```

