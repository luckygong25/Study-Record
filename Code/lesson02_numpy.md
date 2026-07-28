很好，从这一课开始，我们真正进入**机器学习的核心世界**。

很多人以为机器学习最重要的是算法，其实不是。

> **机器学习最重要的是数据，而 NumPy 是处理数据最底层、最核心的工具。**

以后你会发现：

* pandas 的底层是 NumPy；
* scikit-learn 的输入是 NumPy；
* PyTorch Tensor 与 NumPy 可以互相转换；
* TensorFlow 的很多思想也来自 NumPy。

所以，只要 NumPy 学扎实，后面的学习会顺畅很多。

---

# 第二讲：NumPy——机器学习的“数据语言”

## 一、为什么不用 Python 的 list？

先看一个例子。

```python
numbers = [1, 2, 3, 4]

numbers * 2
```

很多人以为结果是：

```text
[2,4,6,8]
```

实际上却是：

```text
[1,2,3,4,1,2,3,4]
```

因为 Python 的 list 本质上是**容器**，不是数学向量。

但是 NumPy 不一样。

```python
import numpy as np

numbers = np.array([1,2,3,4])

numbers * 2
```

结果：

```text
[2 4 6 8]
```

这才符合数学中的向量。

机器学习几乎所有数据都需要这种行为。

---

# 二、第一个 NumPy 程序

创建：

```text
lesson02_numpy.ipynb
```

第一格输入：

```python
import numpy as np

print(np.__version__)
```

看看你的 NumPy 版本。

---

第二格：

```python
a = np.array([1,2,3,4])

print(a)
print(type(a))
```

输出类似：

```text
[1 2 3 4]

<class 'numpy.ndarray'>
```

注意：

以后你会不停看到这个词：

```text
ndarray
```

它的意思就是：

> **N-dimensional Array（N维数组）**

这是整个机器学习最重要的数据结构。

---

# 三、创建数组

最常见的方法：

```python
a = np.array([1,2,3])

b = np.array([5,6,7])

print(a)
print(b)
```

二维数组：

```python
matrix = np.array([
    [1,2,3],
    [4,5,6]
])

print(matrix)
```

输出：

```text
[[1 2 3]
 [4 5 6]]
```

以后：

一维数组：

```text
向量
```

二维数组：

```text
矩阵
```

三维数组：

```text
张量
```

以后深度学习几乎都是张量。

---

# 四、查看数组信息

```python
a = np.array([
    [1,2,3],
    [4,5,6]
])

print(a.shape)
```

输出：

```text
(2,3)
```

什么意思？

表示：

```text
2 行

3 列
```

这是机器学习最重要的属性之一。

继续：

```python
print(a.ndim)
```

输出：

```text
2
```

说明这是二维。

继续：

```python
print(a.size)
```

输出：

```text
6
```

总共有 6 个数字。

继续：

```python
print(a.dtype)
```

输出：

```text
int64
```

说明里面都是整数。

以后你会经常检查：

```python
print(X.shape)
```

因为很多模型报错都是 shape 不正确。

---

# 五、创建特殊数组

全部为 0：

```python
np.zeros((3,4))
```

输出：

```text
3 行

4 列

全是 0
```

全部为 1：

```python
np.ones((2,5))
```

随机数：

```python
np.random.rand(3,3)
```

例如：

```text
0.18
0.73
0.56
```

以后神经网络初始化参数都会用随机数。

整数随机：

```python
np.random.randint(0,10,size=(3,4))
```

例如：

```text
[[4 2 8 1]
 [0 6 5 2]
 [7 9 4 8]]
```

---

# 六、数组运算

NumPy 最大的优势就是：

所有运算默认都是逐元素。

```python
a = np.array([1,2,3])

b = np.array([4,5,6])

print(a+b)
```

输出：

```text
[5 7 9]
```

减法：

```python
a-b
```

乘法：

```python
a*b
```

注意：

这里不是矩阵乘法。

而是：

```text
1×4

2×5

3×6
```

得到：

```text
[4 10 18]
```

平方：

```python
a**2
```

得到：

```text
[1 4 9]
```

开方：

```python
np.sqrt(a)
```

---

# 七、求统计量

机器学习经常需要：

平均值：

```python
a.mean()
```

最大：

```python
a.max()
```

最小：

```python
a.min()
```

求和：

```python
a.sum()
```

标准差：

```python
a.std()
```

以后做数据预处理天天都会用。

---

# 八、切片（非常重要）

一维：

```python
a = np.array([10,20,30,40,50])

print(a[:3])
```

得到：

```text
[10 20 30]
```

二维：

```python
matrix = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
```

第一行：

```python
matrix[0]
```

第二列：

```python
matrix[:,1]
```

输出：

```text
[2 5 8]
```

前两行：

```python
matrix[:2]
```

右下角：

```python
matrix[1:,1:]
```

这些操作以后处理数据时几乎每天都会写。

---

# 九、为什么 Shape 如此重要？

假设：

```python
X = np.array([
    [170,65],
    [180,75],
    [160,55]
])
```

它表示：

| 身高  | 体重 |
| --- | -- |
| 170 | 65 |
| 180 | 75 |
| 160 | 55 |

看看：

```python
print(X.shape)
```

输出：

```text
(3,2)
```

意思：

```text
3 个样本

2 个特征
```

机器学习里几乎约定俗成：

> **二维数组的每一行是一个样本（sample），每一列是一个特征（feature）。**

以后我们看到 `(1000, 20)`，第一反应就应该是：

> 有 1000 个样本，每个样本有 20 个特征。

这会贯穿整个课程。

---

# 十、今天的小练习

请不要复制运行，而是自己敲出来。

### 练习1

创建一个数组：

```text
1 2 3 4 5
```

计算：

* 平均值
* 最大值
* 平方
* 开方

---

### 练习2

创建：

```text
4×5
```

随机整数矩阵。

输出：

```python
shape
ndim
size
dtype
```

---

### 练习3

创建：

```python
students = np.array([
    [90,80,70],
    [60,75,88],
    [100,92,95]
])
```

完成：

1. 输出第二个学生成绩。
2. 输出所有学生第一门课成绩。
3. 输出最后两名学生全部成绩。
4. 计算每门课平均成绩（提示：可以查一下 `axis` 参数的作用，先自己猜一猜，再验证）。

---

# 本课的真正目标

今天你不需要记住所有 API，但一定要形成三个核心认识：

1. **NumPy 数组不是 Python 的 list，它是为科学计算设计的数据结构。**
2. **机器学习中的数据几乎都表示为二维数组：行是样本，列是特征。**
3. **以后遇到模型报错，第一件事就是检查 `shape` 和 `dtype`。**

---

## 一个学习建议

从这一课开始，我建议你**不要只运行代码，而要养成“预测输出”的习惯**。

例如，在运行下面这段代码之前，先在纸上或脑中写下你认为的输出：

```python
a = np.array([[1, 2], [3, 4]])
print(a[:, 0])
print(a.shape)
```

然后再运行验证。这个习惯对理解机器学习中的数据流和调试能力帮助非常大。

---

完成今天的练习后，把你的代码（或遇到的问题）发给我。下一讲我们会学习 **NumPy 的广播（Broadcasting）和向量化计算**——这是 NumPy 最强大的特性，也是机器学习能够高效运行的重要原因。
