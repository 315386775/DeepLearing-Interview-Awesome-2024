# 01. C++中与类型转换相关的4个关键字特点及应用场合

```c++
static_cast<type_id> ()  // 主要用于C++内置基本类型之间的转换
const_cast<>()  //　用于将const类型的数据和非const类型的数据之间进行转换
dynamic_cast<>()  //　可以将基类对象指针(引用)cast到继承类指针，（类中必须有虚函数）
reinterpret_cast<>()  //　type_id必须是指针，引用...可以把一个指针与内置类型之间进行转换
```

# 02. Python装饰器及其作用

- 详细讲述链接：https://foofish.net/python-decorator.html

- Python中的函数可以像普通变量一样当做参数传递给另外一个函数，装饰器本质上是一个类/函数，它可以在不修改函数主体功能的前提下增加额外的功能，装饰器的返回值也是一个类对象或者函数，可以用来插入日志，性能测试等等，装饰器的使用提高了函数的复用性，增加了程序的可读性。

# 03. map,lambda,filter,reduce的用法

```python
##  lambda匿名函数
# lambda 参数:操作(参数)
add = lambda x, y: x + y
print(add(3, 5))
# 按照第二个数排序
a = [(1, 2), (4, 1), (9, 10), (13, -3)]
a.sort(key=lambda x: x[1])

# map会将一个函数映射到一个输入列表的所有元素上 map(func, list_of_inputs)
# func一般会结合lambdas函数食用
squared	= list(map(lambda x: x**2, lists))

# filter过滤列表中的元素,并且返回一个由所有符合要求的元素所构成的列表,符合要求即函数映射到该元素时返回值为True.	
less_than_zero = filter(lambda x: x<0, number_list)

from functools import reduce  # reduce() 函数会对参数序列中元素进行累计。
product	= reduce((lambda x, y:x*y),[1,2,3,4])  # 24

```

# 02. 用Numpy的广播机制实现矩阵之间距离的计算

- 在符合广播条件的前提下，广播机制会为尺寸较小的向量添加一个轴（广播轴），使其维度信息与较大向量的相同。

- 计算 m*2 的矩阵与 n * 2 的矩阵中，m*2 的每一行到 n*2 的两两之间欧氏距离。

```python
# L2 = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)

def L2_dist_1(cloud1, cloud2):
    m, n = len(cloud1), len(cloud2)
    # project 01
    # cloud1 = np.repeat(cloud1, n, axis=0) # (n*m,2)
    # cloud1 = np.reshape(cloud1, (m, n, -1)) # (m,n,2) (n,2)

    # project 02
    # cloud1 = cloud1[:, None, :] # (m,1,2)
    
    # project 03
    cloud1 = np.expand_dims(cloud1, 1)
    
    dist = np.sqrt(np.sum((cloud1 - cloud2)**2, axis=2))
    return dist
```