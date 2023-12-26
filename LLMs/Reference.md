# Transfomer中Attention的计算量如何计算

- 代码中的to_qkv()函数，即用于生成q、k、v三个特征向量

![Alt](assert/attention.png#pic_center=600x400)


```python

self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
self.to_out = nn.Linear(inner_dim, dim)

```