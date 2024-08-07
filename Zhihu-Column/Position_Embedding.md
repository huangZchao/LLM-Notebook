#! https://zhuanlan.zhihu.com/p/707757424
# Position Embedding详解
（未完成，持续补全中）

## 背景
1. 加入位置信息，如果没有位置编码，举例：
   $$
   Sim(x_m, x_n) = Sim(x_m, x_{n+t}) \\
   if \space x_n = x_{n+t}
   $$
即使$t$非常大，显然不合理；（同样的词，不同的相对间距，相似度应该是不一致的）

## 位置编码种类
### 绝对位置编码
$$
Y = W(x_t+pos_t) + b \\
pos_{t, 2i} =sim(\frac{t}{10000^{\frac{2i}{d}}}) \\ 
 \\
pos_{t, 2i+1} =cos(\frac{t}{10000^{\frac{2i}{d}}}) \\
d为向量维度
$$

### 相对位置编码
1. T5相对位置编码[<sup>1</sup>](#1)
$$
A_{s,t} = q_s^T \times k_t + r_{b(t-s)} \\
b()为函数 \\
r为对应位置的数值
$$

2. ALiBi[<sup>3</sup>](#3)

- 计算softmax前对$qk$乘积增加不可学习的bias；
$$
softmax(q_iK^T+m·[-(i-1),...,-2,-1,0]) \\
$$
- 不同的head的 $m$ 也是不一样
$$
m = 2^{\frac{-8}{n} \times [0,1,2,...,n]} \\
n 为head的数量
$$
![alt text](image.png)

1. Rope[<sup>2</sup>](#2)
根据余弦和的定义
$$
A_{s,t}=x_t^TW_Q^TR_{\theta, s}R_{\theta, t}^dW_Kx_s=x_t^TW_Q^TR_{\theta, s-t}^dW_Kx_s \\
$$
$$
R_{\theta, t}^d=\left[
    \begin{matrix}
    cost{\theta}_1 & -sint{\theta}_1 & ... & 0   & 0   \\
    sint{\theta}_1 & cost{\theta}_1  & ... & 0   & 0   \\
    ...            & ...             & ... & 0   & 0   \\
    ...            & ...             & ... & ... & ... \\
    ...            & ...             & ... & cost{\theta}_{\frac{d}{2}} & -sint{\theta}_{\frac{d}{2}} \\
    ...            & ...             & ... & sint{\theta}_{\frac{d}{2}} & cost{\theta}_{\frac{d}{2}} 
    \end{matrix}
\right] \\
$$
$$
{\theta}_n = \frac{1}{10000^{\frac{2(n-1)}{d}}}
$$

$$
R_{\theta, t}^dx_t=\left(\begin{matrix}
x_{t,0} \\
x_{t,1} \\
x_{t,2} \\
x_{t,3} \\
...\\
x_{t,d-2} \\
x_{t,d-1} \\
  \end{matrix}\right) \left(\begin{matrix}
cost{\theta}_{0} \\
cost{\theta}_{0} \\
cost{\theta}_{1} \\
cost{\theta}_{1} \\
...\\
cost{\theta}_{\frac{d}{2}-1} \\
cost{\theta}_{\frac{d}{2}-1} \\
  \end{matrix}\right) + \left(\begin{matrix}
-x_{t,1} \\
x_{t,0} \\
-x_{t,3} \\
x_{t,2} \\
...\\
-x_{t,d-1} \\
x_{t,d-2} \\
  \end{matrix}\right) \left(\begin{matrix}
sint{\theta}_{0} \\
sint{\theta}_{0} \\
sint{\theta}_{1} \\
sint{\theta}_{1} \\
...\\
sint{\theta}_{\frac{d}{2}-1} \\
sint{\theta}_{\frac{d}{2}-1} \\
  \end{matrix}\right)
$$


## 优缺点
后续补充

## 额外特性
后续补充

### 外推性
#### 原因
```
1、预测的时候用到了没训练过的位置编码（不管绝对还是相对）；
2、预测的时候注意力机制所处理的token数量远超训练时的数量。

引用自苏剑林老师
```
### 1. sliding windows attentions
### 2. yarn
$$
freq\_extra=\frac{1}{base^{\frac{[0,2,4,...,dim]}{dim}}}
$$
$$
freq\_inter=\frac{1}{scaling\_factor * base^{\frac{[0,2,4,...,dim]}{dim}}}
$$
$$
\begin{align}
low =& max(\frac{dim*log^{\frac{Max\_Position}{2\pi*low\_root}}}{2*log^{base}}, 0) \\
hight =& min(\frac{dim*log^{\frac{Max\_Position}{2\pi*high\_root}}}{2*log^{base}}, dim - 1) \\
Num\_Rotation =& low\_root \space or \space high\_root
\end{align}
$$
$$
\begin{align}
inv\_freq\_mask=&1-clamp(\frac{[0,1,2,...,dim//2]-low}{high-low}) \\
inv\_freq=&freq\_inter * (1 - inv\_freq\_mask) + freq\_extra * inv\_freq\_mask
\end{align}
$$
```python
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

t = torch.arange(seq_len, device=device, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
_mscale = float(
    yarn_get_mscale(self.scaling_factor, self.mscale)
    / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
)

emb = torch.cat((freqs, freqs), dim=-1)
self.register_buffer(
    "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
)
self.register_buffer(
    "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
)
```

后续补充
（怎么做，原理是什么）

## Ref
<div id="1"></div>

[1][Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v3)

<div id="2"></div>

[2][ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://arxiv.org/pdf/2104.09864)

<div id="3"></div>

[3][TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/pdf/2108.12409)


