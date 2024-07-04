# Position Embedding详解

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

2. ALiBi


3. Rope[<sup>2</sup>](#2)
根据余弦和的定义
$$

A_{s,t}=x_t^TW_Q^TR_{\theta, s}R_{\theta, t}^dW_Kx_s=x_t^TW_Q^TR_{\theta, s-t}^dW_Kx_s \\

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

{\theta}_n = \frac{1}{10000^{\frac{2n}{d}}}
$$

   - GPT_NeoX Style

## 优缺点

## 额外特性
### 外推性
（怎么做，原理是什么）

## Ref
<div id="1"></div>

[1][Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v3)

<div id="2"></div>

[2][ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://arxiv.org/pdf/2104.09864)

