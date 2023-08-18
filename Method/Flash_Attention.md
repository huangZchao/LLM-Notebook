# 目的
减少HBM的使用

# 传统Attention 算法流程
  - 算法流程
![传统算法流程](./../Img/传统Attn算法流程.jpg)

# Flash Attention 算法流程
  - 算法流程
![Flash-Attention](../Img/Flash-Attention.jpg)

  - 自我理解
  - 1. 参数 $Q, K, V \in R^{N \times d}$，SRAM size $M$。
  - 2. 设置block大小为，$B_c=\lceil\frac{M}{4d}\rceil,B_r=\min(\lceil\frac{M}{4d}\rceil,d)$.
  - 3. $O=(0)_{N \times d} \in R^{N \times d}$, $l=(0)_N \in R^N$, $m=(-\infin)_N \in R^N$
  - 4. 将$Q$切分为 $T_r = \lceil\frac{N}{B_r}\rceil$ blocks, $Q_1,Q_2,.... \in R^{B_r \times d}$