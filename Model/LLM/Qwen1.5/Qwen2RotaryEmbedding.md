$N_p$ is the max_position_embedding<br>
$freqs$ shape = [max_position_embeeding, d//2]
<br>
<br>
$$
inv\_freq = \frac{1}{base^{[0,2,4,...,d]/d}}\\
freqs = [0,1,2,...,N_p] \times inv\_freq  
$$

- concat 后计算cos和sin
```python
emb = torch.cat((freqs, freqs), dim=-1)
self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
```
- 所以cos和sin分别为（举例base=10000，cos矩阵）：
$$
  \left[
  \begin{matrix}

   cos(0*\frac{1}{1000^{0/d}}), & cos(0*\frac{1}{1000^{2/d}}), & ..., &  cos(0*\frac{1}{1000^{(d-2)/d}}), & cos(0*\frac{1}{1000^{(d)/d}}), & cos(0*\frac{1}{1000^{0/d}}), & cos(0*\frac{1}{1000^{2/d}}), &  ..., &  cos(0*\frac{1}{1000^{(d-2)/d}}), & cos(0*\frac{1}{1000^{(d)/d}}) \\

   cos(1*\frac{1}{1000^{0/d}}), & cos(1*\frac{1}{1000^{2/d}}), & ..., &  cos(1*\frac{1}{1000^{(d-2)/d}}), & cos(1*\frac{1}{1000^{(d)/d}}), & cos(1*\frac{1}{1000^{0/d}}), & cos(1*\frac{1}{1000^{2/d}}), &  ..., &  cos(1*\frac{1}{1000^{(d-2)/d}}), & cos(1*\frac{1}{1000^{(d)/d}}) \\

   ... \\

   cos(N_p*\frac{1}{1000^{0/d}}), & cos(N_p*\frac{1}{1000^{2/d}}), & ..., &  cos(N_p*\frac{1}{1000^{(d-2)/d}}), & cos(N_p*\frac{1}{1000^{(d)/d}}), & cos(N_p*\frac{1}{1000^{0/d}}), & cos(N_p*\frac{1}{1000^{2/d}}), &  ..., &  cos(N_p*\frac{1}{1000^{(d-2)/d}}), & cos(N_p*\frac{1}{1000^{(d)/d}}) \\

  \end{matrix} 
\right]
$$


