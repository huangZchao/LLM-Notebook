- [激活函数 GELU](#激活函数-gelu)
- [GLU 层](#glu-层)
  - [激活层（GLU层） Code](#激活层glu层-code)
- [位置编码 RoPE](#位置编码-rope)
- [注意力层](#注意力层)
  - [自注意力机制](#自注意力机制)
- [GLMBlock](#glmblock)
- [ChatGLM2-6B 推理](#chatglm2-6b-推理)
- [ChatGLM-130B](#chatglm-130b)


# 激活函数 GELU
$$
GELU(x) = 0.5x(1+tanh(\sqrt\frac{2}{\pi}(x+0.044715x^3)))
$$

# GLU 层
$$
GLU(X)=GELU(XW_1)W_2
$$

## 激活层（GLU层） Code
```python3
    def forward(self, hidden_states):
        """
        hidden_states: [seq_len, batch, hidden_size]
        """
​
        # [seq_len, batch, inner_hidden_size]
        # 投影
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # 激活
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # 投影
        output = self.dense_4h_to_h(intermediate_parallel)
​
        return output
```

# 位置编码 RoPE
<font color='red'><b>待补充</font>

# 注意力层
![attn_layer](../Img/attn_layer.jpg)
1. 随机mask 
2. mask token 随机打乱
3. self mask

## 自注意力机制
<font color='green'><b>公式略</font>

# GLMBlock
![attn_layer](../Img/GLM-BLock.png)


# ChatGLM2-6B 推理
和CHATGLM的新增点：<Br>
1. Multi Query Attention
   ![Alt text](../Img/multi-query-attention.png)
   a. 推理时，把$K,V$ 做 $meanpooling$ <br>
   b. query不变（multi-query）.vs. query分组（group-query）<br>
   ![Alt text](../Img/mult-group-query-head.png)
   c. 计算结果仍然时concat

# ChatGLM-130B
和CHATGLM的不同点：<Br>
1. Normalize Layer: DeepNorm <Br>
   a. $Layernorm (x + f(x)) ---> Layernorm(x*\alpha + f(x))$.
2. Activation Func: GEGLU <br>
   ![Alt text](../Img/GEGLU.png)