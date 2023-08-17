# 背景
## auto-regressive
1. predict在 $t$ 时，需要与 $predict_{t-1}$ 拼接作为 $predict_t$ 的输入;
<br><b><font color='green'>PS: self-attention输出维度为 [seq_len, batch, hidden_size] ，选择 [-1, :] 作为下一次predict的输入</font> 
2. 随着 $t$ 的增加，FLOPS计算也会越来越多；


```shell
step 0 input: Lionel Messi is a player
step 1 input: Lionel Messi is a player who
step 2 input: Lionel Messi is a player who has
step 3 input: Lionel Messi is a player who has been
step 4 input: Lionel Messi is a player who has been a
step 5 input: Lionel Messi is a player who has been a key
step 6 input: Lionel Messi is a player who has been a key part
step 7 input: Lionel Messi is a player who has been a key part of
step 8 input: Lionel Messi is a player who has been a key part of the
step 9 input: Lionel Messi is a player who has been a key part of the team
step 10 input: Lionel Messi is a player who has been a key part of the team's
step 11 input: Lionel Messi is a player who has been a key part of the team's success
step 12 input: Lionel Messi is a player who has been a key part of the team's success.
step 13 input: Lionel Messi is a player who has been a key part of the team's success.

 Input: Lionel Messi is a
Output: Lionel Messi is a player who has been a key part of the team's success.
```

# 方法
1. 保存kv值，代码如下
```
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None: # 当输出第一个token后，layer_past就是非None了
            past_key, past_value = layer_past # 取出之前计算好的 key, value
            key = torch.cat((past_key, key), dim=-2) # past_key 与当前 token 对应的 key 拼接
            value = torch.cat((past_value, value), dim=-2) # past_value 与当前 token 对应的 value 拼接

        if use_cache is True:
            present = (key, value)
        else:
            present = None
```
2. KV Cache 配置开启后，推理过程可以分为2个阶段：
  - 预填充阶段：发生在计算第一个输出token过程中，这时Cache是空的，计算时需要为每个 transformer layer 计算并保存key cache和value cache，在输出token时Cache完成填充；FLOPs同KV Cache关闭一致，存在大量gemm操作，推理速度慢。
  - 使用KV Cache阶段：发生在计算第二个输出token至最后一个token过程中，这时Cache是有值的，每轮推理只需读取Cache，同时将当前轮计算出的新的Key、Value追加写入至Cache；FLOPs降低，gemm变为gemv操作，推理速度相对第一阶段变快，这时属于Memory-bound类型计算。

3. 因为decoder是mask attention的存在，所以 t+1 时刻变化不会引起 t及之前时刻的weight变化；