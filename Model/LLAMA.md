# RoPE
1. 给$q,k$添加绝对位置信息<br>
$\hat{q}_m=f(q,m), \hat{k}_n=f(k,n)$
2. 计算attention，希望带有相对距离信息<br>
内积计算$<f(q,m), f(k,n)>$
3. 

# LlamaAttention
# LlamaMLP
# LlamaRMSNorm 
# LlamaDecoderLayer
# LlamaModel
多个LlamaDecoderLayer组合成一个LlamaModel
# 末端输出
## LlamaForCausalLM 
## LlamaForSequenceClassification 