# Prefill Decode分离部署

## 背景
LLM模型推理分为2个阶段：prefill和decode；前者使用attention_mask的方式对同一seq内的token并行计算，而后者则是自回归的逐字解码。
因此，2者的计算瓶颈也存在不同，prefill为compute bound（计算密集型）；而decode是访存密集型（如果使用了kvcache），计算不是特别高（每次迭代只计算一个token）；
1. prefill compute bounder，影响首token耗时;
2. decode 利用kvcache，memory bound，影响decode耗时，通常会增大batch以提供计算量（可使用性能较低的设备）；

## Batch的利用方式
![alt text](image.png)
1. 

## 实施方案


## 优缺点