# 原因一：学了没用的特征
- 例如user_feat，该user_feat下无论推什么都不会点击，所以ctr没提升；
- 模型比较弱的时候，排序比较混乱时，低质量 item被压制是有用的，但当模型比较好了，低质量 item被压制是没用的，而应提高高质量item的score，正样本的识别能力；

# 原因二：Position-Bias
- 论文参考：User browsing models: relevance versus examination
  1. 线下评估使用的是ctr（综合考虑了true relevance和position bias），而线上时默认所有item的位置相同，即只考虑true relevance；
  2. 对position_bias过拟合；

# 原因三：线下线上数据分布偏差
1. 对没有见过的样本泛化性差；
2. 新模型，一开始相当于都是在拟合老模型产生的样本，刚上线效果如果比较差，经过一段时间迭代，影响的样本分布慢慢趋近于新模型
    - 解决方法1. 对无偏数据进行上采样；
    - 解决方法2. 新老模型融合，老模型权重$\alpha$不断衰减； 

# 原因四：特征穿越