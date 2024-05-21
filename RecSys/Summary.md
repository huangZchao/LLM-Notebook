# 算法基本能力
## 1.基础工作常识
### a.推荐链路各个节点的定位
#### i.召回
#### ii.粗排（粗排的使命，粗排的评估）
#### iii.精排
#### iv.LTR（与精排串行，可用精排目标作为特征，做一些小模型，相当于stacking）
#### v.重排
#### vi.混排

b.工作的评估，比如粗精排一致性，多样性召回评估方式等
#除了粗排样本选用曝光非曝光外，还有没有其他方式，如蒸馏？
#多样性如何评估

c.查case的基础
i.先确定是不是问题，再确定问题出在哪
ii.定位链路的环节，定位目标（如pos_delta)，规则
#举个例子

d.寻找优化点的能力，如从case中发现优化点，比如刷图文刷不到nba，刷视频能刷到nba，那就做一个视频-图文的兴趣泛化
2.分链路基本能力
a.召回
i.召回的样本构建，离线评估，在线评估（独占比，一致性），多样性
#独占比、一致性是什么意思？
ii.多兴趣召回（mind， comirec等）
iii.强化学习召回（youtube 2020那个论文）
iv.多目标召回（多个塔实现多目标召回）
#会存在哪些问题
v.i2i召回（swing，graph）
vi.cascade 召回
vii.对比学习
viii.召回rerank（用一个小的模型去对召回结果重排）
ix.个性化quota代替固定weight
x.粗排merge的方式（如蛇形merge）
#蛇形merge有材料不
xi.vae召回
b.粗排
i.MTL粗排（小模型，蒸馏精排，粗排sim（一般是依赖ps回写，离线match特征等方式)）
ii.LTR（cascade） 以精排为目标，加强链路一致性 
1.pair-wise： bpr-loss
2.list-wise： listnet， listmle
iii.小粗排（对部分效果好的召回单独过一个小粗排）
iv.轻量替代attention的方式： https://zhuanlan.zhihu.com/p/362472527
c.精排
i.序列建模（sim，attention，fafe，match特征, m3net）
ii.多目标建模（cgc，mmoe， loss的reweight， gradnorm等解决梯度量纲）
iii.个性化网络参数（lhuc，poso，这里需要注意用什么特征，乘到哪儿等）
iv.回归目标优化（distill softmax， wce， d2q, wpr, dsst, tpm ）
v.网络交叉（mmcn，deepfm，xdeepfm，dcn，dcn-m）
d.融合公式
i.https://zhuanlan.zhihu.com/p/500237779（经典问题如加法跟乘法的区别）
ii.rank_index, norm
e.特征
i.特征重要度评估
ii.做特征的方法，特征的分类https://www.zhihu.com/question/419906651/answer/3356641671
f.冷启动
i.冷启动评估（高热率，保量成功率等）
ii.冷启动实验开法（独立小世界）
1.https://zhuanlan.zhihu.com/p/666464064
iii.冷启控速
iv.个性化分配冷启流量
v.冷启召回（用多模态，对比学习等）
g.多样性
i.多样性评估
ii.常见算法： centriod，dpp算法
iii.常见规则： 粗排，精排后根据某种tag进行打散频控
h.重排
i.GE框架（https://blog.csdn.net/Taobaojishu/article/details/131136898）
1.精排后， 通过不同权重生成多个队列， 随机队列，经过GE框架生成结果（精排后16出8）
ii.规则，如作者打散等
i.混派
i.多体裁混排，如何定坑
ii.如何拉齐量纲，进行pk
j.自动搜参：
i.nas
ii.cem
k.样本构建：
i.fast-emit（最常见的样本拼接方式）
ii.延迟转化问题如何解决，如何提升拼接率
l.推荐系统debias
i.g侧bias： 如时长bias，常见工作有wtd，快手d2q等
ii.u侧bias，常见工作是bias net， 常见的是跟交互结合的分析，比如长按dislike，但是长按还有分享等选项，导致直接deboost dislike会同时导致分享下降，然后只有极少数的群体会去dislike，导致dislike存在bias
iii.还有一些IPS等方式
m.主动创造的能力
i.如与产品合作推动调查问卷的下发，进而建模问卷目标
n.业务调参：
i.对某个群体，如低活单独调参
ii.基于session预估剩下vv等进行调参
iii.基于留存分调参，降频系数调参（比如对低活提升消费权重等）
o.留存建模，敏感度建模：
i.通过类似与s-learner的方法，将留存，反馈归因到item上
p.业务建模：
i.除了常规的精粗排，转赞评目标外，可以基于业务分析加一些二次加工目标等，比如停留时长》30s，回看等目标
q.推荐上的uplift建模
i.可用于垂类，冷启流量分配，给部分用户提权，部分用户降权
ii.用于调参（比如提升时长目标权重会掉其他互动，那么就用uplift找到提升时长掉互动少的用户，给这部分用户boost时长）
3.工程基础
a.推荐系统如何降级
b.首刷缓存等
c.召回索引如何构建
i.faiss
ii.hnsw等
d.通过回写中间结果，节省线上预估时间


FAQ
1.精排后面为什么需要接ltr？
为了calibration，因为精排模型往往会存在高估或者低估；精排模型通常为二分类模型，仅关注序，而不关注绝对值；
calibration基本原则：
a. 不改变ctr序，即AUC不变；
b. 矫正的目的是ctr序的分桶上CoPC接近1；
2.CoPC含义是什么？
copc=(real cvr) / pcvr

