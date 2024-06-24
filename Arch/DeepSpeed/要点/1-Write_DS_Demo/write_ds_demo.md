1. 初始化
```python
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)
```
2. 创建通信进程组
```python
deepspeed.init_distributed()
```
3. 训练
   - auto gradient average
   - auto loss scale
   - 
```python
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
```

# Extra
1. save_checkpoint
   - 每个节点都需要执行，不然会等待所有节点执行完毕；
2. load_checkpoint
3. CUDA_VISIBLE_DEVICES can’t be used with DeepSpeed to control which devices
