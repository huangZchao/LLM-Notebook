# 入口
```python
deepspeed.__init__.py
```
## 1. 获取通信名称，一般为nccl
```python
dist_backend = get_accelerator().communication_backend_name()
```
### 1.1. 获取accelerator类型
```python
accelerator.realaccelerator.py 里的
get_accelerator()
```
### 1.2. 跟据1.1的accelerator获取通信后端
```python
# 此处以GPU为例
ds_accelerator = CUDA_Accelerator()
# 返回通信后端
def communication_backend_name(self):
    # self._communication_backend_name = 'nccl'
    return self._communication_backend_name
```
# 2. 创建通信
```python
dist.init_distributed(dist_backend=dist_backend,
                      distributed_port=distributed_port,
                      dist_init_required=dist_init_required)
```
## 2.1. 初始化cdb
1. 如果为nccl，init_deepspeed_backend此步只是简单的打印日志；
2. set_backend，设置cdb，但此时nccl_backend仍为None，需要走TorchBackend；
```python
if cdb is None:
    init_deepspeed_backend(get_accelerator().communication_backend_name(), timeout, init_method)
    set_backend()
    utils.logger.info(f'cdb={cdb}')
...
cdb = TorchBackend(dist_backend, timeout, init_method, rank, world_size)
```
3. 初始化TorchBackend的部分function
```python
# deepspeed.comm.torch.py
self.shm_comm_op = build_shm_op()
self.has_all_reduce_coalesced = has_all_reduce_coalesced()
self.has_coalescing_manager = has_coalescing_manager()
self.all_gather_function = self.get_all_gather_function()
self.reduce_scatter_function = self.get_reduce_scatter_function()
```
4. 初始化通信进程组
```python
self.init_process_group(backend, timeout, init_method, rank, world_size)
```
```
torch.distributed.init_process_group(backend,
                                        timeout=timeout,
                                        init_method=init_method,
                                        rank=rank,
                                        world_size=world_size)
```

<font color="red">megatron mpu是什么</font>
# 3. 


