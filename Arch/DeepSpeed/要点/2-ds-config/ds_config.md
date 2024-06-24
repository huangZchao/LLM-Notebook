[link](https://www.deepspeed.ai/docs/config-json/)

1. train_batch_size
   - train_micro_batch_size_per_gpu * gradient_accumulation_steps * the number of GPUs

2. optimizer
   - Adam, AdamW, OneBitAdam, Lamb, and OneBitLamb，ZeroOneAdam，OneBitLamb

3. scheduler

4. Communication options
   - communication_data_type, prescale_gradients, gradient_predivide_factor

5. FP16 training options

6. BF16

7. Automatic mixed precision (AMP) training options

8. Gradient ClippingPermalink
   - [torch.nn.utils.clip_grad_norm](https://blog.csdn.net/m0_46412065/article/details/131396098)

9. ZeRO Optimizations for FP16 TrainingPermalink
   - 