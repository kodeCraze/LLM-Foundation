# Week 6: Distributed Training & Model Parallelism ✅

**Focus:** Training across multiple GPUs - data, model, and pipeline parallelism

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand data parallelism (most common approach)
- ✅ Know when to use model vs pipeline parallelism
- ✅ Understand ZeRO optimizer for memory efficiency
- ✅ Know scaling efficiency limits
- ✅ Be ready to train models that don't fit on single GPU

---

## Week 6 Scripts

### 1. `week6_distributed_training.py`
**Purpose:** Conceptual understanding of distributed training

**What you'll learn:**
- Data parallelism mechanics
- Model parallelism for oversized models
- Pipeline parallelism for better utilization
- ZeRO optimizer (8× memory reduction!)
- Scaling efficiency realities

**Run it:**
```bash
cd learning_journey/week6_distributed
python week6_distributed_training.py
```

**Output:**
- Console: Detailed explanations with ASCII diagrams
- `scaling_efficiency.png`: Speedup vs number of GPUs

---

## Key Concepts

### Data Parallelism (DP)

**The Simplest Approach:**
```
┌─────────────────────────────────────────────────────┐
│               DATA PARALLELISM                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Each GPU has: FULL MODEL                           │
│  Each GPU processes: DIFFERENT BATCH                │
│                                                     │
│  GPU 0: Forward → Backward → Gradients              │
│  GPU 1: Forward → Backward → Gradients              │
│  GPU 2: Forward → Backward → Gradients              │
│         ↓                                           │
│  All-Reduce: Average gradients                      │
│         ↓                                           │
│  All GPUs: Update model (same update!)              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Characteristics:**
- ✅ Simple to implement
- ✅ Near-linear speedup (2 GPUs ≈ 2× faster)
- ❌ Model must fit on single GPU
- ❌ Communication overhead

**When to use:** Model fits on single GPU, want faster training

---

### Model Parallelism (MP)

**When Model Is Too Big:**
```
┌─────────────────────────────────────────────────────┐
│              MODEL PARALLELISM                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  GPT-3 175B = 700GB                                 │
│  GPU memory = 80GB                                  │
│  Problem: Model doesn't fit!                        │
│                                                     │
│  Solution: Split across GPUs                        │
│                                                     │
│  GPU 0: Embedding + Layers 0-7                      │
│         ↓ (pass activations)                        │
│  GPU 1: Layers 8-15                                 │
│         ↓                                           │
│  GPU 2: Layers 16-23                                │
│         ↓                                           │
│  GPU 3: Layers 24-31 + Output                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Characteristics:**
- ✅ Can train arbitrarily large models
- ❌ Poor GPU utilization (only 1 GPU active at a time)
- ❌ Complex to implement
- ❌ High communication overhead

**When to use:** Model doesn't fit on single GPU

---

### Pipeline Parallelism (PP)

**Fixing Model Parallelism's Low Utilization:**
```
┌─────────────────────────────────────────────────────┐
│             PIPELINE PARALLELISM                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Time →                                             │
│                                                     │
│  GPU 0: [F0][F1][F2][F3][B3][B2][B1][B0]           │
│  GPU 1: __[F0][F1][F2][F3][B3][B2][B1][B0]         │
│  GPU 2: ____ [F0][F1][F2][F3][B3][B2][B1][B0]      │
│  GPU 3: ______ [F0][F1][F2][F3][B3][B2][B1][B0]    │
│                                                     │
│  F = Forward on micro-batch                         │
│  B = Backward on micro-batch                        │
│                                                     │
│  Result: All GPUs active!                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**The "Bubble":**
- At start/end of batch, some GPUs are idle
- More micro-batches = smaller bubble
- With 32 micro-batches: bubble is only 12.5%

**Characteristics:**
- ✅ Better GPU utilization than MP
- ✅ Can train large models efficiently
- ❌ Still complex to implement
- ❌ "Bubble" overhead

---

### ZeRO: Memory Optimization

**The Problem:**
```
With 4 GPUs and Adam:
  Parameters: 10 GB
  Gradients: 10 GB  
  Optimizer: 20 GB (2× for Adam momentum/variance)
  ─────────────────────
  Per GPU: 40 GB
  Total: 160 GB (wasteful! Each GPU has identical copy)
```

**ZeRO Solution:** Partition optimizer states

| Stage | What Gets Partitioned | Memory Reduction |
|-------|----------------------|------------------|
| ZeRO-1 | Optimizer states only | 37.5% |
| ZeRO-2 | + Gradients | 56.3% |
| ZeRO-3 | + Parameters | 75% |

**ZeRO-3 Result:**
```
Same 4 GPUs:
  Per GPU: 10 GB (1/4 of model)
  Total: 40 GB (not 160 GB!)
  
  Can train 4× larger models!
```

**Implementations:**
- DeepSpeed (Microsoft)
- FSDP (PyTorch)
- Both give ~8× memory savings!

---

### Scaling Efficiency

**The Dream:** Linear speedup
- 2 GPUs = 2× faster
- 4 GPUs = 4× faster
- 8 GPUs = 8× faster

**The Reality:** Communication overhead

| GPUs | Ideal Speedup | Real Speedup | Efficiency |
|------|---------------|--------------|------------|
| 2 | 2× | 1.9× | 95% |
| 4 | 4× | 3.6× | 90% |
| 8 | 8× | 6.8× | 85% |
| 16 | 16× | 12.8× | 80% |
| 32 | 32× | 24× | 75% |
| 128 | 128× | 76× | 59% |

**Key insight:** Efficiency drops as you add more GPUs. Network bandwidth is the bottleneck!

---

## Week 6 Exercises

### Exercise 1: Simulate Data Parallelism
```python
# Simulate 4-GPU data parallel training

def simulate_data_parallel(batch_size, num_gpus):
    """Simulate DP training."""
    # Each GPU gets batch_size/num_gpus
    per_gpu_batch = batch_size // num_gpus
    
    # Simulate forward pass time
    forward_time = per_gpu_batch * 0.01  # 10ms per sample
    
    # Simulate backward pass time
    backward_time = per_gpu_batch * 0.02  # 20ms per sample
    
    # All-reduce time (communication)
    allreduce_time = 0.05 * num_gpus  # Scales with GPUs
    
    total_time = forward_time + backward_time + allreduce_time
    
    speedup = (batch_size * 0.03) / total_time
    
    return speedup

# Test with different GPU counts
for gpus in [1, 2, 4, 8]:
    speedup = simulate_data_parallel(128, gpus)
    print(f"{gpus} GPUs: {speedup:.2f}× speedup ({speedup/gpus*100:.0f}% efficient)")
```

### Exercise 2: Calculate ZeRO Savings
```python
def calculate_zero_memory(model_size_gb, num_gpus, zero_stage):
    """Calculate memory with ZeRO."""
    # Base memory (no ZeRO)
    params = model_size_gb
    grads = model_size_gb
    optimizer = 2 * model_size_gb  # Adam
    base_per_gpu = params + grads + optimizer
    
    # With ZeRO
    if zero_stage == 0:
        # No ZeRO
        per_gpu = base_per_gpu
    elif zero_stage == 1:
        # ZeRO-1: Partition optimizer
        per_gpu = params + grads + (optimizer / num_gpus)
    elif zero_stage == 2:
        # ZeRO-2: + Partition gradients
        per_gpu = params + (grads / num_gpus) + (optimizer / num_gpus)
    elif zero_stage == 3:
        # ZeRO-3: + Partition parameters
        per_gpu = (params / num_gpus) + (grads / num_gpus) + (optimizer / num_gpus)
    
    total = per_gpu * num_gpus
    savings = (base_per_gpu * num_gpus - total) / (base_per_gpu * num_gpus)
    
    return per_gpu, total, savings

# Calculate for GPT-3 scale model (700GB)
model_gb = 700
gpus = 64

for stage in [0, 1, 2, 3]:
    per_gpu, total, savings = calculate_zero_memory(model_gb, gpus, stage)
    print(f"ZeRO-{stage}: {per_gpu:.0f}GB/GPU, Total: {total:.0f}GB, Savings: {savings*100:.1f}%")
```

### Exercise 3: Research Distributed Frameworks
```bash
# Look up these frameworks:
# 1. DeepSpeed (Microsoft)
# 2. FSDP (PyTorch)
# 3. Horovod (Uber)
# 4. Megatron-LM (NVIDIA)

# Questions:
# 1. Which does your codebase use?
# 2. What parallelism strategies does each support?
# 3. Which is easiest to use?
```

### Exercise 4: Design Training Strategy
```python
"""
Scenario: Train 10B parameter model
Available: 8 GPUs, 32GB each
Model: 40GB (FP32)
Batch size: Need effective 256

Design strategy:
"""

# Questions:
# 1. Can we use pure data parallelism?
# 2. If not, what mix of strategies?
# 3. How many micro-batches for pipeline?
# 4. Should we use ZeRO? Which stage?

# Solution sketch:
strategy = {
    'data_parallel': 2,      # 2 replicas
    'pipeline_parallel': 4,  # 4 stages per replica
    'total_gpus': 2 * 4,     # 8 GPUs
    'zero_stage': 2,         # Partition gradients + optimizer
}

print(f"Strategy: {strategy}")
```

---

## Connection to Modern LLMs

### GPT-3 Training (175B parameters)
```yaml
Hardware: 10,000 V100 GPUs
Strategy: 
  - Data parallelism: 10,000 / (pipeline × tensor) replicas
  - Model parallelism: Split layers across GPUs
  - Pipeline parallelism: Micro-batches
  - Mixed precision: FP16

Memory per GPU: ~8GB (with optimization)
Total memory: 80TB (!)
Training time: Months
```

### Scaling Laws
- More GPUs = faster training (but with diminishing returns)
- Larger batches need linear LR scaling
- Communication is the bottleneck at scale

---

## Week 6 Completion Checklist

- [ ] Ran `week6_distributed_training.py`
- [ ] Can explain data parallelism
- [ ] Know when to use model vs pipeline parallelism
- [ ] Understand ZeRO stages and memory savings
- [ ] Know scaling efficiency limits
- [ ] Can design training strategy for large models
- [ ] Completed at least 1 exercise

---

## Key Takeaways

### Parallelism Types
1. **Data Parallelism:** Simple, most common, model must fit
2. **Model Parallelism:** For oversized models, poor utilization
3. **Pipeline Parallelism:** Better utilization, micro-batching

### ZeRO
1. **ZeRO-1:** Partition optimizer states (37.5% savings)
2. **ZeRO-2:** + Partition gradients (56.3% savings)
3. **ZeRO-3:** + Partition parameters (75% savings)
4. **Result:** Can train 4-8× larger models!

### Scaling
1. **2-8 GPUs:** >90% efficient, great for most use cases
2. **32+ GPUs:** Need careful optimization, communication overhead
3. **Network bandwidth** is the bottleneck

### Modern Training
1. **3D Parallelism:** DP + PP + TP combined
2. **GPT-3:** 10,000 GPUs, months of training
3. **ZeRO/FSDP:** Essential for models > 1B parameters

---

## Bridge to Week 7

Next week: **Advanced Architecture**
- Multi-Query Attention (MQA)
- Grouped Query Attention (GQA)
- RoPE (Rotary Position Embeddings)
- Pre-norm vs Post-norm
- Modern architectural choices

---

## Resources

### Reading
- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- "Megatron-LM: Training Multi-Billion Parameter Language Models"
- "Efficient Large-Scale Language Model Training on GPU Clusters"

### Tools
- **DeepSpeed**: Microsoft's distributed training framework
- **FSDP**: PyTorch's native solution
- **Horovod**: Uber's distributed training framework

### Code Study
- Look for distributed training in your codebase
- Check if ZeRO/FSDP is configured

---

## Week 6 Status

**Your progress:**
- ✅ Data parallelism understanding
- ✅ Model vs pipeline parallelism
- ✅ ZeRO memory optimization
- ✅ Scaling efficiency realities
- ✅ Distributed training strategy

**Ready for Week 7?**

If yes → See `../week7_advanced_architecture/`

If no → Research DeepSpeed or FSDP documentation!

---

*Training at scale is as much about systems engineering as it is about ML.*
