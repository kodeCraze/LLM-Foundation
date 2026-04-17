# Week 5: Memory & Compute Efficiency ✅

**Focus:** Training large models efficiently - memory profiling, gradient accumulation, mixed precision

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand where GPU memory goes during training
- ✅ Know how to use gradient accumulation for large batches
- ✅ Understand mixed precision (FP16/BF16) benefits
- ✅ Know how batch size affects training dynamics
- ✅ Be ready to train larger models efficiently

---

## Week 5 Scripts

### 1. `week5_memory_and_scaling.py`
**Purpose:** Memory profiling and scaling techniques

**What you'll learn:**
- Memory breakdown (model vs activations vs gradients vs optimizer)
- Model size comparison (Tiny → XL)
- Gradient accumulation for large batch simulation
- Mixed precision training (FP16/BF16)
- Batch size effects on convergence

**Run it:**
```bash
cd learning_journey/week5_scale
python week5_memory_and_scaling.py
```

**Output:**
- Console: Detailed memory analysis
- `gradient_accumulation.png`: Accumulation vs full batch
- `mixed_precision.png`: FP32 vs FP16/BF16 comparison
- `batch_size_effects.png`: Training dynamics vs batch size

---

## Key Concepts

### GPU Memory Breakdown

When training a model, memory isn't just for weights!

```
Total Training Memory
├─ Model Weights (~25%)
│  └─ Parameters stored as FP32 (4 bytes each)
│
├─ Activations (~25%)
│  └─ Forward pass outputs, needed for backward
│
├─ Gradients (~25%)
│  └─ ∂Loss/∂Weight for each parameter
│
└─ Optimizer States (~25%)
   └─ Adam: 2× model size (momentum + variance)

Total = 4× Model Size (roughly!)
```

**Example - Your 53K parameter model:**
- Model: 210 KB
- Training: ~840 KB

**Example - GPT-3 (175B parameters):**
- Model: 700 GB
- Training: 2.8+ TB

---

### Gradient Accumulation

**Problem:** Want batch_size=1024 but GPU only fits 32

**Solution:** Accumulate gradients over multiple steps

```python
# Normal training
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Update after each batch
    optimizer.zero_grad()

# With gradient accumulation
accumulation_steps = 32
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # Scale loss
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update after accumulation
        optimizer.zero_grad()
```

**Result:**
- Effective batch size: 32 × 32 = 1024 ✅
- Memory used: 32 (not 1024!) ✅
- Trade-off: 32× more time per update

---

### Mixed Precision Training

**Formats:**

| Format | Bits | Bytes | Range | Precision | Use Case |
|--------|------|-------|-------|-----------|----------|
| FP32 | 32 | 4 | Wide | High | Master weights |
| FP16 | 16 | 2 | Narrow | Medium | Forward/backward |
| BF16 | 16 | 2 | Wide | Low | Forward/backward |

**Strategy:**
```
Forward pass:     FP16/BF16 (fast, efficient)
Loss calculation: FP32 (accurate)
Backward pass:    FP16/BF16 (fast)
Gradients:        FP16 → FP32 (accumulate in FP32)
Weight update:    FP32 (master weights always FP32)
```

**Benefits:**
- 50% memory savings
- 2-3× speedup on modern GPUs (tensor cores)

**In code:**
```python
# Keras
keras.mixed_precision.set_global_policy('mixed_float16')

# PyTorch
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### Batch Size Effects

| Batch Size | Gradient Noise | Memory | Speed | Generalization |
|------------|----------------|--------|-------|----------------|
| Small (4-8) | High | Low | Slow | Often better |
| Medium (32-128) | Medium | Medium | Good | Good |
| Large (512+) | Low | High | Fast | May overfit |

**Linear Scaling Rule:**
```
If you increase batch size by N, increase learning rate by N
```

**Example:**
- Batch 32, LR = 1e-4 works well
- Batch 256, LR = 8e-4 should work similarly

**But:** Large batches need warmup! Start with small LR, scale up.

---

## Week 5 Exercises

### Exercise 1: Profile Your Model
```python
# In week5_memory_and_scaling.py, modify:
profiler.calculate_model_memory(
    vocab_size=YOUR_VOCAB_SIZE,
    max_length=YOUR_MAX_LENGTH,
    embedding_dim=YOUR_EMBED_DIM,
    num_blocks=YOUR_BLOCKS,
    num_heads=YOUR_HEADS
)

# Questions:
# 1. What's the largest model that fits in your GPU?
# 2. How much memory do activations consume?
# 3. What happens if you double batch size?
```

### Exercise 2: Implement Gradient Accumulation
```python
# Modify week3_training_loop.py to use accumulation

class AccumulationTrainer:
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def train_step(self, batch):
        loss = self.model(batch) / self.accumulation_steps
        loss.backward()
        
        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()

# Compare:
# - Without accumulation: batch=4
# - With accumulation: batch=4, steps=4 (effective=16)
# Measure memory and convergence speed
```

### Exercise 3: Try Mixed Precision
```python
import keras

# Enable mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Build and train your model
model = create_model(...)
model.compile(...)

# Compare:
# - Memory usage (should be ~50%)
# - Training speed (should be 2-3× faster)
# - Final accuracy (should be similar)
```

### Exercise 4: Batch Size Sweep
```python
batch_sizes = [4, 8, 16, 32, 64]
results = []

for bs in batch_sizes:
    # Adjust LR: lr = base_lr * (bs / base_bs)
    lr = 0.001 * (bs / 32)
    
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=lr), ...)
    
    history = model.fit(..., batch_size=bs, epochs=20)
    
    results.append({
        'batch_size': bs,
        'final_loss': history.history['loss'][-1],
        'time_per_epoch': measure_time()
    })

# Plot: batch_size vs convergence speed
# Find your optimal batch size
```

### Exercise 5: Calculate GPT-3 Memory
```python
# GPT-3 Large: 175B parameters
# Config:
#   - vocab_size: 50,257
#   - max_length: 2048
#   - embedding_dim: 12,288 (across 96 layers)
#   - num_heads: 96
#   - batch_size: 3.2M tokens (massive!)

# Calculate:
# 1. Model size in GB
# 2. Training memory requirements
# 3. Number of GPUs needed (assuming 80GB A100s)

model_params = 175_000_000_000
model_gb = model_params * 4 / (1024**3)
training_gb = model_gb * 4  # Rough estimate

print(f"GPT-3 model: {model_gb:.1f} GB")
print(f"Training: ~{training_gb:.1f} GB")
print(f"GPUs needed: {training_gb / 80:.0f}× A100 80GB")
```

---

## Connection to Modern LLMs

### GPT-3 Training Scale
```yaml
Model: GPT-3 175B
Parameters: 175,000,000,000
Model size: 700 GB (FP32)
Training memory: ~2.8 TB
Hardware: 10,000+ V100 GPUs
Training time: Months
Batch size: 3.2M tokens
Gradient accumulation: Yes
Mixed precision: Yes (FP16)
```

### Memory Optimization Techniques

| Technique | Memory Savings | Speed Impact |
|-----------|----------------|--------------|
| Mixed precision | 50% | +100-200% |
| Gradient checkpointing | 30-50% | -20% |
| Gradient accumulation | Variable | -time |
| ZeRO (DeepSpeed) | 8× | Minimal |
| Model parallelism | Scales to any size | Communication |

---

## Week 5 Completion Checklist

- [ ] Ran `week5_memory_and_scaling.py`
- [ ] Understand the 4 components of training memory
- [ ] Can calculate memory for any model size
- [ ] Understand gradient accumulation concept
- [ ] Know FP16 vs BF16 trade-offs
- [ ] Understand batch size effects
- [ ] Completed at least 1 exercise

---

## Key Takeaways

### Memory
1. **Training uses 4× model size** (weights + activations + grads + optimizer)
2. **Activations are the surprise** - often as large as model weights
3. **Adam is expensive** - 2× model size just for optimizer states

### Gradient Accumulation
1. **Simulate large batches** with small memory
2. **Effective batch = per_step × num_steps**
3. **Trade-off: memory vs time**

### Mixed Precision
1. **50% memory, 2-3× speed** on tensor cores
2. **BF16 > FP16** (better range, no scaling needed)
3. **Keep master weights FP32** for accuracy

### Batch Size
1. **Small batches:** noisy but generalize better
2. **Large batches:** stable but may overfit
3. **Scale LR with batch size** (linear scaling rule)

---

## Bridge to Week 6

Next week: **Distributed Training**
- Data parallelism (multiple GPUs, same model)
- Model parallelism (split model across GPUs)
- Pipeline parallelism (layer-wise distribution)
- ZeRO and FSDP optimizations

---

## Resources

### Reading
- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- "Mixed Precision Training" (NVIDIA)
- "Training Tips for the Transformer Model"

### Tools
- **DeepSpeed**: Microsoft's training optimization library
- **FSDP**: PyTorch's Fully Sharded Data Parallel
- **Accelerate**: Hugging Face's distributed training wrapper

### Code Study
- Your model sizes in `training/model.py`
- How to enable mixed precision in Keras
- Gradient accumulation implementation

---

## Week 5 Status

**Your progress:**
- ✅ Memory breakdown (weights, activations, grads, optimizer)
- ✅ Model size calculations (Tiny → GPT-3 scale)
- ✅ Gradient accumulation concept
- ✅ Mixed precision (FP16/BF16) benefits
- ✅ Batch size effects on training

**Ready for Week 6?**

If yes → See `../week6_distributed/`

If no → Calculate memory for your dream model size!

---

*Training large models is 50% algorithms, 50% systems engineering.*
