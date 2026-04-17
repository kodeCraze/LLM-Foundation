# Week 4: Optimization Deep Dive ✅

**Focus:** The art of training well - optimizers, learning rates, and stability

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand why Adam beats SGD for transformers
- ✅ Know how to choose and tune learning rates
- ✅ Understand learning rate schedules (warmup + cosine decay)
- ✅ Know why gradient clipping is essential
- ✅ Have intuition for training dynamics

---

## Week 4 Scripts

### 1. `week4_optimizers.py`
**Purpose:** Compare optimizers and demonstrate LR/scheduling concepts

**What you'll learn:**
- SGD vs Momentum vs Adam vs AdamW
- Learning rate effects (too small, too big, just right)
- LR schedules (constant, step, exponential, cosine, warmup)
- Gradient clipping and exploding gradients

**Run it:**
```bash
cd learning_journey/week4_optimization
python week4_optimizers.py
```

**Output:**
- Console: Detailed optimizer explanations
- `optimizer_comparison.png`: SGD vs Adam performance
- `learning_rate_effects.png`: LR too small/big visualization
- `lr_schedules.png`: Common LR schedules
- `gradient_clipping.png`: Why clipping matters

---

## Key Concepts

### Optimizer Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZER EVOLUTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SGD (1986)                                                     │
│  ━━━━━━━━━━                                                     │
│  weight = weight - lr × gradient                               │
│  • Simple, well-understood                                      │
│  • Slow, zigzags, gets stuck                                   │
│                                                                 │
│  ↓                                                              │
│                                                                 │
│  SGD + Momentum                                               │
│  ━━━━━━━━━━━━━━━━━━                                             │
│  velocity = 0.9 × velocity + gradient                          │
│  weight = weight - lr × velocity                               │
│  • Faster (builds up speed)                                     │
│  • Escapes shallow valleys                                      │
│                                                                 │
│  ↓                                                              │
│                                                                 │
│  Adam (2014)                                                    │
│  ━━━━━━━━━━━━                                                   │
│  m = β₁ × m + (1-β₁) × gradient   ← momentum                   │
│  v = β₂ × v + (1-β₂) × gradient²  ← adaptive scaling           │
│  weight = weight - lr × m / √v                                │
│  • Adaptive per-parameter learning rates                       │
│  • Fast convergence, works out-of-box                          │
│                                                                 │
│  ↓                                                              │
│                                                                 │
│  AdamW (2017) ← DEFAULT FOR LLMs!                             │
│  ━━━━━━━━━━━━━━                                                 │
│  Adam + proper weight decay                                     │
│  • Better regularization                                        │
│  • Less overfitting                                             │
│  • What GPT-3, Llama, etc. use                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Your codebase already supports AdamW!**

See `training/model.py` lines 143-147:
```python
keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=0.005,
)
```

---

### Learning Rate: The Goldilocks Problem

```
Learning Rate Too Small (0.001)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loss: ████████████████████████████████████████
      Slow progress, might never converge
      Time wasted!

Learning Rate Just Right (0.1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loss: ████
      Steady, smooth descent
      Optimal!

Learning Rate Too Big (0.5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loss: █  █    ██  █     ███   █
      Unstable, oscillates or diverges
      Training fails!
```

**Typical values for transformers:**
- Small models (your demos): 1e-3 to 1e-4
- Medium models (GPT-2 size): 1e-4 to 5e-5
- Large models (GPT-3+): 1e-5 to 1e-6
- Rule of thumb: Larger model = smaller LR

---

### Learning Rate Schedules

Modern training almost always uses schedules:

```
Learning Rate
    │
    │     ╭─────╮
    │    ╱       ╲_______
    │   ╱                  ╲____
    │  ╱                          ╲
    │ ╱  Warmup    Cosine Decay      ╲
    │╱                                  ╲
    └──────────────────────────────────────────→ Epochs
       0    10%    50%    100%
```

**Components:**
1. **Warmup** (first ~10% of training):
   - Start with very small LR
   - Linearly increase to target LR
   - Prevents early instability

2. **Decay** (rest of training):
   - Cosine decay: smooth curve following cosine
   - Linear decay: straight line down
   - Step decay: reduce by half every N steps

**Why this works:**
- Early training: Large gradients, want small steps to stabilize
- Late training: Small gradients, want even smaller steps to fine-tune

---

### Gradient Clipping

**The Problem:** In deep transformers, gradients can explode:

```
Layer 1: gradient = 0.5
Layer 2: gradient = 0.8  
Layer 3: gradient = 1.2
Layer 4: gradient = 2.1
Layer 5: gradient = 8.7  ← Explosion!
Layer 6: gradient = 156.2
Layer 7: NaN  ← Training dies
```

**The Solution:** Clip gradients above threshold:

```python
if gradient_norm > threshold:
    gradient = gradient × (threshold / gradient_norm)
```

**Result:** Gradients bounded, stable training!

**In your code:**
```python
optimizer = keras.optimizers.Adam(
    clipnorm=1.0  # Clip if norm > 1.0
)
```

---

## Week 4 Exercises

### Exercise 1: Try Different Optimizers
```python
# In week4_optimizers.py, add more optimizers to test:
optimizers = [
    (keras.optimizers.SGD(learning_rate=0.01), "SGD"),
    (keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), 
     "SGD + Nesterov"),
    (keras.optimizers.RMSprop(learning_rate=0.001), "RMSprop"),
    (keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
     "Adam (tuned)"),
]

# Questions:
# 1. Which converges fastest?
# 2. Which is most stable?
# 3. Which achieves best final loss?
```

### Exercise 2: Find Your Model's Learning Rate
```python
# Run with different LRs and plot final accuracy
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# For each LR:
# 1. Train model
# 2. Record final loss and accuracy
# 3. Plot: LR vs Final Performance

# Find the "Goldilocks zone"
```

### Exercise 3: Implement Cosine Schedule
```python
class CosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with warmup."""
    
    def __init__(self, initial_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        # Warmup phase
        warmup_lr = self.initial_lr * step / self.warmup_steps
        
        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        step_in_decay = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * step_in_decay / decay_steps))
        decayed_lr = self.initial_lr * cosine_decay
        
        # Choose based on step
        return tf.cond(step < self.warmup_steps, 
                        lambda: warmup_lr, 
                        lambda: decayed_lr)

# Use it:
optimizer = keras.optimizers.Adam(learning_rate=CosineSchedule(1e-4, 1000, 10000))
```

### Exercise 4: Compare Clipped vs Unclipped
```python
# Train two identical models:
# Model A: clipnorm=None (no clipping)
# Model B: clipnorm=1.0 (clipped)

# Monitor:
# 1. Gradient norms during training
# 2. Loss stability
# 3. Final performance

# Deep transformers: Clipping is essential!
```

### Exercise 5: Study Your Training Code
```bash
# Read your training implementation
cat training/model.py | grep -A 10 "AdamW"
cat training/model.py | grep -A 5 "learning_rate"

# Questions:
# 1. What optimizer does create_model() use by default?
# 2. Is weight decay enabled?
# 3. How would you add a learning rate schedule?
# 4. How would you add gradient clipping?
```

---

## Connection to Modern LLMs

### GPT-3 Training Configuration (Public Info)
```yaml
optimizer: AdamW
beta_1: 0.9
beta_2: 0.95
learning_rate: 6e-5  # Very small for huge model
weight_decay: 0.1
lr_schedule: cosine with warmup
warmup_steps: 375M tokens
total_steps: 300B tokens
gradient_clipping: 1.0
```

### Llama 2 Training Configuration
```yaml
optimizer: AdamW
learning_rate: 1e-4 to 3e-4 (varies by model size)
lr_schedule: cosine with warmup
warmup_ratio: 0.01 (1% of total steps)
weight_decay: 0.1
gradient_clipping: 1.0
```

**Pattern:** All use AdamW + cosine schedule + warmup + clipping!

---

## Week 4 Completion Checklist

- [ ] Ran `week4_optimizers.py` and saw optimizer comparison
- [ ] Understand why Adam beats SGD for this task
- [ ] Visualized LR effects and found "Goldilocks zone"
- [ ] Saw different LR schedules and understand warmup+decay
- [ ] Understand gradient clipping importance
- [ ] Know that AdamW is standard for modern LLMs
- [ ] Can explain when to use each optimizer
- [ ] Completed at least 1 exercise

---

## Key Takeaways

### Optimizers
1. **SGD:** Simple but slow
2. **Momentum:** Faster, builds up speed
3. **Adam:** Adaptive per-parameter, fast convergence
4. **AdamW:** Adam + proper regularization ← **USE THIS**

### Learning Rates
1. **Too small:** Never converges, wastes time
2. **Too big:** Unstable, diverges
3. **Just right:** Smooth descent to optimum
4. **Typical:** 1e-5 to 1e-3 depending on model size

### Schedules
1. **Constant:** Simple, often suboptimal
2. **Step decay:** Halve every N epochs
3. **Cosine:** Smooth, modern standard
4. **Warmup + Cosine:** Best practice for LLMs

### Stability
1. **Gradient clipping:** Essential for deep transformers
2. **Threshold:** Usually clipnorm=1.0
3. **Prevents:** Exploding gradients, NaN losses

---

## Bridge to Week 5

Next week: **Scale & Compute Efficiency**
- Mixed precision training (FP16/BF16)
- Gradient accumulation (simulate large batches)
- Memory profiling
- Multi-GPU training basics

---

## Resources

### Reading
- "Adam: A Method for Stochastic Optimization" (original paper)
- "Decoupled Weight Decay Regularization" (AdamW paper)
- "One Cycle Policy" (super-convergence)

### Interactive
- [LR Finder Explanation](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
- Experiment with our demo code!

### Code Study
- `training/model.py`: Your optimizer setup
- [Keras Optimizers Docs](https://keras.io/api/optimizers/)

---

## Week 4 Status

**Your progress:**
- ✅ Optimizer comparison (SGD, Momentum, Adam, AdamW)
- ✅ Learning rate effects visualization
- ✅ LR schedules (constant, step, exponential, cosine, warmup)
- ✅ Gradient clipping demonstration
- ✅ AdamW identified as modern standard
- ✅ Connection to GPT-3/Llama training configs

**Ready for Week 5?**

If yes → See `../week5_scale/`

If no → Experiment with optimizer hyperparameters in the demo!

---

*Good optimizers don't just find the minimum—they find it fast and reliably.*
