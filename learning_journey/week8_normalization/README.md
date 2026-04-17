# Week 8: Normalization & Architecture Stability ✅

**Focus:** RMSNorm vs LayerNorm, Pre-norm vs Post-norm, training stability techniques

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand RMSNorm and why it's faster than LayerNorm
- ✅ Know the difference between Pre-norm and Post-norm
- ✅ Understand why Pre-norm enables deeper networks
- ✅ Know training stability techniques
- ✅ See how to modernize your architecture

---

## Week 8 Scripts

### 1. `week8_normalization.py`
**Purpose:** Compare normalization techniques and architectures

**What you'll learn:**
- LayerNorm vs RMSNorm (computational differences)
- Pre-norm vs Post-norm (stability differences)
- Gradient flow visualization
- Why modern models use Pre-norm + RMSNorm

**Run it:**
```bash
cd learning_journey/week8_normalization
python week8_normalization.py
```

**Output:**
- Console: Detailed comparisons
- `normalization_comparison.png`: LayerNorm vs RMSNorm visualization
- `gradient_flow.png`: Pre-norm vs Post-norm gradient stability
- `training_stability.png`: Impact of architecture choices

---

## Key Concepts

### LayerNorm vs RMSNorm

```python
# LayerNorm (Original Transformer)
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
x_norm = (x - mean) / sqrt(var + eps) * gamma + beta
# Operations: mean, var, sub, div, scale, shift
# Parameters: 2 * dim

# RMSNorm (Modern Standard)
rms = sqrt(mean(x^2) + eps)
x_norm = x / rms * gamma
# Operations: rms, div, scale
# Parameters: dim (no beta!)
```

**Why RMSNorm is faster:**
1. **No mean subtraction** (saves computation)
2. **No beta parameter** (fewer parameters)
3. **~30-40% speedup** in practice
4. **Similar quality** to LayerNorm

**Adoption:**
- 2019: LayerNorm (BERT, GPT-2, T5)
- 2023+: RMSNorm (Llama, Mistral, Gemma)

---

### Pre-norm vs Post-norm

**Post-norm (Original Transformer):**
```
x = x + Sublayer(Norm(x))
```
```
Input
  ↓
[Norm]         ← Normalize first
  ↓
[Attention]    ← Then apply attention
  ↓
+ Residual     ← Add skip connection
  ↓
Output
```

**Pre-norm (Modern Standard):**
```
x = x + Sublayer(x)
x = Norm(x)
```
```
Input
  ↓
[Attention]    ← Apply attention first
  ↓
+ Residual     ← Add skip connection
  ↓
[Norm]         ← Then normalize
  ↓
Output
```

**Why Pre-norm is better:**

| Aspect | Post-norm | Pre-norm |
|--------|-----------|----------|
| Gradient flow | Through Norm (unstable) | Direct residual (stable) |
| Deep networks | Struggles >12 layers | Works with 100+ layers |
| Warmup | Critical | More forgiving |
| Stability | Can explode | Much more stable |

**Visualization:**
```
Post-norm gradients:  [1.0, 0.5, 2.5, 0.3, 4.0, 0.1] (wild swings)
Pre-norm gradients:   [1.0, 0.9, 1.1, 0.8, 1.2, 0.9] (stable near 1.0)
```

---

### The Modern Architecture Stack (2024)

```
┌────────────────────────────────────────────────────────────┐
│           MODERN TRANSFORMER (Llama 3, Mistral)              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input                                                     │
│    ↓                                                       │
│  Token Embedding                                           │
│    ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Transformer Block × N (N=32 for Llama 3 8B)           │ │
│  │                                                       │ │
│  │  ┌────────────────────────────────────────────────┐  │ │
│  │  │ 1. GQA Attention with RoPE                      │  │ │
│  │  │    • Grouped Query Attention                     │  │ │
│  │  │    • Rotary Position Embeddings                │  │ │
│  │  │                                                   │  │ │
│  │  │ 2. Add Residual                                  │  │ │
│  │  │    • x = x + attention_output                   │  │ │
│  │  │                                                   │  │ │
│  │  │ 3. RMSNorm (PRE-NORM!)                           │  │ │
│  │  │    • x = x / RMS(x) * gamma                     │  │ │
│  │  └────────────────────────────────────────────────┘  │ │
│  │    ↓                                                  │ │
│  │  ┌────────────────────────────────────────────────┐  │ │
│  │  │ 4. SwiGLU Feedforward                           │  │ │
│  │  │    • Swish activation + Gating                   │  │ │
│  │  │    • Expansion: 8/3 × embed_dim                  │  │ │
│  │  │                                                   │  │ │
│  │  │ 5. Add Residual                                  │  │ │
│  │  │    • x = x + ffn_output                         │  │ │
│  │  │                                                   │  │ │
│  │  │ 6. RMSNorm (PRE-NORM!)                          │  │ │
│  │  └────────────────────────────────────────────────┘  │ │
│  │                                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│    ↓                                                       │
│  Output Projection                                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**vs GPT-3 (2020):**
- ❌ MHA → ✅ GQA
- ❌ Learned position → ✅ RoPE
- ❌ LayerNorm → ✅ RMSNorm
- ❌ Post-norm → ✅ Pre-norm
- ❌ GELU → ✅ SwiGLU

---

## Week 8 Exercises

### Exercise 1: Implement RMSNorm
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdims=True) + self.eps)
        
        # Normalize and scale
        x_norm = x / rms * self.gamma
        
        return x_norm

# Compare with PyTorch LayerNorm
import time

dim = 1024
batch = 32
seq = 512

# Create inputs
x = torch.randn(batch, seq, dim)

# RMSNorm
rms_norm = RMSNorm(dim)
start = time.time()
for _ in range(100):
    out = rms_norm(x)
rms_time = time.time() - start

# LayerNorm
layer_norm = nn.LayerNorm(dim)
start = time.time()
for _ in range(100):
    out = layer_norm(x)
ln_time = time.time() - start

print(f"RMSNorm: {rms_time:.3f}s")
print(f"LayerNorm: {ln_time:.3f}s")
print(f"Speedup: {ln_time/rms_time:.2f}×")
```

### Exercise 2: Switch to Pre-norm
```python
# In your layers.py, modify TransformerBlock:

class TransformerBlock(keras.layers.Layer):
    """Pre-norm transformer block."""
    
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        
        # Use RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(embed_dim)  # Was: LayerNormalization()
        self.norm2 = RMSNorm(embed_dim)  # Was: LayerNormalization()
        
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, mlp_dim)
    
    def call(self, x):
        # PRE-NORM ARCHITECTURE
        # Attention with residual
        x = x + self.attn(self.norm1(x))  # Norm BEFORE attention
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))   # Norm BEFORE FFN
        
        return x

# OLD (Post-norm):
# attn_out = self.attn(x)
# x = self.norm1(x + attn_out)  # Norm AFTER

# NEW (Pre-norm):
# x = x + self.attn(self.norm1(x))  # Norm BEFORE
```

### Exercise 3: Train with Both Architectures
```python
# Train identical models with Post-norm vs Pre-norm
# Measure:
# 1. Convergence speed
# 2. Final loss
# 3. Gradient norms throughout training
# 4. Whether training explodes without warmup

configs = [
    {'norm_type': 'LayerNorm', 'norm_place': 'post'},
    {'norm_type': 'LayerNorm', 'norm_place': 'pre'},
    {'norm_type': 'RMSNorm', 'norm_place': 'pre'},
]

for config in configs:
    print(f"\nTesting: {config}")
    model = build_model(config)
    history = train(model, epochs=100)
    
    # Check for:
    # - Did it converge?
    # - Any loss spikes?
    # - Final accuracy?
```

### Exercise 4: Upgrade Your Codebase
```python
# 1. Add RMSNorm class to transformers/layers.py

# 2. Modify TokenAndPositionEmbedding to use RMSNorm
# (if you have any norm there)

# 3. Modify TransformerBlock:
#    - Replace LayerNorm with RMSNorm
#    - Switch from Post-norm to Pre-norm

# 4. Test:
#    - Does model still train?
#    - Is it faster?
#    - Is loss curve smoother?
```

---

## Connection to Modern LLMs

### Model Architecture Evolution

```
2019 (BERT/GPT-2):
  - LayerNorm
  - Post-norm
  - GELU

2020 (GPT-3):
  - LayerNorm
  - Pre-norm  ← Changed!
  - GELU

2023 (Llama 1/2, Mistral):
  - RMSNorm   ← Changed!
  - Pre-norm
  - SwiGLU    ← Changed!
  - GQA       ← Changed!
  - RoPE      ← Changed!

2024 (Llama 3, Gemma):
  - RMSNorm
  - Pre-norm
  - SwiGLU
  - GQA/MQA
  - RoPE
  - (Stabilized, optimized)
```

**Why the changes?**
1. **Pre-norm:** Training stability (enables deeper models)
2. **RMSNorm:** Speed (30-40% faster)
3. **SwiGLU:** Better activation function
4. **GQA:** Memory efficiency
5. **RoPE:** Better position encoding

---

## Week 8 Completion Checklist

- [ ] Ran `week8_normalization.py`
- [ ] Understand LayerNorm vs RMSNorm formulas
- [ ] Know why RMSNorm is faster (no mean subtraction)
- [ ] Understand Pre-norm vs Post-norm
- [ ] Know why Pre-norm is more stable
- [ ] Can name modern architecture choices
- [ ] Completed at least 1 exercise

---

## Key Takeaways

### Normalization
1. **LayerNorm:** `(x - mean) / std * gamma + beta`
   - Centers and scales
   - 2× parameters
   - Slower

2. **RMSNorm:** `x / RMS * gamma`
   - Just scales
   - 1× parameters
   - ~30-40% faster
   - **Modern choice**

### Architecture
1. **Post-norm:** `Norm → Sublayer → Residual` (unstable)
2. **Pre-norm:** `Sublayer → Residual → Norm` (stable)
3. **Pre-norm enables:** Training 100+ layer models

### Modern Stack (2024)
1. **RMSNorm** (faster)
2. **Pre-norm** (stable)
3. **SwiGLU** (better activation)
4. **GQA** (memory efficient)
5. **RoPE** (good extrapolation)

### Your Upgrade Path
1. Replace LayerNorm with RMSNorm
2. Switch from Post-norm to Pre-norm
3. Add gradient clipping
4. Use learning rate warmup

---

## Bridge to Week 9

Next week: **Modern Training Recipes**
- Chinchilla scaling laws
- Training compute-optimal models
- How to choose model size vs training tokens

---

## Resources

### Reading
- "Root Mean Square Layer Normalization" (original RMSNorm paper)
- "On Layer Normalization in the Transformer Architecture" (Pre-norm analysis)
- "GLU Variants Improve Transformer" (SwiGLU)

### Papers
- Llama 2 paper (architecture details)
- Mistral paper (efficient architecture)

### Code Study
- Your `layers.py` normalization
- Compare with Hugging Face implementations
- Check if Pre-norm or Post-norm

---

## Week 8 Status

**Your progress:**
- ✅ LayerNorm vs RMSNorm understanding
- ✅ Pre-norm vs Post-norm differences
- ✅ Gradient flow visualization
- ✅ Modern architecture evolution
- ✅ Upgrade path for your codebase

**Ready for Week 9?**

If yes → See `../week9_training_recipes/`

If no → Implement RMSNorm and Pre-norm in your codebase!

---

*Small architectural changes → Big stability improvements.*
