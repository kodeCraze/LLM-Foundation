# Week 7: Advanced Architecture - MQA, GQA, RoPE ✅

**Focus:** Modern attention mechanisms and position encodings used in 2024-2025 models

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand Multi-Query Attention (MQA) and its benefits
- ✅ Know when to use Grouped Query Attention (GQA) vs MQA vs MHA
- ✅ Understand RoPE (Rotary Position Embeddings)
- ✅ Know why modern models use these techniques
- ✅ See how to upgrade your codebase

---

## Week 7 Scripts

### 1. `week7_mqa_gqa_rope.py`
**Purpose:** Understand modern attention and position encoding

**What you'll learn:**
- MQA: 8× memory savings by sharing K, V
- GQA: Middle ground between MHA and MQA
- RoPE: Relative position encoding with extrapolation
- Why 2024 models (Llama, Mistral, Gemma) use these

**Run it:**
```bash
cd learning_journey/week7_advanced_architecture
python week7_mqa_gqa_rope.py
```

**Output:**
- Console: Detailed comparisons
- `attention_comparison.png`: MHA vs GQA vs MQA comparison
- `rope_visualization.png`: RoPE mechanism visualized

---

## Key Concepts

### The Evolution of Attention

```
2020: Multi-Head Attention (MHA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Each head has separate Q, K, V
• High quality
• High memory (8× KV cache per head)
• Used in: Original Transformer, GPT-3

2022: Multi-Query Attention (MQA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• All heads share single K, V
• 8× memory reduction!
• Fastest inference
• Slight quality trade-off
• Used in: Gemma, PaLM

2023: Grouped Query Attention (GQA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Groups of heads share K, V
• Middle ground
• 2-4× memory reduction
• Best quality/speed trade-off
• Used in: Llama 2/3, Mistral
```

### MHA vs GQA vs MQA Comparison

| Aspect | MHA | GQA (4 groups) | MQA |
|--------|-----|----------------|-----|
| KV Cache / Token | 1024 | 512 | 128 |
| Parameter Count | 524K | 393K | 266K |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Config:** 8 heads × 64 dim = 512 total dim

---

### RoPE (Rotary Position Embeddings)

**The Problem with Sinusoidal:**
```python
# Sinusoidal: Position added to embedding
embedding = token_embedding + position_embedding

# Problem: Position info can get lost in deep layers
# Problem: Absolute positions only
```

**RoPE Solution:**
```python
# RoPE: Rotate Q and K by position-dependent angle
Q_rotated = rotate(Q, position)
K_rotated = rotate(K, position)

# Attention score: Q_rotated · K_rotated
# This naturally encodes RELATIVE position!
```

**Visualization:**
```
Position 0:    [x, y] rotated by 0°       → [x, y]
Position 10:   [x, y] rotated by 10°      → [x', y']
Position 100:  [x, y] rotated by 100°     → [x'', y'']

Dot product: Q(pos=10) · K(pos=5)
           = |Q||K|cos(angle between them)
           = |Q||K|cos(10° - 5°)
           = |Q||K|cos(5°)

The attention score naturally depends on relative distance!
```

**Benefits:**
1. **Relative position encoding** - Dot product encodes distance
2. **Extrapolation** - Works on longer sequences than trained
3. **No parameters** - Unlike learned position embeddings
4. **Preserved through layers** - Unlike additive embeddings

---

### Modern Model Architecture (2024-2025)

```
┌────────────────────────────────────────────────────────────┐
│           MODERN LLM ARCHITECTURE (Llama 3, Mistral)       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input: Token IDs                                          │
│      ↓                                                     │
│  Token Embedding                                           │
│      ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Transformer Block (× N)                             │  │
│  │                                                      │  │
│  │  ┌───────────────────────────────────────────────┐   │  │
│  │  │ GQA Attention                               │   │  │
│  │  │ • Grouped Query Attention                   │   │  │
│  │  │ • RoPE position encoding                    │   │  │
│  │  │ • RMSNorm (Pre-norm)                        │   │  │
│  │  └───────────────────────────────────────────────┘   │  │
│  │      ↓                                               │  │
│  │  ┌───────────────────────────────────────────────┐   │  │
│  │  │ SwiGLU Feedforward                            │   │  │
│  │  │ • Swish activation + Gating                   │   │  │
│  │  │ • RMSNorm (Pre-norm)                        │   │  │
│  │  └───────────────────────────────────────────────┘   │  │
│  │                                                      │  │
│  └─────────────────────────────────────────────────────┘  │
│      ↓                                                     │
│  Output: Logits over vocabulary                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**vs Old Architecture (GPT-3):**
- ❌ MHA → ✅ GQA
- ❌ Learned position → ✅ RoPE
- ❌ LayerNorm → ✅ RMSNorm
- ❌ Post-norm → ✅ Pre-norm
- ❌ ReLU → ✅ SwiGLU

---

## Week 7 Exercises

### Exercise 1: Implement GQA
```python
class GroupedQueryAttention:
    """GQA with configurable number of KV heads."""
    
    def __init__(self, embed_dim, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        
        # Q projection: embed_dim → embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # K, V projection: embed_dim → num_kv_heads * head_dim
        kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(embed_dim, kv_dim)
        self.v_proj = nn.Linear(embed_dim, kv_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch, seq, _ = x.shape
        
        # Project
        q = self.q_proj(x)  # [batch, seq, embed_dim]
        k = self.k_proj(x)  # [batch, seq, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq, num_kv_heads * head_dim]
        
        # Reshape
        q = q.view(batch, seq, self.num_heads, self.head_dim)
        k = k.view(batch, seq, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim)
        
        # Repeat K, V to match number of query heads
        # [batch, seq, num_kv_heads, head_dim] → [batch, seq, num_heads, head_dim]
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        # Compute attention
        # ... standard attention computation ...
        
        return output

# Test: Compare MHA, GQA(4 groups), MQA
for num_kv in [8, 4, 1]:  # 8=MHA, 4=GQA, 1=MQA
    layer = GroupedQueryAttention(512, 8, num_kv)
    print(f"KV heads: {num_kv}, Params: {count_params(layer)}")
```

### Exercise 2: Implement RoPE
```python
import torch
import math

def precompute_rope_angles(dim, max_seq_len, theta=10000.0):
    """Precompute RoPE rotation angles."""
    # Frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Position × frequency
    positions = torch.arange(max_seq_len)
    angles = torch.outer(positions, freqs)  # [max_seq_len, dim/2]
    
    return torch.polar(torch.ones_like(angles), angles)  # e^(iθ)

def apply_rope(x, cos_sin):
    """Apply RoPE to input.
    
    x: [batch, seq, heads, dim]
    cos_sin: precomputed complex rotations
    """
    # Split into pairs
    x1 = x[..., ::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # View as complex
    x_complex = torch.view_as_complex(torch.stack([x1, x2], dim=-1))
    
    # Rotate
    x_rotated = x_complex * cos_sin[:x.size(1)]
    
    # Back to real
    x_out = torch.view_as_real(x_rotated)
    x1_rot, x2_rot = x_out[..., 0], x_out[..., 1]
    
    # Interleave
    result = torch.empty_like(x)
    result[..., ::2] = x1_rot
    result[..., 1::2] = x2_rot
    
    return result

# Compare with and without RoPE
q = torch.randn(1, 100, 8, 64)
k = torch.randn(1, 50, 8, 64)

# Without RoPE: absolute positions only
scores_no_rope = torch.matmul(q, k.transpose(-2, -1))

# With RoPE: relative positions
angles = precompute_rope_angles(64, 100)
q_rot = apply_rope(q, angles)
k_rot = apply_rope(k, angles)
scores_rope = torch.matmul(q_rot, k_rot.transpose(-2, -1))

# Observe: RoPE scores depend on relative distance
```

### Exercise 3: Research Model Configs
```python
# Look up these models and fill in the table:

models = {
    'Llama 2 7B': {
        'attention': 'GQA',
        'num_kv_heads': 4,
        'position': 'RoPE',
        'norm': 'RMSNorm',
        'activation': 'SwiGLU',
    },
    'Llama 3 8B': {
        'attention': '?',
        'num_kv_heads': '?',
        'position': '?',
        'norm': '?',
        'activation': '?',
    },
    'Mistral 7B': {
        'attention': '?',
        'num_kv_heads': '?',
        'position': '?',
        'norm': '?',
        'activation': '?',
    },
    'Gemma 7B': {
        'attention': '?',
        'num_kv_heads': '?',
        'position': '?',
        'norm': '?',
        'activation': '?',
    },
}

# Hint: Check Hugging Face model cards or papers
```

### Exercise 4: Upgrade Your Codebase
```python
# Modify your transformers/layers.py

# 1. Add GQA option to MultiHeadSelfAttention
class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, ...):
        if num_kv_heads is None:
            num_kv_heads = num_heads  # MHA
        self.num_kv_heads = num_kv_heads
        # ... implement GQA logic

# 2. Add RoPE option to TokenAndPositionEmbedding
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, ..., position_encoding='sinusoidal'):
        if position_encoding == 'rope':
            self.use_rope = True
            # No position embedding needed!
        # ... implement RoPE in attention instead

# 3. Try RMSNorm
class RMSNorm(keras.layers.Layer):
    """Root Mean Square Layer Normalization."""
    def call(self, x):
        return x * tf.rsqrt(tf.reduce_mean(x**2, axis=-1, keepdims=True) + 1e-6)
```

---

## Connection to Modern LLMs

### Llama 3 8B Configuration
```yaml
Architecture:
  Attention: GQA (4 KV heads for 32 query heads)
  Position: RoPE (theta=500000, extended to 128K)
  Norm: RMSNorm (Pre-norm)
  Activation: SwiGLU
  
Improvements over Llama 2:
  - Better tokenizer (128K vocab vs 32K)
  - Larger context (128K vs 4K)
  - Grouped attention (more efficient)
```

### Why These Choices?

| Choice | Why |
|--------|-----|
| GQA | Inference speed (less KV cache to load) |
| RoPE | Extrapolation to longer contexts |
| RMSNorm | Faster, simpler than LayerNorm |
| Pre-norm | Better gradient flow, more stable |
| SwiGLU | Better than ReLU/GLU in practice |

---

## Week 7 Completion Checklist

- [ ] Ran `week7_mqa_gqa_rope.py`
- [ ] Understand MQA memory savings (8×!)
- [ ] Know GQA as middle ground
- [ ] Understand RoPE rotation mechanism
- [ ] Know why RoPE beats sinusoidal
- [ ] Can name 2024 models and their architectures
- [ ] Completed at least 1 exercise

---

## Key Takeaways

### Attention Mechanisms
1. **MHA:** Best quality, high memory (legacy choice)
2. **MQA:** Fastest, 8× memory savings (Gemma)
3. **GQA:** Best trade-off (Llama, Mistral)

### Position Encoding
1. **Sinusoidal:** Classic, limited extrapolation
2. **Learned:** Good quality, can't extrapolate
3. **RoPE:** Best of both worlds (relative + extrapolation)

### Modern Stack (2024)
1. **GQA or MQA** for attention
2. **RoPE** for position
3. **RMSNorm** for normalization
4. **Pre-norm** architecture
5. **SwiGLU** activation

---

## Bridge to Week 8

Next week: **Normalization & Stability**
- RMSNorm vs LayerNorm
- Pre-norm vs Post-norm
- Training stability techniques

---

## Resources

### Reading
- "GQA: Training Generalized Multi-Query Transformer Models"
- "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- "Llama 2: Open Foundation and Fine-Tuned Chat Models" (architecture details)

### Papers
- "Root Mean Square Layer Normalization" (RMSNorm)
- "GLU Variants Improve Transformer" (SwiGLU)

### Code Study
- Your `attention/` directory
- Your `transformers/layers.py` position encoding
- Compare with Llama/Mistral implementations

---

## Week 7 Status

**Your progress:**
- ✅ MQA/GQA comparison and trade-offs
- ✅ RoPE mechanism and benefits
- ✅ 2024 model architecture trends
- ✅ Upgrade path for your codebase

**Ready for Week 8?**

If yes → See `../week8_normalization/`

If no → Implement GQA or RoPE in your codebase!

---

*Architecture evolution: Same concepts, better implementations.*
