# Week 9: Modern Training Recipes & Chinchilla Scaling ✅

**Focus:** Chinchilla scaling laws, compute-optimal training, modern hyperparameters

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand Chinchilla scaling laws (the 20:1 rule)
- ✅ Know why most models are undertrained
- ✅ Understand compute-optimal model/data trade-offs
- ✅ Know modern hyperparameter choices
- ✅ Be able to plan compute-optimal training runs

---

## Week 9 Scripts

### 1. `week9_chinchilla_scaling.py`
**Purpose:** Chinchilla scaling laws and modern training recipes

**What you'll learn:**
- Chinchilla 20:1 rule (tokens = 20 × params)
- Why GPT-3 was undertrained
- Optimal model/data trade-offs
- Modern hyperparameters (β2=0.95, not 0.999!)
- Common training mistakes

**Run it:**
```bash
cd learning_journey/week9_training_recipes
python week9_chinchilla_scaling.py
```

**Output:**
- Console: Scaling law explanations
- `chinchilla_scaling.png`: Visualizations

---

## Key Concepts

### The Chinchilla Paper

**Paper:** "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)

**The Core Finding:**
```
Most models are UNDERTRAINED!

GPT-3 (2020):
  • 175B parameters
  • 300B tokens
  • Ratio: 1,700 tokens per parameter
  • Status: Severely undertrained

Chinchilla (2022):
  • 70B parameters
  • 1.4T tokens
  • Ratio: 20,000 tokens per parameter
  • Status: Compute-optimal
```

**The Rule:**
```
Optimal tokens = 20 × parameters

Examples:
  1B model  → 20B tokens (optimal)
  7B model  → 140B tokens (optimal)
  70B model → 1.4T tokens (optimal)
```

---

### Why This Matters

**Scenario: You have $1M compute budget**

Option A (GPT-3 style):
```
Model: 175B params
Tokens: 300B
Result: Undertrained, suboptimal performance
```

Option B (Chinchilla style):
```
Model: 70B params
Tokens: 1.4T
Result: Compute-optimal, better performance!
```

**Key insight:** Smaller model + more data > Larger model + less data

---

### Real Model Analysis

| Model | Params (B) | Tokens (B) | Ratio | Status |
|-------|-----------|------------|-------|--------|
| GPT-3 | 175 | 300 | 1.7 | Undertrained |
| LaMDA | 137 | 168 | 1.2 | Undertrained |
| PaLM | 540 | 780 | 1.4 | Undertrained |
| Chinchilla | 70 | 1,400 | 20 | ✅ Optimal |
| LLaMA-7B | 7 | 1,024 | 146 | Overtrained |
| Mistral-7B | 7 | 7,000 | 1,000 | Overtrained |
| Llama 3 8B | 8 | 15,000 | 1,875 | Massively overtrained |

**Trend (2020 → 2024):**
- 2020: Huge models, little data (undertrained)
- 2022: Balanced approach (Chinchilla)
- 2024: Small models, massive data (high quality)

---

### Modern Hyperparameters

**Learning Rate:**
```python
# Scale down with model size
1B model:   LR = 3e-4
7B model:   LR = 3e-4  
70B model:  LR = 1.5e-4
175B model: LR = 1e-4

# Schedule: Warmup + Cosine decay
warmup_steps = total_steps * 0.01  # 1%
final_lr = base_lr * 0.1  # 10% of base
```

**Adam Optimizer:**
```python
# NOT the default!
optimizer = AdamW(
    lr=1e-4,
    betas=(0.9, 0.95),  # β2=0.95, not 0.999!
    weight_decay=0.1
)

# Why β2=0.95?
# • Faster adaptation to changing gradients
# • Better for non-stationary training (language modeling)
# • Modern standard since Chinchilla
```

**Batch Size:**
```python
# Modern standard: ~4M tokens per batch
# Constant across model sizes (evidence-based)

global_batch_tokens = 4_000_000  # 4M tokens

# For 512 sequence length:
sequences_per_batch = 4_000_000 / 512 ≈ 8,000 sequences
```

---

### Training Recipes

**Recipe 1: Small Research Model (1B)**
```yaml
Parameters: 1B
Tokens: 20B (optimal ratio)
Compute: ~120 PetaFLOPs
Cost: ~$5K
Use case: Research, prototyping
Time: ~1 week on small GPU cluster
```

**Recipe 2: Production Model (7B)**
```yaml
Parameters: 7B
Tokens: 2T (overtrained for quality)
Compute: ~84 ExaFLOPs
Cost: ~$100K
Use case: Production deployment, APIs
Time: ~1 month
Result: Mistral-7B class (beats GPT-3!)
```

**Recipe 3: Large Model (70B)**
```yaml
Parameters: 70B
Tokens: 1.4T (Chinchilla optimal)
Compute: ~600 ExaFLOPs
Cost: ~$1M
Use case: State-of-the-art
Time: ~2-3 months
Result: Chinchilla-class
```

---

## Week 9 Exercises

### Exercise 1: Calculate Optimal Training
```python
# Given a model size, calculate optimal training config

def calculate_optimal_training(params_billions):
    """Calculate optimal training configuration."""
    params = params_billions * 1e9
    
    # Chinchilla rule
    optimal_tokens = 20 * params
    
    # Compute budget
    compute_flops = 6 * params * optimal_tokens
    
    # Cost estimate (rough)
    # A100: ~$1/hour, ~300 TFLOPS
    # Cost per ExaFLOP: ~$100
    cost_estimate = compute_flops / 1e18 * 100
    
    # Training time on 256 A100s
    # Each A100: 300 TFLOPS = 0.3 PFLOPS
    # Cluster: 256 × 0.3 = 76.8 PFLOPS = 0.0768 EFLOPS
    time_hours = (compute_flops / 1e18) / 0.0768
    time_days = time_hours / 24
    
    return {
        'params_b': params_billions,
        'tokens_b': optimal_tokens / 1e9,
        'compute_e': compute_flops / 1e18,
        'cost_k': cost_estimate / 1000,
        'time_days': time_days
    }

# Test different sizes
for size in [1, 7, 70, 175]:
    config = calculate_optimal_training(size)
    print(f"\n{size}B Model:")
    print(f"  Tokens: {config['tokens_b']:.0f}B")
    print(f"  Compute: {config['compute_e']:.1f} ExaFLOPs")
    print(f"  Cost: ${config['cost_k']:.0f}K")
    print(f"  Time: {config['time_days']:.0f} days (256 A100s)")
```

### Exercise 2: Analyze Your Favorite Model
```python
# Look up any LLM and analyze it

models = {
    'GPT-3': {'params': 175, 'tokens': 300},
    'GPT-4': {'params': 1760, 'tokens': 13000},  # Estimated
    'Claude 3': {'params': None, 'tokens': None},  # Find the numbers
    'Gemini Pro': {'params': None, 'tokens': None},
    'Your favorite': {'params': None, 'tokens': None},
}

for name, specs in models.items():
    if specs['params'] and specs['tokens']:
        ratio = specs['tokens'] / specs['params']
        
        if ratio < 10:
            status = "🔴 Undertrained"
        elif ratio < 20:
            status = "🟡 Slightly under"
        elif ratio < 100:
            status = "🟢 Good"
        else:
            status = "🔵 Overtrained"
        
        print(f"{name}: {ratio:.1f} ratio - {status}")
```

### Exercise 3: Plan Your Training Run
```python
# Plan a compute-optimal training run

my_budget = 50000  # $50K
# Assume $100 per ExaFLOP
available_compute = my_budget / 100  # ExaFLOPs

# For compute-optimal:
# C = 6 × P × T
# T = 20 × P
# C = 6 × P × 20 × P = 120 × P²
# P = √(C / 120)

optimal_params = (available_compute / 120)**0.5
optimal_tokens = 20 * optimal_params

print(f"Budget: ${my_budget:,}")
print(f"Available compute: {available_compute:.0f} ExaFLOPs")
print(f"Optimal model: {optimal_params/1e9:.1f}B parameters")
print(f"Optimal tokens: {optimal_tokens/1e9:.0f}B tokens")
print(f"Training time: {available_compute/0.0768/24:.0f} days (256 A100s)")
```

### Exercise 4: Compare GPT-3 vs Chinchilla
```python
# Deep comparison of GPT-3 and Chinchilla approaches

gpt3 = {
    'name': 'GPT-3',
    'params': 175e9,
    'tokens': 300e9,
}

chinchilla = {
    'name': 'Chinchilla',
    'params': 70e9,
    'tokens': 1.4e12,
}

# Calculate compute
for model in [gpt3, chinchilla]:
    model['compute'] = 6 * model['params'] * model['tokens']
    model['ratio'] = model['tokens'] / model['params']

print("COMPARISON:")
print(f"GPT-3:      {gpt3['params']/1e9:.0f}B params × {gpt3['tokens']/1e9:.0f}B tokens = {gpt3['compute']/1e21:.1f} ZettaFLOPs")
print(f"Chinchilla: {chinchilla['params']/1e9:.0f}B params × {chinchilla['tokens']/1e9:.0f}B tokens = {chinchilla['compute']/1e21:.1f} ZettaFLOPs")
print()
print(f"Chinchilla uses {gpt3['compute']/chinchilla['compute']:.1f}× MORE compute")
print(f"But gets better performance per FLOP!")

# Key question: If GPT-3 had used Chinchilla ratio?
gpt3_optimal_tokens = 20 * gpt3['params']
gpt3_optimal_compute = 6 * gpt3['params'] * gpt3_optimal_tokens

print(f"\nIf GPT-3 used Chinchilla ratio:")
print(f"Tokens needed: {gpt3_optimal_tokens/1e12:.1f}T")
print(f"Compute needed: {gpt3_optimal_compute/1e21:.1f} ZettaFLOPs")
print(f"That's {gpt3_optimal_compute/gpt3['compute']:.0f}× more training!")
```

---

## Connection to Modern LLMs

### The Shift (2020 → 2024)

**2020 Approach (GPT-3):**
- Biggest model possible
- Train on available data
- Stop when loss plateaus

**2022 Approach (Chinchilla):**
- Balance model size and data
- Train with 20:1 ratio
- Optimize for compute efficiency

**2024 Approach (Llama 3, Mistral):**
- Smaller models (7B-70B)
- Massive data (1T-15T tokens)
- High quality, curated data
- Overtrain for best quality

### Why Overtrain Now?

```python
# Chinchilla says: tokens = 20 × params is compute-optimal
# But what if you want BEST quality, not compute-optimal?

# Modern models use:
tokens = 100 to 1000 × params  # 5× to 50× more than Chinchilla!

# Examples:
# Mistral-7B: 7,000B tokens (1000× ratio)
# Llama 3 8B: 15,000B tokens (1875× ratio)

# Why? Smaller model is:
# 1. Faster inference
# 2. Easier to deploy
# 3. Can train on more data with same compute
# Result: Better quality than larger, undertrained models
```

---

## Week 9 Completion Checklist

- [ ] Ran `week9_chinchilla_scaling.py`
- [ ] Understand the 20:1 rule
- [ ] Know why GPT-3 was undertrained
- [ ] Can calculate optimal training for any model size
- [ ] Know modern hyperparameters (β2=0.95!)
- [ ] Can plan a compute-optimal training run
- [ ] Completed at least 1 exercise

---

## Key Takeaways

### Chinchilla Scaling
1. **20:1 rule:** tokens = 20 × parameters
2. **Most models are undertrained** (GPT-3, PaLM, LaMDA)
3. **Smaller + more data > Larger + less data**
4. **Modern trend:** Even more data (100-1000× ratio)

### Modern Hyperparameters
1. **LR:** 1e-4 to 3e-4 (scale down with size)
2. **β2:** 0.95 (not 0.999!)
3. **Batch:** ~4M tokens (constant)
4. **Warmup:** 1-2% of steps

### Practical Recipes
1. **Research (1B):** 20B tokens, ~$5K, ~1 week
2. **Production (7B):** 2T tokens, ~$100K, ~1 month
3. **SOTA (70B):** 1.4T tokens, ~$1M, ~2-3 months

### Common Mistakes
1. Training too short (most common)
2. Learning rate too high
3. Using β2=0.999 (default, but wrong)
4. Not using LR decay
5. Ignoring loss spikes

---

## Bridge to Week 10

Next week: **Post-Training (RLHF)**
- Supervised Fine-Tuning (SFT)
- RLHF and PPO
- DPO (Direct Preference Optimization)
- How to align models to human preferences

---

## Resources

### Papers
- "Training Compute-Optimal Large Language Models" (Chinchilla)
- "Llama 2: Open Foundation and Fine-Tuned Chat Models" (training details)

### Blogs
- "Lessons from Training the Chinchilla Models" (DeepMind)
- "The Chinchilla Paper: A Guide" (various blogs)

### Tools
- Training cost calculators
- FLOP counting tools

---

## Week 9 Status

**Your progress:**
- ✅ Chinchilla 20:1 rule
- ✅ Compute-optimal trade-offs
- ✅ Real model analysis
- ✅ Modern hyperparameters
- ✅ Training cost estimation

**Ready for Week 10?**

If yes → See `../week10_rlhf/`

If no → Plan your own training run with Chinchilla rules!

---

*The best model is not the biggest—it's the best trained.*
