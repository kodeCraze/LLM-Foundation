# LLM Foundations Learning Journey 🚀

**Your 12-Week Path to Deep LLM Understanding**

> *"Fear comes from the unknown. Each week, you're making the unknown known."*

---

## 📁 Organized Structure

```
learning_journey/
├── README.md                    ← You are here
├── week1_foundations/
│   ├── README.md               ← Week 1 summary
│   ├── week1_attention_visualization.py
│   ├── week1_model_trace.py
│   ├── week1_positional_encoding_viz.py
│   └── week1_tokenizer.json
│
├── week2_tokenization/
│   ├── README.md               ← Week 2 summary
│   ├── week2_bpe_tokenizer.py
│   ├── week2_embedding_visualization.py
│   └── week2_tokenizer.json
│
├── week3_training/
│   ├── README.md               ← Week 3 summary
│   ├── week3_training_loop.py
│   └── training_progress.png   ← Generated
│
├── week4_optimization/
│   ├── README.md               ← Week 4 summary
│   ├── week4_optimizers.py
│   ├── optimizer_comparison.png    ← Generated
│   ├── learning_rate_effects.png   ← Generated
│   ├── lr_schedules.png            ← Generated
│   └── gradient_clipping.png       ← Generated
│
├── week5_scale/
│   ├── README.md               ← Week 5 summary
│   └── week5_memory_and_scaling.py
│
├── week6_distributed/
│   ├── README.md               ← Week 6 summary
│   └── week6_distributed_training.py
│
├── week7_advanced_architecture/
│   ├── README.md               ← Week 7 summary
│   ├── week7_mqa_gqa_rope.py
│   ├── attention_comparison.png   ← Generated
│   └── rope_visualization.png     ← Generated
│
├── week8_normalization/
│   ├── README.md               ← Week 8 summary
│   ├── week8_normalization.py
│   ├── normalization_comparison.png  ← Generated
│   ├── gradient_flow.png            ← Generated
│   └── training_stability.png       ← Generated
│
├── week9_training_recipes/
│   ├── README.md               ← Week 9 summary
│   ├── week9_chinchilla_scaling.py
│   └── chinchilla_scaling.png       ← Generated
│
├── week10_rlhf/
│   ├── README.md               ← Week 10 summary
│   ├── week10_post_training.py
│   └── post_training_phases.png    ← Generated
│
└── [more weeks coming...]
```

---

## 🎯 Your Learning Profile

| Aspect | Your Choice |
|--------|-------------|
| **Current Level** | Know ML basics, new to transformers |
| **Learning Style** | Visual/intuitive first |
| **Goal** | Personal mastery and confidence |
| **Time Commitment** | 20+ hours/week (intensive) |

---

## 📚 Week-by-Week Roadmap

### ✅ Week 1: Foundation & Intuition
**Status:** COMPLETE

**Scripts:**
- `week1_attention_visualization.py` - Attention heatmaps & concepts
- `week1_model_trace.py` - Tensor tracing through your code
- `week1_positional_encoding_viz.py` - Position encoding at layers.py:126

**Key Learnings:**
- Attention = weighted averaging with learned importance
- Causal masking prevents peeking at future
- Your code creates a 53K parameter functional model
- Sinusoidal encoding gives unique position fingerprints

**Checkpoint:** Can you explain attention to a non-technical friend?

---

### ✅ Week 2: Tokenization & Embeddings
**Status:** COMPLETE

**Scripts:**
- `week2_bpe_tokenizer.py` - Build BPE from scratch
- `week2_embedding_visualization.py` - Word embeddings in 2D space

**Key Learnings:**
- BPE: Character-level → subword tokens via frequency merging
- Embeddings: Similar words cluster in vector space
- Modern LLMs use 30K-100K+ token vocabularies

**Checkpoint:** Can you tokenize "unseenword" with your trained BPE?

---

### ✅ Week 3: Training & The Forward Pass
**Status:** COMPLETE

**Scripts:**
- `week3_training_loop.py` - Train tiny transformer on toy data

**Key Learnings:**
- 4-step training loop: Forward → Loss → Backward → Update
- Cross-entropy loss measures "surprise" at correct answer
- Gradient descent = walking downhill on loss landscape
- Overfitting small data is essential debugging tool

**Checkpoint:** Can you explain backpropagation to a junior developer?

---

### ✅ Week 4: Optimization Deep Dive
**Status:** COMPLETE

**Scripts:**
- `week4_optimizers.py` - Compare optimizers & LR strategies

**Key Learnings:**
- AdamW is the modern standard (GPT, Llama use it)
- Learning rate: Goldilocks problem (not too small, not too big)
- Warmup + Cosine decay: The modern schedule
- Gradient clipping: Essential for deep transformers

**Checkpoint:** Can you tune a model's LR and optimizer?

---

### ✅ Week 5: Memory & Compute Efficiency
**Status:** COMPLETE

**Scripts:**
- `week5_memory_and_scaling.py` - Memory profiling and scaling

**Key Learnings:**
- Training uses 4× model size in memory
- Gradient accumulation simulates large batches
- Mixed precision: 50% memory, 2-3× speed
- Batch size affects convergence and generalization

**Checkpoint:** Can you calculate memory for any model size?

---

### ✅ Week 6: Distributed Training
**Status:** COMPLETE

**Scripts:**
- `week6_distributed_training.py` - DP, MP, pipeline, ZeRO

**Key Learnings:**
- Data parallelism: Simple, most common
- Model parallelism: For oversized models
- Pipeline parallelism: Better GPU utilization
- ZeRO: 8× memory reduction for large models!
- Scaling efficiency: 2-8 GPUs >90% efficient

**Checkpoint:** Can you design training strategy for 10B parameter model?

---

### ✅ Week 7: Advanced Architecture
**Status:** COMPLETE

**Scripts:**
- `week7_mqa_gqa_rope.py` - MQA, GQA, RoPE modern standards

**Key Learnings:**
- MQA: 8× KV cache reduction, fastest inference
- GQA: Best quality/speed trade-off (Llama, Mistral)
- RoPE: Relative position encoding, extrapolates to longer sequences
- Modern stack: GQA + RoPE + RMSNorm + Pre-norm

**Checkpoint:** Can you upgrade your attention to GQA?

---

### ✅ Week 8: Normalization & Stability
**Status:** COMPLETE

**Scripts:**
- `week8_normalization.py` - RMSNorm, Pre-norm, stability

**Key Learnings:**
- RMSNorm is 30-40% faster than LayerNorm (no mean subtraction)
- Pre-norm enables training 100+ layer models (stable gradients)
- Modern stack: RMSNorm + Pre-norm + SwiGLU + GQA + RoPE
- Stability techniques: residual, clipping, warmup

**Checkpoint:** Can you upgrade your architecture to 2024 standards?

---

### ✅ Week 9: Modern Training Recipes
**Status:** COMPLETE

**Scripts:**
- `week9_chinchilla_scaling.py` - Chinchilla scaling laws

**Key Learnings:**
- 20:1 rule (tokens = 20 × parameters)
- GPT-3 was severely undertrained (1.7K vs 20K ratio)
- Smaller model + more data > larger model + less data
- Modern hyperparameters (β2=0.95, not 0.999!)

**Checkpoint:** Can you plan a compute-optimal training run?

---

### 🔄 Week 10: Post-Training (RLHF)
**Status:** READY

**Scripts:**
- `week10_post_training.py` - RLHF, DPO, alignment

**Key Learnings:**
- Three-stage pipeline: Pretrain → SFT → RLHF/DPO
- RLHF: Reward Model + PPO (complex, 4 models)
- DPO: Direct preference optimization (simpler, 2 models, better)
- 2024 recommendation: Use DPO over RLHF+PPO

**Checkpoint:** Can you explain why DPO is simpler than RLHF?

---

### ⏳ Week 11: Inference Optimization
**Coming soon**

- KV-cache (speed up generation)
- Quantization (reduce memory)
- Speculative decoding (faster inference)

---

### ⏳ Week 12: 2026 Frontiers
**Coming soon**

- Mixture of Experts (MoE)
- Test-time compute scaling
- Multimodal LLMs
- Reasoning models (o1-style)

---

## 🚀 Quick Start

### Run Week 1 (Foundation)
```bash
cd week1_foundations
python week1_attention_visualization.py
python week1_positional_encoding_viz.py
python week1_model_trace.py
```

### Run Week 2 (Tokenization)
```bash
cd week2_tokenization
python week2_bpe_tokenizer.py
python week2_embedding_visualization.py
```

---

## 🧠 Fear-Busting Mantras

When you feel overwhelmed, remember:

1. **"Every expert was once a beginner"**
   - The people who built GPT-4 started exactly where you are

2. **"Complexity is just layers of simplicity"**
   - Transformers = attention + FFN, repeated. Each piece is simple.

3. **"Debugging is learning"**
   - Every error message teaches you something

4. **"Your model doesn't need to be GPT-4"**
   - A 2-layer transformer you understand > black-box giant

5. **"Your codebase is already 70% there"**
   - You have working components. The journey is understanding them.

---

## 📊 Progress Tracker

| Week | Topic | Status | Key Script | Output |
|------|-------|--------|------------|--------|
| 1 | Foundations | ✅ Done | `week1_model_trace.py` | 53K param model |
| 2 | Tokenization | ✅ Done | `week2_bpe_tokenizer.py` | Working BPE |
| 3 | Training Loop | ✅ Done | `week3_training_loop.py` | 100% accuracy |
| 4 | Optimization | ✅ Done | `week4_optimizers.py` | AdamW, schedules |
| 5 | Scale & Memory | ✅ Done | `week5_memory_and_scaling.py` | Profiling, accumulation |
| 6 | Distributed | ✅ Done | `week6_distributed_training.py` | DP, MP, ZeRO |
| 7 | Advanced Arch | ✅ Done | `week7_mqa_gqa_rope.py` | MQA, GQA, RoPE |
| 8 | Normalization | ✅ Done | `week8_normalization.py` | RMSNorm, Pre-norm |
| 9 | Chinchilla | ✅ Done | `week9_chinchilla_scaling.py` | 20:1 scaling law |
| 10 | RLHF/DPO | 🔄 Current | `week10_post_training.py` | Alignment |
| 11 | Inference | ⏳ Pending | TBD | KV-cache |
| 12 | Frontiers | ⏳ Pending | TBD | 2026 techniques |

---

## 🔗 Connection to Your Codebase

Your existing project structure:

```
ai_foundations/
├── attention/      → Week 7-9 focus (MQA, GQA, RoPE)
├── embeddings/     → Week 2 focus (visualization)
├── feedback/       → Week 10 focus (RLHF concepts)
├── generation/     → Week 11 focus (sampling)
├── machine_learning/ → Week 4-6 focus (optimization)
├── tokenization/   → Week 2 focus (BPE, your tokenizers)
├── training/       → Week 3-6 focus (loops, callbacks)
├── transformers/   → Week 1 focus (layers.py)
│   └── layers.py   ← Where your cursor was!
└── utils/          → Support utilities
```

---

## 📖 How to Use This Journey

### For Visual/Intuitive Learning (That's You!)

1. **Start with the visualization scripts**
   - Run the `.py` files
   - Study the generated `.png` images
   - Understand concepts before code

2. **Trace through the code**
   - Use the model trace scripts
   - See tensor shapes at each step
   - Connect visuals to implementation

3. **Read the READMEs**
   - Each week has its own README
   - Contains theory, exercises, connections
   - Checkpoint questions to test understanding

4. **Explore your codebase**
   - Compare journey scripts with your code
   - Find the parallel implementations
   - Understand every line

5. **Do the exercises**
   - Modify parameters
   - Train on new data
   - Break things and fix them

---

## 🎓 Success Metrics

By the end of this journey, you will:

- ✅ Understand every line in your `layers.py` and why it's there
- ✅ Implement any transformer component from scratch
- ✅ Debug training issues by reading loss curves
- ✅ Confidently discuss LLM architecture with practitioners
- ✅ Have working models you trained yourself
- ✅ Know what makes 2026 models different from 2020 models

---

## 📞 When You're Stuck

### Debugging Tips
1. Add `print(tensor.shape)` everywhere
2. Use `model.summary()` to see architecture
3. Check tensor dtypes (mismatches cause errors)
4. Verify batch dimensions match

### Understanding Tips
1. Draw the architecture on paper
2. Trace through with a tiny example (batch=1, seq_len=4)
3. Compare your code with working examples
4. Ask: "What would happen if I removed this line?"

### Motivation Tips
1. Celebrate small wins ("I understood attention!")
2. Take breaks when frustrated
3. Re-read old code to see progress
4. Remember: confusion is the feeling of learning

---

## 🎯 Next Steps

### If You're in Week 1:
1. Run all 3 week1 scripts
2. Study the generated PNG files
3. Read `week1_foundations/README.md`
4. Answer the checkpoint questions
5. Move to Week 2

### If You're in Week 2:
1. Run `week2_bpe_tokenizer.py`
2. Study the training output
3. Run `week2_embedding_visualization.py`
4. Study the embedding space plot
5. Read `week2_tokenization/README.md`
6. Do the exercises
7. Explore your `tokenization/` directory

---

## 📝 Weekly Check-in Questions

Each week, ask yourself:

- [ ] Can I draw this week's concept on a whiteboard?
- [ ] Did I write at least 50 lines of code this week?
- [ ] Can I explain what I learned to someone else?
- [ ] What was confusing? (This is your learning edge)
- [ ] Did I explore the corresponding directory in my codebase?

---

## 🌟 Remember

> **The goal isn't to rush through—it's to truly understand.**

Your existing codebase is excellent. The scripts in this journey are here to help you:
1. Build intuition through visualization
2. Connect concepts to your actual code
3. Practice with hands-on exercises
4. Track progress and celebrate wins

**You've got this! 🚀**

---

*Start with Week 1 → Then Week 2 → Ask questions anytime*
