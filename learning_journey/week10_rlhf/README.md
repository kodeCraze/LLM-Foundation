# Week 10: Post-Training - RLHF & DPO ✅

**Focus:** Aligning LLMs with human preferences - RLHF, PPO, DPO

---

## Learning Objectives

By the end of this week, you will:
- ✅ Understand the three-stage training pipeline (Pretrain → SFT → RLHF)
- ✅ Know how RLHF works (Reward Model + PPO)
- ✅ Understand DPO (modern alternative to RLHF)
- ✅ Know why DPO is often preferred in 2024
- ✅ Understand preference data and human feedback

---

## Week 10 Scripts

### 1. `week10_post_training.py`
**Purpose:** Understanding RLHF and DPO

**What you'll learn:**
- SFT (Supervised Fine-Tuning)
- RLHF pipeline (Reward Model + PPO)
- DPO (Direct Preference Optimization)
- Comparison of RLHF vs DPO
- Preference data examples

**Run it:**
```bash
cd learning_journey/week10_rlhf
python week10_post_training.py
```

**Output:**
- Console: Detailed explanations
- `post_training_phases.png`: Training phases visualization

---

## Key Concepts

### The Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  LLM DEVELOPMENT PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STAGE 1: PRETRAINING                                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━                                        │
│  • Data: Internet text (hundreds of billions of tokens)         │
│  • Goal: Learn language, world knowledge, reasoning               │
│  • Duration: Weeks/months on 1000s of GPUs                      │
│  • Output: "Base model" (can complete text, not conversational)   │
│                                                                 │
│  STAGE 2: SUPERVISED FINE-TUNING (SFT)                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                          │
│  • Data: Instruction-response pairs (10K-100K examples)           │
│  • Goal: Learn to follow instructions, be helpful               │
│  • Duration: Hours to days                                       │
│  • Output: "Instruct model" (can follow instructions)           │
│                                                                 │
│  STAGE 3: RLHF / DPO                                           │
│  ━━━━━━━━━━━━━━━━━━━━━                                        │
│  • Data: Human preferences (comparisons)                          │
│  • Goal: Align with human values (helpful, harmless, honest)      │
│  • Duration: Hours to days                                       │
│  • Output: "Aligned model" (safe, helpful assistant)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### RLHF (Reinforcement Learning from Human Feedback)

**The Goal:** Make models helpful, harmless, and honest

**Three Components:**

1. **Reward Model (RM)**
   ```python
   # Input: (prompt, response)
   # Output: scalar score (how good is this response?)
   
   reward_model(prompt, winning_response) → 8.5
   reward_model(prompt, losing_response) → 3.2
   ```
   
   - Trained on human comparison data
   - Learns to predict human preferences
   - Becomes the "objective function" for RL

2. **Policy (Language Model)**
   - The model being trained
   - Generates responses
   - Optimized to maximize reward

3. **PPO (Proximal Policy Optimization)**
   ```python
   # RL algorithm to train policy
   
   for batch in data:
       # 1. Generate responses
       responses = policy.generate(prompts)
       
       # 2. Score with reward model
       rewards = reward_model(prompts, responses)
       
       # 3. Update policy with PPO
       # - Maximize reward
       # - But don't change too much (clip)
       # - Stay close to SFT model (KL penalty)
       policy.update_ppo(rewards, kl_penalty)
   ```

**Challenges:**
- Complex (4 models in memory)
- Unstable (hyperparameter sensitive)
- Reward hacking (game the reward model)

---

### DPO (Direct Preference Optimization)

**The Innovation:** Skip RL entirely!

```python
# DPO Loss
# Given: (prompt, winning_response, losing_response)

loss = -log σ(β × [log(π_win/π_ref) - log(π_lose/π_ref)])

# Where:
#   π_win = policy probability of winning response
#   π_lose = policy probability of losing response  
#   π_ref = reference (SFT) model probability
#   β = temperature parameter

# Simplified:
# • Increase probability of winning response
# • Decrease probability of losing response
# • β controls divergence from reference model
```

**Why DPO is Better:**

| Aspect | RLHF + PPO | DPO |
|--------|-----------|-----|
| Models needed | 4 (policy, ref, reward, critic) | 2 (policy, ref) |
| Complexity | High | Low |
| Stability | Can be unstable | Very stable |
| Speed | Slow | 2× faster |
| Memory | 4× model size | 2× model size |
| Performance | Good | Often better |

**2024 Recommendation:** Use DPO instead of RLHF+PPO

---

### Preference Data

**What humans label:**
```
Prompt: "How do I make a bomb?"

Response A: "I can't help with that. Making explosives is illegal..."
Response B: "Here's how to make a simple bomb: [detailed instructions]"

Label: A is better (safety)
```

```
Prompt: "Explain photosynthesis"

Response A: "Plants use sunlight to make food through a process called..."
Response B: "Photosynthesis is when plants eat sun."

Label: A is better (helpfulness)
```

**Criteria:**
- **Helpful:** Accurate, comprehensive, follows instructions
- **Harmless:** Doesn't help with dangerous/illegal things
- **Honest:** Doesn't hallucinate, admits uncertainty

**Data Collection:**
- Human annotators compare responses
- Usually 2-4 responses per prompt
- Need diversity (different prompt types)
- Quality > Quantity (10K-100K comparisons)

---

## Week 10 Exercises

### Exercise 1: Understand the Pipeline
```python
# For each stage, identify:

stages = {
    'Pretraining': {
        'data': '__________',
        'goal': '__________',
        'duration': '__________',
        'output': '__________',
    },
    'SFT': {
        'data': '__________',
        'goal': '__________',
        'duration': '__________',
        'output': '__________',
    },
    'RLHF': {
        'data': '__________',
        'goal': '__________',
        'duration': '__________',
        'output': '__________',
    },
}

# Fill in the blanks
```

### Exercise 2: Implement Simple Reward Model
```python
class SimpleRewardModel(nn.Module):
    """Reward model that scores responses."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.score_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, prompt_tokens, response_tokens):
        # Concatenate prompt + response
        full_tokens = torch.cat([prompt_tokens, response_tokens], dim=1)
        
        # Get hidden states from base model
        outputs = self.base(full_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # Last layer
        
        # Use last token's hidden state for scoring
        last_token_hidden = hidden[:, -1, :]
        
        # Score
        score = self.score_head(last_token_hidden)
        
        return score

# Training:
# loss = -log σ(score_win - score_lose)
```

### Exercise 3: Implement DPO Loss
```python
def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """Compute DPO loss.
    
    batch contains:
    - prompts: list of prompt token IDs
    - winning: list of winning response token IDs
    - losing: list of losing response token IDs
    """
    
    # Get log probabilities from policy model
    policy_win_logprobs = get_logprobs(policy_model, batch['prompts'], batch['winning'])
    policy_lose_logprobs = get_logprobs(policy_model, batch['prompts'], batch['losing'])
    
    # Get log probabilities from reference model
    with torch.no_grad():
        ref_win_logprobs = get_logprobs(ref_model, batch['prompts'], batch['winning'])
        ref_lose_logprobs = get_logprobs(ref_model, batch['prompts'], batch['losing'])
    
    # Compute log ratios
    policy_logratios = policy_win_logprobs - policy_lose_logprobs
    ref_logratios = ref_win_logprobs - ref_lose_logprobs
    
    # DPO loss
    logits = beta * (policy_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    return loss

def get_logprobs(model, prompts, responses):
    """Get log probabilities of responses given prompts."""
    # Implementation: forward pass, extract per-token logprobs
    pass
```

### Exercise 4: Research Alignment Methods
```python
# Look up these alignment methods:

methods = {
    'RLHF': {
        'paper': 'InstructGPT (2022)',
        'components': ['Reward Model', 'PPO', 'Reference Model'],
        'pros': ['Flexible reward', 'Proven at scale'],
        'cons': ['Complex', 'Unstable'],
    },
    'DPO': {
        'paper': 'Direct Preference Optimization (2023)',
        'components': ['Policy', 'Reference'],
        'pros': ['Simple', 'Stable', 'Fast'],
        'cons': ['Less flexible', 'Implicit reward'],
    },
    'RLAIF': {
        'paper': 'Constitutional AI (2023)',
        'components': ['AI feedback', 'Critique', 'Revision'],
        'pros': ['Scalable', 'No humans needed'],
        'cons': ['AI may be biased', 'Quality depends on AI'],
    },
    'KTO': {
        'paper': 'Kahneman-Tversky Optimization (2024)',
        'components': ['Binary feedback', 'Loss aversion'],
        'pros': ['Simpler data', 'Human-like preferences'],
        'cons': ['Newer method', 'Less tested'],
    },
}

# Questions:
# 1. Which method does Llama 2 use?
# 2. Which method does Mistral use?
# 3. Which would you use for a new project?
```

### Exercise 5: Analyze Your Model
```python
# Check if your codebase has any alignment components

# Look for:
# 1. SFT datasets or training code
# 2. Reward model implementation
# 3. PPO or DPO training code
# 4. Preference data handling

# If not present, consider:
# - Would SFT be easy to add?
# - Would DPO be easier than RLHF?
```

---

## Connection to Modern LLMs

### Llama 2 Training Pipeline

```
1. Pretraining
   • 2 trillion tokens
   • Learn language and knowledge

2. SFT (Supervised Fine-Tuning)
   • 27,540 high-quality instruction examples
   • Carefully curated and verified
   • Learn instruction following

3. RLHF
   • Human preference data: ~1M comparisons
   • Reward model trained on preferences
   • PPO optimization (iterative)
   • Ghost attention (special technique)
   
Result: Llama 2 Chat
```

### Comparison: Base vs Instruct vs Aligned

| Model Type | Use Case | Capabilities |
|------------|----------|--------------|
| Base | Text completion | Can continue text, not conversational |
| Instruct (SFT) | Following instructions | Can follow commands, may be unsafe |
| Aligned (RLHF/DPO) | Assistant/chat | Helpful, harmless, honest |

---

## Week 10 Completion Checklist

- [ ] Ran `week10_post_training.py`
- [ ] Understand three-stage pipeline
- [ ] Know how RLHF works (Reward Model + PPO)
- [ ] Understand DPO and its advantages
- [ ] Can compare RLHF vs DPO
- [ ] Know what preference data looks like
- [ ] Completed at least 1 exercise

---

## Key Takeaways

### Training Pipeline
1. **Pretrain:** Learn language (hundreds of billions of tokens)
2. **SFT:** Learn to follow instructions (10K-100K examples)
3. **RLHF/DPO:** Align with human preferences (comparisons)

### RLHF
1. **Reward Model:** Predicts human preferences
2. **PPO:** RL algorithm to optimize
3. **KL Penalty:** Prevents drift from SFT
4. **Complex:** 4 models, unstable

### DPO (2024 Standard)
1. **Direct:** Train on preferences
2. **Simple:** No reward model, no RL
3. **Fast:** 2× speed of RLHF
4. **Better:** Often outperforms RLHF

### Preference Data
1. **Comparisons:** A vs B (which is better?)
2. **Criteria:** Helpful, harmless, honest
3. **Quality:** Careful labeling over quantity

---

## Bridge to Week 11

Next week: **Inference Optimization**
- KV-cache (speed up generation)
- Quantization (reduce memory)
- Speculative decoding (faster inference)

---

## Resources

### Papers
- "Training language models to follow instructions" (InstructGPT)
- "Direct Preference Optimization" (DPO, 2023)
- "Llama 2: Open Foundation and Fine-Tuned Chat Models"

### Blogs
- "Illustrating RLHF" (Hugging Face)
- "DPO vs PPO" (various comparisons)

### Code
- TRL (Transformer Reinforcement Learning) library
- Alignment Handbook (Hugging Face)

---

## Week 10 Status

**Your progress:**
- ✅ Three-stage pipeline understanding
- ✅ RLHF mechanism (Reward Model + PPO)
- ✅ DPO advantages
- ✅ Preference data
- ✅ Modern alignment methods

**Ready for Week 11?**

If yes → See `../week11_inference/`

If no → Research which alignment method your favorite model uses!

---

*Alignment is what turns a language model into a helpful assistant.*
